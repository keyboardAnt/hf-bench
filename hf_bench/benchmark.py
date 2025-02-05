# NOTE: These lines set up the `HF_HOME` cache directory. Ensure that they run before other imports.
#       We mark `# noqa: E402` to avoid ruff complaining about the order of imports.
from dotenv import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import gc  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from dataclasses import astuple, dataclass  # noqa: E402
from datetime import datetime  # noqa: E402
from threading import Thread  # noqa: E402
from typing import List, Optional  # noqa: E402

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.generation.streamers import BaseStreamer  # noqa: E402

import wandb  # noqa: E402
from hf_bench.config import (  # noqa: E402
    DatasetConfig,
    ExperimentConfig,
    experiment_configs,
)
from hf_bench.utils import (  # noqa: E402
    log_hardware_info,
    login_to_hf,
    login_wandb,
    set_hf_cache_env,
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Generation Script")
    parser.add_argument(
        "--experiment_config",
        default="default",
        type=str,
        help="The experiment config to run. For example, `llama70b-it`.",
    )
    parser.add_argument(
        "--num_of_examples",
        default=30,
        type=int,
        help="The number of examples from the dataset to run.",
    )
    parser.add_argument(
        "--results_checkpoint_dirpath",
        default="",
        type=str,
        help="The directory for storing benchmark results. If provided, the benchmark will resume from the latest checkpoint in this directory.",
    )
    return parser.parse_args()


def get_results_dirpath(root_dirpath: str) -> str:
    """Create a timestamped output directory for storing benchmark results."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    commit_hash = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("utf-8")
        .strip()
    )
    dirpath = f"{root_dirpath}/{timestamp}_{commit_hash}"
    os.makedirs(dirpath, exist_ok=True)
    return dirpath


def clear_memory():
    """
    Clears Python and GPU memory to ensure a fresh start for experiments.
    Includes additional cleanup steps for more thorough memory management.
    """
    # Run garbage collection multiple times to handle circular references
    for _ in range(3):
        gc.collect()

    # Clear CUDA memory if available
    if torch.cuda.is_available():
        # Explicitly empty CUDA cache
        torch.cuda.empty_cache()

        # Force synchronization of CUDA threads
        torch.cuda.synchronize()

        # Collect inter-process CUDA memory
        torch.cuda.ipc_collect()

        # Print memory stats for debugging (optional)
        for i in range(torch.cuda.device_count()):
            print(
                f"GPU {i} memory allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f}GB",
                flush=True,
            )
            print(
                f"GPU {i} memory cached: {torch.cuda.memory_reserved(i) / 1e9:.2f}GB",
                flush=True,
            )

    print(
        "Memory cleared: Python memory garbage collected and GPU cache emptied.",
        flush=True,
    )


# ------------------------------------------------------------------------------
# Streaming
# ------------------------------------------------------------------------------


class IdsIteratorStreamer(BaseStreamer):
    """
    A custom streamer that yields token IDs instead of decoded text.
    Skips the first `prompt_len` tokens, so you don't stream the prompt.
    """

    def __init__(self, prompt_len: int = 0):
        super().__init__()
        self.prompt_len = prompt_len
        self.tokens_seen = 0
        self.buffer = []
        self.is_finished = False

    def put(self, token_ids: Optional[torch.Tensor]):
        """
        Called by the generate() method whenever new tokens become available.

        Args:
            token_ids (Optional[torch.Tensor]): A tensor containing newly generated token IDs.
                If None, it signals that generation has ended.
        """
        if token_ids is None:
            # End of generation
            self.is_finished = True
        else:
            # If token_ids has shape (1, N), flatten it to shape (N,)
            if token_ids.dim() == 2 and token_ids.shape[0] == 1:
                token_ids = token_ids.squeeze(0)
            for tid in token_ids:
                # Skip the first `prompt_len` tokens
                if self.tokens_seen < self.prompt_len:
                    self.tokens_seen += 1
                else:
                    self.buffer.append(tid)
                    self.tokens_seen += 1

    def end(self):
        """Signals that generation is complete."""
        self.is_finished = True

    def __iter__(self):
        """
        Yields token IDs as they become available.
        """
        while not self.is_finished or self.buffer:
            if self.buffer:
                yield self.buffer.pop(0)
            else:
                # Avoid busy waiting
                time.sleep(0.01)


# ------------------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------------------


@dataclass
class Result:
    """
    A class to store the results of a generation experiment.
    """

    tok_ids_prompt: List[int]
    tok_ids_new: List[int]
    total_gen_time_s: float
    ttft_s: float
    tpot_s: float


@dataclass
class ResultsTableRow:
    target: str
    dataset_path: str
    dataset_name: str
    dataset_split: str
    num_of_examples: int
    drafter: str
    temperature: float
    example_id: int
    new_toks: int
    ttft_ms: float
    tpot_ms: float
    out_toks_per_sec: float

    @classmethod
    def from_experiment_config_and_result(
        cls,
        target: str,
        dataset_path: str,
        dataset_name: str,
        dataset_split: str,
        num_of_examples: int,
        drafter: str,
        temperature: float,
        example_id: int,
        result: Result,
    ) -> "ResultsTableRow":
        return cls(
            target=target,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            num_of_examples=num_of_examples,
            drafter=drafter,
            example_id=example_id,
            temperature=temperature,
            new_toks=len(result.tok_ids_new),
            ttft_ms=result.ttft_s * 1000,
            tpot_ms=result.tpot_s * 1000,
            out_toks_per_sec=1 / result.tpot_s,
        )


# ------------------------------------------------------------------------------
# Model Handling
# ------------------------------------------------------------------------------


class HFModel:
    """
    Lightweight class to wrap a Hugging Face model and tokenizer for convenience.
    """

    def __init__(
        self, model_name: str, device_map: str = "auto", torch_dtype=torch.float16
    ):
        """
        Load a model and tokenizer from the Hugging Face Hub.
        """
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device_map, torch_dtype=torch_dtype, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __del__(self):
        """
        Clean up model resources when the object is deleted.
        """
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        clear_memory()

    def generate_text(
        self, prompt: str, do_sample: bool, max_new_tokens: int = 512, **kwargs
    ) -> Result:
        """
        Generate text from the underlying model, measuring detailed latency metrics.

        Parameters:
            prompt (str): The input text to generate from.
            do_sample (bool): Whether to sample or use greedy decoding.
            max_new_tokens (int): Maximum number of tokens to generate.
            **kwargs: Additional arguments for the generation method.
        """
        # Clear any cached memory before starting
        clear_memory()

        # Tokenize the input prompt and move it to the model's device
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
        inputs["input_ids"] = inputs["input_ids"].to(
            self.model.device, dtype=torch.int64
        )

        prompt_len = inputs["input_ids"].shape[1]

        # Create a streamer for raw token IDs (instead of TextIteratorStreamer)
        streamer = IdsIteratorStreamer(prompt_len=prompt_len)

        # Handle the attention mask to ensure valid memory alignment
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(inputs["input_ids"], dtype=torch.int64)
        attention_mask = attention_mask.to(self.model.device)

        generation_kwargs = dict(
            inputs=inputs["input_ids"],
            attention_mask=attention_mask,
            cache_implementation="offloaded",
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            streamer=streamer,
            return_dict_in_generate=False,  # Return only the generated sequences
            output_scores=False,
            output_hidden_states=False,
            output_attentions=False,
            **kwargs,
        )

        # Warmup
        for _ in range(2):
            self.model.generate(**generation_kwargs)

        # Reset the streamer
        streamer = IdsIteratorStreamer(prompt_len=prompt_len)
        generation_kwargs["streamer"] = streamer

        # Create thread with daemon=True to ensure it's cleaned up
        thread = Thread(
            target=self.model.generate, kwargs=generation_kwargs, daemon=True
        )
        start_time = time.time()
        thread.start()

        new_token_ids_tensors = []
        time_to_first_token = None

        for chunk_of_ids in streamer:
            # Record TTFT if it's the very first token(s)
            if time_to_first_token is None:
                time_to_first_token = time.time() - start_time

            # chunk_of_ids might be shape=() (a single scalar) or shape=(n,) (n tokens)
            # -> force it to be at least shape=(1,):
            if chunk_of_ids.dim() == 0:
                chunk_of_ids = chunk_of_ids.unsqueeze(0)

            new_token_ids_tensors.append(chunk_of_ids)

        # Stop the timer here
        total_gen_time = time.time() - start_time

        # Now flatten all the chunks into a single 1-D tensor:
        if new_token_ids_tensors:
            # E.g. [tensor([101, 102]), tensor([103]), tensor([104, 105])]
            new_token_ids = torch.cat(new_token_ids_tensors, dim=0)  # shape=(N,)
        else:
            new_token_ids = torch.empty(0, dtype=torch.int64, device=self.model.device)

        # Move to model device if necessary
        new_token_ids = new_token_ids.to(self.model.device)
        generated_text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
        print("=" * 50, flush=True)
        print("Generated text:\n", generated_text, flush=True)
        print("=" * 50, flush=True)

        tpot: float = float("inf")
        if len(new_token_ids) > 1:
            tpot = (total_gen_time - time_to_first_token) / (len(new_token_ids) - 1)

        # Make sure to set a timeout for join to prevent hanging
        thread.join(timeout=300)  # 5 minute timeout
        if thread.is_alive():
            print(
                "Warning: Generation thread did not complete within timeout", flush=True
            )
            wandb.log({"warning": "Generation thread did not complete within timeout"})

        return Result(
            tok_ids_prompt=inputs["input_ids"][0].tolist(),
            tok_ids_new=new_token_ids,
            total_gen_time_s=total_gen_time,
            ttft_s=time_to_first_token,
            tpot_s=tpot,
        )


# ------------------------------------------------------------------------------
# Generation Logic
# ------------------------------------------------------------------------------


def generate_assisted(
    example_id: int,
    prompt: str,
    target_model_obj: HFModel,
    temperature: float,
    assistant_model_obj: Optional[HFModel] = None,
) -> Result:
    """
    Demonstrates an assisted generation approach:
    Optionally pass an assistant model or additional arguments if there's
    custom logic that merges two models. By default, standard Transformers
    doesn't accept a second 'assistant_model', so adjust as needed.
    """
    generate_kwargs = {}
    if assistant_model_obj is not None:
        generate_kwargs["assistant_model"] = assistant_model_obj.model
        are_tokenizers_identical: bool = False
        try:
            target_model_obj.model._validate_assistant(
                assistant_model=assistant_model_obj.model,
                tokenizer=target_model_obj.tokenizer,
                assistant_tokenizer=assistant_model_obj.tokenizer,
            )
        except ValueError as e:
            print(f"Warning: {e}", flush=True)
            if (
                "`assistant_tokenizer` is not required when the main and assistant models use the same tokenizer."
                in str(e)
            ):
                are_tokenizers_identical = True
            elif "The main and assistant moedels have different tokenizers." not in str(
                e
            ):
                raise ValueError(e)
        print("Tokenizers are identical:", are_tokenizers_identical, flush=True)
        if not are_tokenizers_identical:
            generate_kwargs["assistant_tokenizer"] = assistant_model_obj.tokenizer
            generate_kwargs["tokenizer"] = target_model_obj.tokenizer
    do_sample: bool = temperature != 0.0
    if do_sample is True:
        generate_kwargs["temperature"] = temperature
    return target_model_obj.generate_text(
        prompt=prompt, do_sample=do_sample, **generate_kwargs
    )


# ------------------------------------------------------------------------------
# Main Script
# ------------------------------------------------------------------------------


def get_checkpoint_path(dirpath: str, run_name: str) -> str:
    """Returns the path to the checkpoint file."""
    return os.path.join(dirpath, f"{run_name}_checkpoint.csv")


def load_checkpoint(checkpoint_path: str) -> tuple[pd.DataFrame, set]:
    """
    Load existing results and completed examples from checkpoint.
    Returns (DataFrame of results, set of completed example IDs)
    """
    if not os.path.exists(checkpoint_path):
        return pd.DataFrame(), set()

    df = pd.read_csv(checkpoint_path)
    completed = set(
        (
            row.dataset_path,
            row.dataset_name,
            row.dataset_split,
            row.drafter,
            row.temperature,
            row.example_id,
        )
        for row in df.itertuples()
    )
    return df, completed


def save_checkpoint(df: pd.DataFrame, checkpoint_path: str):
    """Save current results to checkpoint file."""
    try:
        df.to_csv(checkpoint_path, index=False)
    except Exception as e:
        print(f"Warning: Failed to save checkpoint: {e}", flush=True)
        wandb.log({"warning": f"Failed to save checkpoint: {e}"})


def main():
    # Environment setup
    set_hf_cache_env()
    login_wandb()
    login_to_hf()

    # Parse arguments
    args = parse_args()
    print("=" * 100, flush=True)
    print(f"{args=}", flush=True)
    print("=" * 100, flush=True)
    print(f"{locals()=}", flush=True)
    print("=" * 100, flush=True)

    # Log hardware info
    log_hardware_info()

    experiment_config_name: str = args.experiment_config
    assert experiment_config_name in experiment_configs, (
        f"Unknown experiment config: {experiment_config_name}"
    )
    experiment_config: ExperimentConfig = experiment_configs[experiment_config_name]
    target_checkpoint: str = experiment_config.target
    print("Loading target model...", flush=True)
    target_obj = HFModel(target_checkpoint)

    results_root_dirpath: str = "benchmark_results"
    dirpath: str = args.results_checkpoint_dirpath
    print(f"Received from args: {dirpath=}")
    if not dirpath:
        print(
            "No checkpoint dirpath provided, using default results dirpath", flush=True
        )
        dirpath = get_results_dirpath(results_root_dirpath)
    print(f"Using results dirpath: {dirpath}", flush=True)

    df_results: pd.DataFrame = pd.DataFrame()

    dataset_config: DatasetConfig
    for dataset_config in tqdm(
        experiment_config.dataset_configs,
        desc="Datasets",
        position=0,
        leave=True,
        ascii=True,
        file=sys.stdout,
    ):
        dataset_path = dataset_config.path
        dataset_name = dataset_config.name
        dataset_split = dataset_config.split

        print("Loading dataset:", flush=True)
        print(f"Dataset path: {dataset_path}", flush=True)
        print(f"Dataset name: {dataset_name}", flush=True)
        print(f"Dataset split: {dataset_split}", flush=True)
        dataset = load_dataset(
            path=dataset_path,
            name=dataset_name,
            split=dataset_split,
            trust_remote_code=True,
        )
        dataset_sample = dataset.take(args.num_of_examples)

        # Setting up wandb run
        assistant_checkpoints: List[str] = experiment_config.assistants
        run_name = f"{target_checkpoint}_{dataset_path}_{dataset_name}_{dataset_split}_{args.num_of_examples}_{'-'.join(assistant_checkpoints)}".replace(
            "/", "-"
        )
        print(f"{run_name=}", flush=True)
        wandb_run = wandb.init(
            project="hf-bench",
            config={
                "target": experiment_config.target,
                "dataset_path": dataset_path,
                "dataset_name": dataset_name,
                "dataset_split": dataset_split,
                "num_of_examples": args.num_of_examples,
            },
            tags=[
                f"target:{target_checkpoint}",
                f"dataset:{dataset_path}/{dataset_name}/{dataset_split}",
                f"num_of_examples:{args.num_of_examples}",
            ],
            name=run_name,
        )
        wandb_artifact = wandb.Artifact(
            name=f"results_per_example_{run_name}", type="dataset"
        )
        columns = list(ResultsTableRow.__dataclass_fields__.keys())
        wandb_table = wandb.Table(columns=columns)
        print(f"{wandb_table=}", flush=True)
        wandb_artifact.add(wandb_table, "my_table")

        # Initialize results tracking
        checkpoint_path = get_checkpoint_path(dirpath, run_name)
        df_results, completed_examples = load_checkpoint(checkpoint_path)

        assistant_checkpoints = [None] + assistant_checkpoints
        for assistant_checkpoint in tqdm(
            assistant_checkpoints,
            desc="Assistants",
            position=1,
            leave=True,
            ascii=True,
            file=sys.stdout,
        ):
            print(f"Loading assistant model {assistant_checkpoint}...", flush=True)
            assistant_obj = (
                None if assistant_checkpoint is None else HFModel(assistant_checkpoint)
            )
            try:
                for temperature in tqdm(
                    experiment_config.temperatures,
                    desc="Temperatures",
                    position=2,
                    leave=True,
                    ascii=True,
                    file=sys.stdout,
                ):
                    # Generation loop
                    for example_id, example in tqdm(
                        enumerate(dataset_sample),
                        desc="Examples",
                        position=3,
                        leave=True,
                        ascii=True,
                        file=sys.stdout,
                    ):
                        # Check if this example was already processed
                        example_key = (
                            dataset_config.path,
                            dataset_config.name,
                            dataset_config.split,
                            assistant_checkpoint,
                            temperature,
                            example_id,
                        )
                        if example_key in completed_examples:
                            print(
                                f"Skipping already completed example {example_key}",
                                flush=True,
                            )
                            continue

                        # Get prompt
                        match dataset_path:
                            case "tau/scrolls":
                                prompt = f"Summarize the following text:\n{example['input']}\nSummary:\n"
                            case "cnn_dailymail":
                                prompt = f"Summarize the following article:\n{example['article']}\nSummary:\n"
                            case "openai/openai_humaneval":
                                prompt = f"Implement the function so that it passes the tests.\nTests:\n{example['test']}\nFunction:\n{example['prompt']}\n\nYour code:\n"
                            case _:
                                raise ValueError(
                                    f"Unknown dataset path: {dataset_path}"
                                )

                        # Run generation
                        print("=" * 100, flush=True)
                        print(f"Running input prompt {example_id}...", flush=True)
                        print("Prompt:\n", prompt, flush=True)
                        print("=" * 100, flush=True)

                        print(
                            f"Running `assistant={assistant_checkpoint}` with `temp={temperature}` for {target_checkpoint}...",
                            flush=True,
                        )
                        try:
                            result = generate_assisted(
                                example_id=example_id,
                                prompt=prompt,
                                target_model_obj=target_obj,
                                temperature=temperature,
                                assistant_model_obj=assistant_obj,
                            )
                            results_table_row = (
                                ResultsTableRow.from_experiment_config_and_result(
                                    target=target_checkpoint,
                                    dataset_path=dataset_path,
                                    dataset_name=dataset_name,
                                    dataset_split=dataset_split,
                                    num_of_examples=args.num_of_examples,
                                    drafter=assistant_checkpoint,
                                    temperature=temperature,
                                    example_id=example_id,
                                    result=result,
                                )
                            )

                            # Stream results after each generation
                            df_results = pd.concat(
                                [df_results, pd.DataFrame([results_table_row])],
                                ignore_index=True,
                            )
                            save_checkpoint(df_results, checkpoint_path)
                            completed_examples.add(example_key)

                            # Update W&B table and log progress
                            wandb_table.add_data(*astuple(results_table_row))
                            wandb.log(
                                {
                                    "completed_examples": len(completed_examples),
                                    "progress": len(completed_examples)
                                    / (
                                        len(dataset_sample)
                                        * len(experiment_config.temperatures)
                                        * len(assistant_checkpoints)
                                    ),
                                }
                            )
                        except Exception as e:
                            print(f"Error: {e}", flush=True)
                            wandb.log({"error": str(e)})
                            traceback.print_exc()
                            wandb.log({"traceback": traceback.format_exc()})
            finally:
                if assistant_obj is not None:
                    del assistant_obj
                    clear_memory()

    wandb_run.log_artifact(wandb_artifact)
    wandb_run.log({"results": wandb.Table(dataframe=df_results)})

    # Save to the benchmark_results directory
    filepath_results = os.path.join(dirpath, f"{run_name}.csv")
    os.makedirs(os.path.dirname(filepath_results), exist_ok=True)
    df_results.to_csv(filepath_results, index=False)
    print(f"Results saved to {filepath_results}", flush=True)

    wandb_run.finish()


if __name__ == "__main__":
    main()
