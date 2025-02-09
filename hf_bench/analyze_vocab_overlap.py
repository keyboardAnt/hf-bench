from transformers import AutoTokenizer, AutoConfig
from transformers.generation.candidate_generator import (
    AssistantToTargetTranslator
)
from datasets import load_dataset
from tqdm import tqdm
import torch
import argparse

from hf_bench.config import experiment_configs


def compare_vocabularies(target_model, assistant_model, num_samples, dataset_config):
    # Load tokenizers
    target_tokenizer = AutoTokenizer.from_pretrained(target_model)
    assistant_tokenizer = AutoTokenizer.from_pretrained(assistant_model)
    
    # Get vocabularies as sets
    vocab1 = set(target_tokenizer.get_vocab().keys())
    vocab2 = set(assistant_tokenizer.get_vocab().keys())
    
    # # Vocabulary sizes
    vocab_size1 = len(vocab1)
    vocab_size2 = len(vocab2)

    # Intersection of vocabularies
    common_tokens = vocab1.intersection(vocab2)
    num_common_tokens = len(common_tokens)

    print(f"Model: {target_model}")
    print(f"Target Tokenizer Vocabulary size: {vocab_size1}")
    print(f"Model: \t {assistant_model}")
    print(f"Assistant Tokenizer Vocabulary size: {vocab_size2}")
    print(f"Number of tokens in both vocabularies: {num_common_tokens}")

    # Load configurations
    config1 = AutoConfig.from_pretrained(target_model)
    config2 = AutoConfig.from_pretrained(assistant_model)
    
    # Vocabulary sizes from config
    target_vocab_size = config1.vocab_size
    vocab_size2 = config2.vocab_size
    
    # Print results
    print(f"Target Config Vocabulary size: {target_vocab_size}")
    print(f"Assistant Config Vocabulary size: {vocab_size2}")

    translator = AssistantToTargetTranslator(
        target_tokenizer=target_tokenizer,
        assistant_tokenizer=assistant_tokenizer,
        assistant_model_device='cpu',
        target_vocab_size=target_vocab_size,
    )
    
    count = torch.sum(-1 != translator._assistant_to_target_input_ids).item()
    print(f'Vocab overlap from translator: {count}')
    print(f'Translator Overlap: {round(count/vocab_size1*100,0)}')

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
    ds_iterator = iter(dataset.take(num_samples))


    pbar = tqdm(range(num_samples))
    count = 0
    total_len = 0
    for i in pbar:
        match dataset_path:
            case "tau/scrolls":
                prompt = f"Summarize the following text:\n{next(ds_iterator)['input']}\nSummary:\n"
            case "cnn_dailymail":
                prompt = f"Summarize the following article:\n{next(ds_iterator)['article']}\nSummary:\n"
            case "openai/openai_humaneval":
                dsi = next(ds_iterator)
                prompt = f"Implement the function so that it passes the tests.\nTests:\n{dsi['test']}\nFunction:\n{dsi['prompt']}\n\nYour code:\n"
            case _:
                raise ValueError(
                    f"Unknown dataset path: {dataset_path}"
                )
        token_ids = assistant_tokenizer([prompt], add_special_tokens=False, return_tensors="pt")["input_ids"]
        target_token_ids = translator._assistant_to_target_input_ids[token_ids]
        count += torch.sum(target_token_ids != -1).item()
        total_len += target_token_ids.shape[1]
    

    print(f'Prompt Freq Overlap: {round(count/total_len*100,0)}')

def generate_latex_tables(target_model, assistant_model, num_samples, dataset_config, vocab_overlap_table, dataset_overlap_table):
    assistant_tokenizer = None
    id = f'{target_model.split("/")[-1]} & {assistant_model.split("/")[-1]} & '
    if id not in vocab_overlap_table:
        # Load tokenizers
        target_tokenizer = AutoTokenizer.from_pretrained(target_model)
        assistant_tokenizer = AutoTokenizer.from_pretrained(assistant_model)
    
        # Load configurations
        config_target = AutoConfig.from_pretrained(target_model)
        
        # Vocabulary sizes from config
        target_vocab_size = config_target.vocab_size

        translator = AssistantToTargetTranslator(
            target_tokenizer=target_tokenizer,
            assistant_tokenizer=assistant_tokenizer,
            target_vocab_size=target_vocab_size,
        )
        
        T_inter_D = torch.sum(-1 != translator._assistant_to_target_input_ids).item()
        T_inter_D_by_T = round(T_inter_D/target_vocab_size,2)
        vocab_overlap_table[id] = f'{T_inter_D} & {T_inter_D_by_T}'

    dataset_path = dataset_config.path
    id = f'{target_model.split("/")[-1]} & {assistant_model.split("/")[-1]} & {dataset_path} & '
    if id not in dataset_overlap_table:
        match dataset_path:
            case "tau/scrolls":
                task = "long-ctx summ"
            case "cnn_dailymail":
                task = "summ"
            case "openai/openai_humaneval":
                task = "coding"
            case _:
                raise ValueError(
                    f"Unknown dataset path: {dataset_path}"
                )
        id = f'{target_model} & {assistant_model} & {task} & {dataset_path} & '
        if assistant_tokenizer is None:
            assistant_tokenizer = AutoTokenizer.from_pretrained(assistant_model)
            target_tokenizer = AutoTokenizer.from_pretrained(target_model)
            translator = AssistantToTargetTranslator(
                target_tokenizer=target_tokenizer,
                assistant_tokenizer=assistant_tokenizer,
            )
        dataset_name = dataset_config.name
        dataset_split = dataset_config.split

        dataset = load_dataset(
            path=dataset_path,
            name=dataset_name,
            split=dataset_split,
            trust_remote_code=True,
        )
        ds_iterator = iter(dataset.take(num_samples))

        pbar = tqdm(range(num_samples))
        T_inter_D = 0
        total_len = 0
        for i in pbar:
            match dataset_path:
                case "tau/scrolls":
                    prompt = f"Summarize the following text:\n{next(ds_iterator)['input']}\nSummary:\n"
                case "cnn_dailymail":
                    prompt = f"Summarize the following article:\n{next(ds_iterator)['article']}\nSummary:\n"
                case "openai/openai_humaneval":
                    dsi = next(ds_iterator)
                    prompt = f"Implement the function so that it passes the tests.\nTests:\n{dsi['test']}\nFunction:\n{dsi['prompt']}\n\nYour code:\n"
                case _:
                    raise ValueError(
                        f"Unknown dataset path: {dataset_path}"
                    )
            token_ids = assistant_tokenizer([prompt], add_special_tokens=False, return_tensors="pt")["input_ids"]
            target_token_ids = translator._assistant_to_target_input_ids[token_ids]
            T_inter_D += torch.sum(target_token_ids != -1).item()
            total_len += target_token_ids.shape[1]
        
        overlap = round(T_inter_D/total_len,2)
        dataset_overlap_table[id] = f"{overlap}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze vocabulary overlap of specified configuration.")
    parser.add_argument("--config_names", type=str, nargs="*", help="The names of the configurations to use (space-separated list).")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to use for analysis.")
    parser.add_argument("--generate_latex", action="store_true", help="Generate latex tables for the overlap.")
    args = parser.parse_args()
    if args.generate_latex:
        vocab_overlap_table = dict()
        dataset_overlap_table = dict()
        args.config_names = [key for key in experiment_configs if "deepseek" in key.lower() and key != "deepseek-r1"]

    for config_name in args.config_names:
        config = experiment_configs.get(config_name)
        print(f'{config_name=}')
        for assistant_model in config.assistants:
            for dataset_config in config.dataset_configs:
                if args.generate_latex:
                    generate_latex_tables(config.target, assistant_model, args.num_samples, dataset_config, vocab_overlap_table, dataset_overlap_table)
                else:
                    compare_vocabularies(config.target, assistant_model, args.num_samples, dataset_config)
    
    if args.generate_latex:
        print("Vocab Overlap Table")
        for k, v in sorted(vocab_overlap_table.items()):
            k = k.replace('_', '\_')
            print(f'{k} {v} \\\\')
        
        print("Dataset Overlap Table")
        for k, v in sorted(dataset_overlap_table.items()):
            k = k.replace('_', '\_')
            print(f'{k} {v} \\\\')
    



