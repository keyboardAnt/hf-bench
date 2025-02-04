#


from dataclasses import dataclass
from typing import List


@dataclass
class DatasetConfig:
    path: str
    name: str
    split: str

    @classmethod
    def from_path(cls, path: str) -> "DatasetConfig":
        return dataset_configs[path]


dataset_configs = {
    "tau/scrolls": DatasetConfig(
        path="tau/scrolls",
        name="qasper",
        split="test",
    ),
    "cnn_dailymail": DatasetConfig(
        path="cnn_dailymail",
        name="3.0.0",
        split="validation",
    ),
    "openai/openai_humaneval": DatasetConfig(
        path="openai/openai_humaneval",
        name="openai_humaneval",
        split="test",
    ),
}


@dataclass
class ExperimentConfig:
    target: str
    dataset_configs: List[DatasetConfig]
    assistants: List[str]
    temperatures: List[float]


experiment_configs = {
    "default": ExperimentConfig(
        target="Qwen/Qwen2.5-1.5B-Instruct",
        dataset_configs=[DatasetConfig.from_path("tau/scrolls")],
        assistants=["Qwen/Qwen2.5-0.5B-Instruct", "double7/vicuna-68m"],
        temperatures=[0, 1],
    ),
    "llama70b-it": ExperimentConfig(
        target="meta-llama/Llama-3.1-70B-Instruct",
        dataset_configs=list(dataset_configs.values()),
        assistants=[
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.2-1B-Instruct",
            "Qwen/Qwen2.5-0.5B-Instruct",
        ],
        temperatures=[0, 1e-7, 1],
    ),
    "llama70b-it-vicuna-68m": ExperimentConfig(
        target="meta-llama/Llama-3.1-70B-Instruct",
        dataset_configs=[
            DatasetConfig.from_path("tau/scrolls"),
            DatasetConfig.from_path("cnn_dailymail"),
        ],
        assistants=["double7/vicuna-68m"],
        temperatures=[0, 0.2, 1],
    ),
    "llama70b": ExperimentConfig(
        target="meta-llama/Llama-3.1-70B",
        dataset_configs=list(dataset_configs.values()),
        assistants=[
            "meta-llama/Llama-3.1-8B",
            "meta-llama/Llama-3.2-3B",
            "meta-llama/Llama-3.2-1B",
            "Qwen/Qwen2.5-0.5B-Instruct",
        ],
        temperatures=[0, 1e-7, 1],
    ),
    "llama70b-instruct-cnndm": ExperimentConfig(
        target="meta-llama/Llama-3.1-70B-Instruct",
        dataset_configs=[DatasetConfig.from_path("cnn_dailymail")],
        assistants=[
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.2-1B-Instruct",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "double7/vicuna-68m",
        ],
        temperatures=[0, 1e-7, 1],
    ),
    "llama8b-instruct-cnndm": ExperimentConfig(
        target="meta-llama/Llama-3.1-8B-Instruct",
        dataset_configs=[DatasetConfig.from_path("cnn_dailymail")],
        assistants=[
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.2-1B-Instruct",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "double7/vicuna-68m",
        ],
        temperatures=[0, 1e-7, 1],
    ),
    "llama70b-cnndm": ExperimentConfig(
        target="meta-llama/Llama-3.1-70B",
        dataset_configs=[DatasetConfig.from_path("cnn_dailymail")],
        assistants=[
            "meta-llama/Llama-3.1-8B",
            "meta-llama/Llama-3.2-3B",
            "meta-llama/Llama-3.2-1B",
            "Qwen/Qwen2.5-0.5B",
            "double7/vicuna-68m",
        ],
        temperatures=[0, 1e-7, 1],
    ),
    "llama8b-cnndm": ExperimentConfig(
        target="meta-llama/Llama-3.1-8B",
        dataset_configs=[DatasetConfig.from_path("cnn_dailymail")],
        assistants=[
            "meta-llama/Llama-3.2-3B",
            "meta-llama/Llama-3.2-1B",
            "Qwen/Qwen2.5-0.5B",
            "double7/vicuna-68m",
        ],
        temperatures=[0, 1e-7, 1],
    ),


    "mixtral-8x22b-it": ExperimentConfig(
        target="mistralai/Mixtral-8x22B-Instruct-v0.1",
        dataset_configs=list(dataset_configs.values()),
        assistants=["Qwen/Qwen2.5-0.5B-Instruct", "double7/vicuna-68m"],
        temperatures=[0, 1e-7, 1],
    ),
    "gemma-9b-it": ExperimentConfig(
        target="google/gemma-2-9b-it",
        dataset_configs=list(dataset_configs.values()),
        assistants=["google/gemma-2-2b-it", "double7/vicuna-68m"],
        temperatures=[0, 1e-7, 1],
    ),
    "phi-4": ExperimentConfig(
        target="microsoft/phi-4",
        dataset_configs=list(dataset_configs.values()),
        assistants=["microsoft/Phi-3.5-mini-instruct", "Qwen/Qwen2.5-0.5B-Instruct"],
        temperatures=[0, 1e-7, 1],
    ),
    "llama-8b-it": ExperimentConfig(
        target="meta-llama/Llama-3.1-8B-Instruct",
        dataset_configs=list(dataset_configs.values()),
        assistants=[
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.2-1B-Instruct",
            "Qwen/Qwen2.5-0.5B-Instruct",
        ],
        temperatures=[0, 1e-7, 1],
    ),
    "llama-8b-it-vicuna-68m": ExperimentConfig(
        target="meta-llama/Llama-3.1-8B-Instruct",
        dataset_configs=[
            DatasetConfig.from_path("tau/scrolls"),
            DatasetConfig.from_path("cnn_dailymail"),
        ],
        assistants=["double7/vicuna-68m"],
        temperatures=[0, 0.2, 1],
    ),
    "codellama-13b-it": ExperimentConfig(
        target="codellama/CodeLlama-13b-Instruct-hf",
        dataset_configs=[DatasetConfig.from_path("openai/openai_humaneval")],
        assistants=["codellama/CodeLlama-7b-Instruct-hf", "bigcode/tiny_starcoder_py"],
        temperatures=[0, 1e-7, 1],
    ),
    "deepseek-r1-qwen-32b-sum": ExperimentConfig(
        target="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        dataset_configs=[
            DatasetConfig.from_path("tau/scrolls"),
            DatasetConfig.from_path("cnn_dailymail"),
        ],
        assistants=[
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-0.5B", 
            "double7/vicuna-68m"],
        temperatures=[0, 1],
    ),

    "deepseek-r1-llama-70b-sum": ExperimentConfig(
        target="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        dataset_configs=[
            DatasetConfig.from_path("tau/scrolls"),
            DatasetConfig.from_path("cnn_dailymail"),
        ],
        assistants=[
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "double7/vicuna-68m", 
            "meta-llama/Llama-3.1-8B",
            "meta-llama/Llama-3.2-3B",
            "meta-llama/Llama-3.2-1B"],
        temperatures=[0, 1],
    ),

        "deepseek-r1-qwen-32b-code": ExperimentConfig(
        target="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        dataset_configs=[
            DatasetConfig.from_path("openai/openai_humaneval")
        ],
        assistants=[
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-0.5B", 
            "codellama/CodeLlama-7b-Instruct-hf", 
            "bigcode/tiny_starcoder_py"],
        temperatures=[0, 1],
    ),

    "deepseek-r1-llama-70b-code": ExperimentConfig(
        target="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        dataset_configs=[
            DatasetConfig.from_path("openai/openai_humaneval")
        ],
        assistants=[
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "codellama/CodeLlama-7b-Instruct-hf", 
            "bigcode/tiny_starcoder_py"],
        temperatures=[0, 1],
    ),

}
