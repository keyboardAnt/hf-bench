from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path

@dataclass
class GenerationConfig:
    temperatures: List[float]
    max_new_tokens: Optional[int] = None

@dataclass
class ModelConfig:
    hf_path: str
    generation_config: GenerationConfig

@dataclass
class DatasetConfig:
    hf_path: str
    name: str
    split: str
    num_examples: int

@dataclass
class TaskConfig:
    dataset: DatasetConfig
    max_new_tokens: int

@dataclass
class GPUConfig:
    type: Optional[str]
    count: int
    memory_gb: int
    is_exclusive: bool

@dataclass
class LSFConfig:
    gpu: GPUConfig
    cpu_cores: int
    memory_gb: int
    queue_names: List[str]
    num_hosts: int
    num_processes: int
    modules: List[str]

@dataclass
class ClusterConfig:
    type: str
    lsf: LSFConfig
    slurm: Optional[Any] = None

@dataclass
class LoggingConfig:
    level: str
    format: str
    output_dir: Path
    filename_pattern: str

@dataclass
class ExperimentConfig:
    output_dir: Path
    target_model: ModelConfig
    drafter_models: List[ModelConfig]
    task: TaskConfig
    cluster_config: ClusterConfig
    logging: LoggingConfig
    metrics: List[str]

@dataclass
class ExperimentMetrics:
    num_new_toks: int
    ttft: float  # Time to first token
    tpot_hmean_ms: float  # Time per output token (harmonic mean)
    tpot_min_ms: float  # Time per output token (minimum)
    output_tokens_per_sec_hmean: float
    output_tokens_per_sec_max: float
    output_tokens_per_sec_std: float
    per_example_speedup_over_ar_star_hmean: float
    per_example_speedup_over_ar_star_max: float
    per_example_speedup_over_ar_star_std: float 