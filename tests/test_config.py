import pytest
from pathlib import Path
from hf_bench.utils.config import load_config, _generate_experiment_id
from hf_bench.schemas import ExperimentConfig, ModelConfig, GenerationConfig
import yaml
from datetime import datetime

def test_experiment_id_generation():
    """Test that experiment IDs are generated correctly."""
    # Load the config first to get the dict
    with open("configs/default_config.yaml") as f:
        config_dict = yaml.safe_load(f)
    
    # Generate ID
    experiment_id = _generate_experiment_id(config_dict)
    
    # Get today's date
    today = datetime.now().strftime("%Y%m%d")
    
    # Test format
    assert experiment_id.startswith(today)  # Starts with date
    assert "meta-llama/Llama-3.1-70B" in experiment_id  # Contains full model path
    assert "qasper" in experiment_id    # Contains task name
    assert "4drafters" in experiment_id # Contains number of drafters
    
    # Test that parts are properly separated
    parts = experiment_id.split('_')
    assert len(parts) == 5  # date, time, model, task, drafters
    assert len(parts[0]) == 8  # YYYYMMDD
    assert len(parts[1]) == 4  # HHMM

def test_load_default_config():
    """Test that the default config loads correctly and has expected values."""
    # Load the default config
    config = load_config("configs/default_config.yaml")
    
    # Test that we got the right type
    assert isinstance(config, ExperimentConfig)
    
    # Test basic structure and some key values
    assert isinstance(config.output_dir, Path)
    assert str(config.output_dir).startswith("results/raw/")
    
    # Test target model config
    assert isinstance(config.target_model, ModelConfig)
    assert config.target_model.hf_path == "meta-llama/Llama-3.1-70B"
    assert isinstance(config.target_model.generation_config, GenerationConfig)
    assert config.target_model.generation_config.temperatures == [0, 1]
    
    # Test drafter models
    assert len(config.drafter_models) == 4
    assert isinstance(config.drafter_models[0], ModelConfig)
    assert config.drafter_models[0].hf_path == "meta-llama/Llama-3.2-1B-Instruct"
    assert config.drafter_models[-1].hf_path == "Qwen/Qwen2.5-0.5B-Instruct"
    assert config.drafter_models[-1].generation_config.temperatures == [0, 1e-7, 1]
    
    # Test task config
    assert config.task.dataset.hf_path == "tau/scrolls"
    assert config.task.dataset.name == "qasper"
    assert config.task.dataset.split == "test"
    assert config.task.dataset.num_examples == 30
    assert config.task.max_new_tokens == 512
    
    # Test cluster config
    assert config.cluster_config.type == "lsf"
    assert config.cluster_config.lsf.gpu.count == 1
    assert config.cluster_config.lsf.gpu.memory_gb == 80
    assert config.cluster_config.lsf.gpu.is_exclusive is True
    assert config.cluster_config.lsf.cpu_cores == 8
    assert config.cluster_config.lsf.memory_gb == 200
    assert "long-gpu" in config.cluster_config.lsf.queue_names
    assert len(config.cluster_config.lsf.modules) == 2
    
    # Test logging config
    assert isinstance(config.logging.output_dir, Path)
    assert str(config.logging.output_dir).startswith("~")
    assert config.logging.level == "DEBUG"
    assert config.logging.format == '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Test metrics
    assert "ttft" in config.metrics
    assert "num_new_toks" in config.metrics
    assert len(config.metrics) == 10

    # Update test for resolved experiment_id in output_dir
    output_dir_str = str(config.output_dir)
    assert any(c.isdigit() for c in output_dir_str)  # Contains numbers (date)
    assert "meta-llama/Llama-3.1-70B" in output_dir_str  # Contains full model path
    assert "qasper" in output_dir_str        # Contains task name
    assert "drafters" in output_dir_str      # Contains drafters info

def test_load_nonexistent_config():
    """Test that loading a nonexistent config raises an error."""
    with pytest.raises(FileNotFoundError):
        load_config("configs/nonexistent.yaml")
