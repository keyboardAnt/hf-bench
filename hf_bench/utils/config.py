from dacite import from_dict, Config
import yaml
from pathlib import Path
from datetime import datetime
import re
from hf_bench.schemas import ExperimentConfig

def _generate_experiment_id(config_dict: dict) -> str:
    """Generate a unique experiment ID based on config contents."""
    now = datetime.now()
    date = now.strftime("%Y%m%d")
    time = now.strftime("%H%M")
    
    # Keep the model path exactly as is
    target_model = config_dict['target_model']['hf_path']
    
    # Get task name
    task = config_dict['task']['dataset']['name']
    
    # Count drafters
    num_drafters = len(config_dict['drafter_models'])
    
    return f"{date}_{time}_{target_model}_{task}_{num_drafters}drafters"

def load_config(config_path: str) -> ExperimentConfig:
    """Load and process the configuration file."""
    with open(config_path) as f:
        # Use float for scientific notation
        config_dict = yaml.safe_load(f)
    
    # Generate experiment ID
    experiment_id = _generate_experiment_id(config_dict)
    
    # Resolve variables in paths and convert to Path objects
    config_dict['output_dir'] = Path(config_dict['output_dir'].replace('${experiment_id}', experiment_id))
    config_dict['logging']['output_dir'] = Path(config_dict['logging']['output_dir'])
    
    # Convert all temperature values to float
    for model in [config_dict['target_model']] + config_dict['drafter_models']:
        model['generation_config']['temperatures'] = [
            float(t) for t in model['generation_config']['temperatures']
        ]
    
    return from_dict(
        data_class=ExperimentConfig,
        data=config_dict,
        config=Config()
    )