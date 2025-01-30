from dacite import from_dict
import yaml
from pathlib import Path

def load_config(config_path: str) -> ExperimentConfig:
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    # Convert string paths to Path objects
    config_dict['output_dir'] = Path(config_dict['output_dir'])
    config_dict['logging']['output_dir'] = Path(config_dict['logging']['output_dir'])
    
    return from_dict(data_class=ExperimentConfig, data=config_dict)
