from datetime import datetime


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