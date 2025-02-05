from hf_bench.benchmark import HFModel

import argparse
from hf_bench.config import experiment_configs

def validate_assistant(target_model_name, assistant_model_name):
    """
    Given two model names, load them and call _validate_assistant.
    
    Args:
        target_model_name (str): Name of the target model.
        assistant_model_name (str): Name of the assistant model.
    """
    # Load models and tokenizers
    target_model_obj = HFModel(target_model_name)
    assistant_model_obj = HFModel(assistant_model_name)

    try:
        # Call the validation function
        target_model_obj.model._validate_assistant(
            assistant_model=assistant_model_obj.model,
            tokenizer=target_model_obj.tokenizer,
            assistant_tokenizer=assistant_model_obj.tokenizer,
        )
        print("Need universal SD")
    except ValueError:
        print(f"Can run with regular SD")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check whether target/draft Models can run with regular SD")
    parser.add_argument("--config_name", type=str, help="The name of the configuration to use.")
    args = parser.parse_args()

    config = experiment_configs.get(args.config_name)
    if config:
        print(f"Running experiment with target: {config.target}")
        for assistant_model in config.assistants:
            print(f"{config.target=},{assistant_model}")
            common_tokens = validate_assistant(config.target, assistant_model)