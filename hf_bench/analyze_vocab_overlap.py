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
    #indices = torch.arange(len(translator._assistant_to_target_input_ids))
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


# Example usage

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze vocabulary overlap of specified configuration.")
    parser.add_argument("--config_name", type=str, help="The name of the configuration to use.")
    args = parser.parse_args()

    config = experiment_configs.get(args.config_name)
    if config:
        print(f"Running experiment with target: {config.target}")
        for assistant_model in config.assistants:
            num_samples = 50
            for dataset_config in config.dataset_configs:
                compare_vocabularies(config.target, assistant_model, num_samples, dataset_config)
    else:
        print(f"No configuration found for {args.config_name}")



