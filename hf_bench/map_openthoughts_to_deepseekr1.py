import argparse
import json
import os
from transformers import AutoTokenizer
from datasets import load_dataset
from jinja2 import Template
from huggingface_hub import HfApi


# args to choose whether to push the dataset to the hub
parser = argparse.ArgumentParser()
parser.add_argument("--push_to_hub", action="store_true", help="Push the dataset to the hub")
parser.add_argument("--repo_id_name", type=str, default=None)
# dataset range
parser.add_argument("--num_samples", type=int, default=None)
args = parser.parse_args()


def main():
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    # Now load the dataset open-thoughts/OpenThoughts-114k and apply the chat template
    # slice the dataset to the number of samples if specified
    if args.num_samples:
        dataset = load_dataset("open-thoughts/OpenThoughts-114k", split='train').select(range(args.num_samples))
    else:
        dataset = load_dataset("open-thoughts/OpenThoughts-114k", split='train')

    output_jsonl_file = "deepseek_r1_prompts_openthoughts.jsonl"
    # Get and print the chat template
    chat_template_deepseek = tokenizer.chat_template
    jinja_template = Template(chat_template_deepseek)

    for sample in dataset:
        messages_input = []
        for message_dict in sample['conversations']:
            role_from_dataset = message_dict['from'] # Get role from 'from' key
            content = message_dict['value']         # Get content from 'value' key

            if role_from_dataset == 'user':       # Map 'user' role
                messages_input.append({'role': 'user', 'content': content})
            elif role_from_dataset == 'assistant': # Map 'assistant' role
                messages_input.append({'role': 'assistant', 'content': content})

        # Keep `add_generation_prompt=True` to add the |Assistant| tag to the prompt
        bos_token = "<|startoftext|>"
        prompt = jinja_template.render(messages=messages_input,
                            bos_token=bos_token, add_generation_prompt=True)

        with open(output_jsonl_file, 'a') as f: # 'a' for append mode
            json_record = {"prompt": prompt} # Or store more info if needed
            f.write(json.dumps(json_record) + '\n')

    if args.push_to_hub:
        try:
            api = HfApi(token=os.environ["HF_TOKEN"])
        except Exception as e:
            raise Exception(f"HF_TOKEN not found in environment variables {e}")

        api.upload_file(
            path_or_fileobj=output_jsonl_file,
            path_in_repo="deepseek_r1_prompts_openthoughts.jsonl",
            repo_id=args.repo_id_name,
            repo_type="dataset",
        )


if __name__ == "__main__":
    main()
