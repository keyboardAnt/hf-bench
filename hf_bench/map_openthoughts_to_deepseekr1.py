import json
from transformers import AutoTokenizer
from datasets import load_dataset
from jinja2 import Template


def main():
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    # Now load the dataset open-thoughts/OpenThoughts-114k and apply the chat template
    dataset = load_dataset("open-thoughts/OpenThoughts-114k", split='train')

    output_prompts = [] # To store generated prompts
    output_jsonl_file = "deepseek_r1_prompts_openthoughts.jsonl"

    # Get and print the chat template
    chat_template_deepseek = tokenizer.chat_template
    jinja_template = Template(chat_template_deepseek)

    for sample in dataset:
        messages_input = []
        for message_dict in sample['conversations']:
            # TODO - Remove these once the following format is confirmed
            # role = message_dict['from']
            # content = message_dict['value']
            # messages_input.append({'role': role, 'content': content})

            role_from_dataset = message_dict['from'] # Get role from 'from' key
            content = message_dict['value']         # Get content from 'value' key

            if role_from_dataset == 'user':       # Map 'user' role
                messages_input.append({'role': 'user', 'content': content})
            elif role_from_dataset == 'assistant': # Map 'assistant' role
                messages_input.append({'role': 'assistant', 'content': content})

        # Keep `add_generation_prompt=False` for dataset preprocessing
        bos_token = "<|startoftext|>"
        prompt = jinja_template.render(messages=messages_input,
                            bos_token=bos_token, add_generation_prompt=False)
        output_prompts.append(prompt) # Store in a list

        with open(output_jsonl_file, 'a') as f: # 'a' for append mode
            json_record = {"prompt": prompt} # Or store more info if needed
            f.write(json.dumps(json_record) + '\n')


if __name__ == "__main__":
    main()
