from transformers import AutoTokenizer
import torch

# take model names as args inputs for the tokenizers
import argparse
parser = argparse.ArgumentParser(description='Compare two tokenizers')
parser.add_argument('-t', '--target_model', type=str, required=True, help='Target model name')
parser.add_argument('-a', '--assistant_model', type=str, required=True, help='Assistant model name')
args = parser.parse_args()


def get_assistant_to_target_input_ids(target_tokenizer, assistant_tokenizer):
    suppress_tokens_id= -1
    target_vocab = target_tokenizer.get_vocab()
    assistant_vocab = assistant_tokenizer.get_vocab()
    
    space_str = " "
    target_space_ids = target_tokenizer(space_str, add_special_tokens=False)["input_ids"]
    assistant_space_ids = assistant_tokenizer(space_str, add_special_tokens=False)["input_ids"]
    if len(target_space_ids) > 0:
        target_space_sign = target_tokenizer.convert_ids_to_tokens(target_space_ids)[0][0]

        assistant_space_ids = assistant_tokenizer(space_str, add_special_tokens=False)["input_ids"]
        if len(assistant_space_ids) > 0:
            assistant_space_sign = assistant_tokenizer.convert_ids_to_tokens(assistant_space_ids)[0][0]

            if target_space_sign != assistant_space_sign:
                assistant_vocab = {
                    (
                        tok.replace(assistant_space_sign, target_space_sign, 1)
                        if tok.startswith(assistant_space_sign)
                        else tok
                    ): idx
                    for tok, idx in assistant_vocab.items()
                }

    max_assistant_index = max(assistant_vocab.values())
    assistant_to_target_input_ids = torch.full((max_assistant_index + 1,), suppress_tokens_id, dtype=int)
    target_to_assistant_input_ids = {}
    for tok, assistant_id in assistant_vocab.items():
        target_id = target_vocab.get(tok)
        if target_id is not None:
            assistant_to_target_input_ids[assistant_id] = target_id
            target_to_assistant_input_ids[target_id] = assistant_id

    # True vocab lengths
    # There is a difference between the vocab size and the length of the vocab
    print(f"Target vocab length {len(target_vocab)}")
    print(f"Target Tokenizer vocab size {target_tokenizer.vocab_size}")
    print(f"Assistant vocab length {len(assistant_vocab)}")

    # We have assistant to target input ids and target to assistant input ids
    print(f"Assistant to target input ids {assistant_to_target_input_ids.shape}")
    # This is the overlap, i.e. the translation of Target Ids to Assistant Ids
    print(f"Target to assistant input ids (Overlap) - {len(target_to_assistant_input_ids.keys())}")

    return

# Load two tokenizers
tokenizer1 = AutoTokenizer.from_pretrained(args.target_model)
tokenizer2 = AutoTokenizer.from_pretrained(args.assistant_model)

get_assistant_to_target_input_ids(tokenizer1, tokenizer2)
