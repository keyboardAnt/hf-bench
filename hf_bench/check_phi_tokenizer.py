from transformers import AutoTokenizer
import torch

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

    for tok, idx in assistant_vocab.items():
        if tok.startswith(assistant_space_sign):
            print(f"We shouldn't be here {tok}, idx {idx}")

    vocab1 = set(target_vocab.keys())
    vocab2 = set(assistant_vocab.keys())
    # Compare vocabularies
    common_tokens = vocab1.intersection(vocab2)
    print(f"Len Target Tokenizer {len(vocab1)}, assistant tokenizer {len(vocab2)}")
    print(f"Len Common Tokens {len(common_tokens)}")

    unique_to_tokenizer1 = vocab1.difference(vocab2)
    unique_to_tokenizer2 = vocab2.difference(vocab1)

    # Print results
    print("Common tokens:", len(common_tokens))
    print("Unique to Tokenizer 1:", len(unique_to_tokenizer1))
    print("Unique to Tokenizer 2:", len(unique_to_tokenizer2))

    # Optional: Display some sample tokens
    print("Sample common tokens:", list(common_tokens)[:10])
    print("Sample unique to Tokenizer 1:", list(unique_to_tokenizer1)[:10])
    print("Sample unique to Tokenizer 2:", list(unique_to_tokenizer2)[:10])

# Load two tokenizers
tokenizer1 = AutoTokenizer.from_pretrained("microsoft/phi-4")
tokenizer2 = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")

get_assistant_to_target_input_ids(tokenizer1, tokenizer2)
