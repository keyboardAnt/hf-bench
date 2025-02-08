from functools import cache
from transformers import AutoModelForCausalLM, AutoTokenizer


class VocabError(Exception):
    """
    Exception raised when we are unable to determine if two tokenizers are identical.
    """


@cache
def eq_tokenizers(
    m1: AutoModelForCausalLM,
    m2: AutoModelForCausalLM,
    t1: AutoTokenizer,
    t2: AutoTokenizer,
) -> bool:
    """
    Check if two models have the same tokenizer.
    """
    try:
        m1._validate_assistant(
            assistant_model=m2,
            tokenizer=t1,
            assistant_tokenizer=t2,
        )
        return False
    except ValueError as e:
        if (
            "`assistant_tokenizer` is not required when the main and assistant models use the same tokenizer."
            in str(e)
        ):
            return True
        elif "The main and assistant moedels have different tokenizers." not in str(e):
            raise VocabError(e)
