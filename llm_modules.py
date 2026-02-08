import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(
    model_path : str,
) -> AutoModelForCausalLM:
    """
    Returns an LLM model loaded onto GPU for use, specified by whatever is on the
    model path.

    Parameters:
        model_path : string, specifies where the models live. I store most of my models locally.
            An example model path might be: "./models/qwen3-1point7b/"

    Returns:
        tokenizer : AutoTokenizer.
        model : AutoModelForCausalLM.
    """
    model_name = "./models/qwen3-1point7b/"

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    return tokenizer, model
    
def hook(_, __, output) -> torch.Tensor:
    """
    Creates a hook to grab the output of a particular layer, retaining the gradients
    """
    global activation
    activation = output