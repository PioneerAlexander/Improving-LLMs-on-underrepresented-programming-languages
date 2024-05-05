"""
    This module creates the model and tokenizer from the HuggingFace Phi-1.5 model
    https://huggingface.co/microsoft/phi-1_5.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(
        model_name: str,
        model_save_name: str,
        tokenizer_save_path_name: str,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto").to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    torch.save(model, model_save_name)
    tokenizer.save_pretrained(tokenizer_save_path_name)


if __name__ == "__main__":
    main(
        model_name="microsoft/phi-1_5",
        model_save_name="phi-1_5.pth",
        tokenizer_save_path_name="phi-1_5-tokenizer/",
    )
