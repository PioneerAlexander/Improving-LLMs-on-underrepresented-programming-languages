"""
    This module contains the run code for the evaluation the loaded model and tokenizer on given
    dataset.
"""
from functools import partial
from typing import Optional

import nltk
import torch
import tyro
import wandb
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset.KotlinCodeCompletionDataset import KotlinCodeCompletionDataset
from src.dataset.CodeXGLUETestDataset import CodeXGLUETestDataset
from src.dataset.preprocess import decode_function_body
from src.model.utils import SpecialTokens, preprocess_batch_for_eval, \
    instruct_prompt, get_body_from_instruct_answer, collate_fn_eval
from src.model.metrics import bleu_score, exact_match, edit_similarity


def main(
        model_load_path: str,
        tokenizer_load_path: str,
        dataset_load_mode: str,
        dataset_load_path: str,

        /,  # Mark the end of positional arguments

        max_len: Optional[int] = 512,
        batch_size: Optional[int] = 32,
        finetuned: bool = False,

        use_wandb: bool = True,
        wandb_project: Optional[str] = "jb_llm_test_task",

        log_output_every_steps: int = 10,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    assert (not use_wandb) or (use_wandb and wandb_project is not None)

    if use_wandb:
        # Log in to your wandb account with netrc file in your home directory
        # wandb.login(key=get_api_key_from_netrc("api.wandb.ai"))
        wandb.init(
            project=wandb_project,
            reinit=True,
            resume="allow",
            entity="kariakinaleksandr"  # Change to your_username for reproduction
        )

    model = torch.load(model_load_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if dataset_load_mode == "kotlin":
        test_dataset = KotlinCodeCompletionDataset(
            json_with_functions_filename=dataset_load_path,
            max_len=max_len,
            eol_token=SpecialTokens.EOL_TOKEN,
            indent_token=SpecialTokens.INDENT_TOKEN,
            dedent_token=SpecialTokens.DEDENT_TOKEN,
            is_subset=True,
        )

    elif dataset_load_mode == "python":
        test_dataset = CodeXGLUETestDataset(
            json_with_functions_filename=dataset_load_path
        )

    else:
        raise AttributeError('Only "python" and "kotlin" dataset load modes are supported.'
                             f'Found "{dataset_load_mode}".')

    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    nltk.download('punkt')

    instruction = ""
    if dataset_load_mode == "kotlin":
        instruction = ("Complete a Kotlin function body by given its signature. Note that this is a Kotlin function. "
                       "You need to implement only on Kotlin body of the function with Kotlin signature")
    if dataset_load_mode == "python":
        instruction = "Implement a body of the function with signature"

    instruct_prompt_fn = partial(instruct_prompt, instruction=instruction)
    with tqdm(enumerate(dataloader)) as progressbar:
        for step, batch in progressbar:
            signatures = batch["signature"]
            if not finetuned and dataset_load_mode == "python":
                # Some signatures from the python dataset are also encoded with special tokens
                # (<EOL>, <INDENT>, <DEDENT>)
                signatures = [decode_function_body(signature, end_of_line_token=SpecialTokens.EOL_TOKEN,
                                                   indent_token=SpecialTokens.INDENT_TOKEN,
                                                   dedent_token=SpecialTokens.DEDENT_TOKEN) for signature in signatures]
            bodies = batch["body"]

            # Preprocess all signatures in parallel
            model_inputs = preprocess_batch_for_eval(signatures, tokenizer, device, prompt_function=instruct_prompt_fn)
            model_inputs = collate_fn_eval(model_inputs, device)
            # Generate outputs in parallel
            with torch.no_grad():
                outputs = model.generate(**model_inputs, max_length=max_len)

            # Decode outputs in parallel
            decoded_outputs = tokenizer.batch_decode(outputs)

            generated_code = ""
            reference = ""
            for idx, (output_text, reference) in enumerate(zip(decoded_outputs, bodies)):
                if not finetuned:
                    generated_code = get_body_from_instruct_answer(output_text, signatures[idx])
                    reference = decode_function_body(reference,
                                                     end_of_line_token=SpecialTokens.EOL_TOKEN,
                                                     indent_token=SpecialTokens.INDENT_TOKEN,
                                                     dedent_token=SpecialTokens.DEDENT_TOKEN)
                else:
                    generated_code = output_text

                # Calculate metrics
                bleu = bleu_score(reference, generated_code)
                match = exact_match(reference, generated_code)
                edit_sim = edit_similarity(reference, generated_code)

                wandb.log({"BLEU": bleu, "Exact match": match, "Edit similarity:": edit_sim, "step": step})
            if step % log_output_every_steps == 0:
                # Log prediction and reference
                print(f"Generated code: {generated_code}")
                print("-" * 50)
                print(f"Reference: {reference}")
                print("-" * 50)
                wandb.log({"Output": str(generated_code)})


if __name__ == "__main__":
    tyro.cli(main)
