"""
    Model fine-tuning pipeline
"""
from functools import partial
from typing import Optional

import torch
import tyro
import wandb
from transformers import AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType

from src.dataset.FinetuningDataset import FineTuningDataset
from src.dataset.KotlinCodeCompletionDataset import KotlinCodeCompletionDataset
from src.model.utils import SpecialTokens, tokenize_dataset_for_peft, FineTuningDataCollator


def main(
        model_load_file_name: str,
        tokenizer_load_path_name: str,
        train_dataset_name: str,

        /,  # Mark the end of positional arguments

        max_len: int = 512,
        batch_size: int = 32,
        lr: float = 0.001,
        weight_decay: float = 0.01,

        epochs: int = 1,

        use_wandb: bool = True,
        wandb_project: Optional[str] = "jb_llm_test_task",

):
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

    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32,
                             lora_dropout=0.1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = torch.load(model_load_file_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path_name)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    special_tokens_dict = {"cls_token": "<CLS>", "sep_token": "<SEP>"}

    num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    print("We have added", num_added_tokens, "tokens")
    model.resize_token_embeddings(len(tokenizer))

    assert tokenizer.cls_token == "<CLS>"

    train_dataset = KotlinCodeCompletionDataset(
        json_with_functions_filename=train_dataset_name,
        max_len=max_len,
        eol_token=SpecialTokens.EOL_TOKEN,
        indent_token=SpecialTokens.INDENT_TOKEN,
        dedent_token=SpecialTokens.DEDENT_TOKEN,
        is_subset=True,
    )

    training_args = TrainingArguments(
        output_dir="your-name/bigscience/mt0-large-lora",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=32,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb"
    )

    train_dataset_lst = train_dataset.map(partial(tokenize_dataset_for_peft, tokenizer=tokenizer, device=device))

    tokenized_dataset = FineTuningDataset(train_dataset_lst)

    data_collator = FineTuningDataCollator()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,
    )

    trainer.train()

    torch.save(model, 'fine-tuned_phi-1_5.pth')
    output_dir = "./model"
    trainer.save_model(output_dir)
    wandb.finish()


if __name__ == "__main__":
    tyro.cli(main)
