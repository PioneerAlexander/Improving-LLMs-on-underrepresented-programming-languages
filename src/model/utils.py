"""
   Utils functions for the model training and fine-tuning
"""
import netrc
from typing import Dict, Union, List, Any

import torch
from strenum import StrEnum
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.dataset.KotlinCodeCompletionDataset import KotlinCodeCompletionDataset


class SpecialTokens(StrEnum):
    EOL_TOKEN = "<EOL>"
    INDENT_TOKEN = "<INDENT>"
    DEDENT_TOKEN = "<DEDENT>"


def to_device(inputs: Dict, device: str) -> Dict:
    """
        Returns an updated dict for the input where all tensors in dict
        values are transferred to the specified device
    """

    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}


def preprocess_batch_for_training(batch: Dict, tokenizer, device: str, batch_size: int) -> List[Dict[str, Any]]:
    """
        Encodes all elements in batch, in order to pass to the model and thereafter calculate the loss between
        generated text and targeted text
    """
    encoded_inputs = tokenizer(batch["signature"], return_tensors='pt', padding=True, return_attention_mask=False)
    encoded_labels = tokenizer(batch["body"], return_tensors='pt', padding=True, return_attention_mask=False)
    encoded = to_device(encoded_inputs, device)
    encoded["labels"] = encoded_labels["input_ids"].to(device)

    batch_input_ids = []
    cls_token_id = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
    sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
    for i in range(batch_size):
        batch_input_ids.append(
            [cls_token_id] + encoded_inputs["input_ids"][i].tolist() + [sep_token_id] + encoded["labels"][
                i].tolist() + [sep_token_id]
        )
    print(batch_input_ids)

    return [{"input_ids": torch.tensor(input_ids).to(device), "labels": labels} for input_ids, labels in
            zip(batch_input_ids, encoded["labels"])]


def preprocess_batch_for_eval(signatures: List[str], tokenizer, device, prompt_function):
    """
        For the evaluation we just need to tokenize input
    """
    encoded_inputs = [tokenizer(
        prompt_function(signature=signature),
        return_tensors='pt',
        return_attention_mask=False
    ) for signature in signatures]
    encoded = [to_device(encoded_input, device) for encoded_input in encoded_inputs]

    return encoded


def tokenize_dataset_for_peft(example, tokenizer, device):
    input_ids = tokenizer(example["signature"], return_tensors='pt', padding=True, return_attention_mask=True)
    input_ids = to_device(input_ids, device)
    labels = tokenizer(example["body"], return_tensors='pt', padding=True, return_attention_mask=False)
    input_ids["labels"] = to_device(labels, device)["input_ids"]

    return input_ids


def calculate_loss(outputs, labels):
    """
    Calculates correctly cross entropy loss between the predicted distributions
    and target distributions
    """
    logits = outputs.logits[:, -1 - labels.shape[-1]:-1, :]

    output_logprobs = F.log_softmax(logits, dim=-1)
    outputs_logprobs_flat = output_logprobs.view(-1, outputs.logits.shape[-1])
    labels_flat = labels.view(-1)

    mask = torch.isnan(output_logprobs)
    has_nan = mask.any().item()

    if has_nan:
        print(f"The tensor logprobs contains {mask.sum().item()} NaN values.")
        logits[mask] = 0.0
    else:
        print("The tensor logprobs does not contain NaN values.")
    print(labels)

    return F.nll_loss(outputs_logprobs_flat, labels_flat)


def dialogue_prompt(signature: str) -> str:
    """
        Returns the dialogue start prompt
    """
    return ("Alice: Could you please help me? I am stuck with writing the body " +
            "to the function. Can you please help me to complete it? " +
            f"The function signature is {signature}. Write a body please \nBob:")


def instruct_prompt(instruction: str, signature: str) -> str:
    """
        Returns the instruction-based prompt
    """
    return f"{instruction}. Signature: {signature}"


def get_body_from_dialogue_answer(answer: str, signature: str) -> str:
    """
        The pretrained version of the Phi-1.5 model was not trained on the
        instruction following. It means that with any effort investigated,
        it will more probably not follow our instructions.
        By that reason for the first part of the evaluation
        we get the actual generated code related to the task trimming the model
        answer.
    """
    generated_body = answer
    # Normally, the Phi-1.5 starts the generation with repeating the prompt.
    if "Bob:" in answer:
        generated_body = ''.join(answer.split("Bob:")[1:])

    # Our prompt wraps the model answer to a dialogue generation with Alice and Bob
    if "Alice" in generated_body:
        generated_body = generated_body.split(sep="Alice")[0]

    if signature in generated_body:
        generated_body = generated_body.replace(signature, "")

    return generated_body


def get_body_from_instruct_answer(answer: str, signature: str) -> str:
    """
        The pretrained version of the Phi-1.5 model was not trained on the
        instruction following. It means that with any effort investigated,
        it will more probably not follow our instructions.
        By that reason for the first part of the evaluation
        we get the actual generated code related to the task trimming the model
        answer.
    """
    generated_body = answer

    if "signature" in generated_body:
        generated_body = ''.join(generated_body.split(signature)[1:])

        if "def" in generated_body:
            generated_body = generated_body.split("def")[0]

    if "Exercise" in generated_body:
        generated_body = generated_body.split("Exercise")[0]

    if "Section" in generated_body:
        generated_body = generated_body.split("Section")[0]

    return generated_body


def get_api_key_from_netrc(machine_name):
    try:
        credentials = netrc.netrc()
        auth_tokens = credentials.authenticators(machine_name)
        if auth_tokens:
            _, _, api_key = auth_tokens
            return api_key
        else:
            raise ValueError("No credentials found for the specified machine in netrc file.")
    except FileNotFoundError:
        raise FileNotFoundError("netrc file not found.")


def collate_fn_eval(batch: List[Dict], device) -> Dict:
    max_len_input_ids = max(len(element["input_ids"]) for element in batch)
    input_ids = []
    for element in batch:
        input_id = element["input_ids"].squeeze(0)
        padded_input_id = torch.nn.functional.pad(input_id, (0, max_len_input_ids - input_id.shape[0]), value=0)
        input_ids.append(padded_input_id)

    return {"input_ids": torch.stack(input_ids).to(device)}


class FineTuningDataCollator:

    def __call__(self, batch: List[Dict[str, torch.Tensor]]):
        return self.collate_fn(batch)

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len_input_ids = max(len(element["input_ids"]) for element in batch)
        max_len_labels = max(len(element["labels"]) for element in batch)
        max_len_attn_mask = max(len(element["attention_mask"]) for element in batch)

        batch_max_length = max(max_len_attn_mask, max_len_labels, max_len_input_ids)
        input_ids = []
        attention_mask = []
        labels = []
        for example in batch:
            # Pad input_ids
            input_id = example["input_ids"].squeeze(0)
            padded_input_id = torch.nn.functional.pad(input_id, (0, batch_max_length - input_id.shape[0]), value=0)
            input_ids.append(padded_input_id)

            # Pad attention_mask
            att_mask = example["attention_mask"].squeeze(0)
            padded_att_mask = torch.nn.functional.pad(att_mask, (0, batch_max_length - att_mask.shape[0]), value=0)
            attention_mask.append(padded_att_mask)

            # Pad labels
            label = example["labels"].squeeze(0)
            padded_label = torch.nn.functional.pad(label, (0, batch_max_length - label.shape[0]), value=-100)
            labels.append(padded_label)

        return {
            "input_ids": torch.stack(input_ids).to('cuda'),
            "attention_mask": torch.stack(attention_mask).to('cuda'),
            "labels": torch.stack(labels).to('cuda')
        }
