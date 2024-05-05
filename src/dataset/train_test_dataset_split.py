"""
    This module's aim is to define the split of the collected Kotlin dataset only one time,
    in order to fix the random in test part and train part definition.
"""
import pandas as pd
from torch.utils.data import random_split

from src.dataset.KotlinCodeCompletionDataset import KotlinCodeCompletionDataset
from src.model.utils import SpecialTokens


def save_subset_into_json(subset, filename: str):
    data = [{
        "signature": subset.dataset[idx]["signature"],
        "body": subset.dataset[idx]["body"]
    } for idx in subset.indices]
    df = pd.DataFrame(data)
    df.to_json(filename)


def main(
        dataset_load_path: str,
        max_len: int,
):
    dataset = KotlinCodeCompletionDataset(
        json_with_functions_filename=dataset_load_path,
        max_len=max_len,
        eol_token=SpecialTokens.EOL_TOKEN,
        indent_token=SpecialTokens.INDENT_TOKEN,
        dedent_token=SpecialTokens.DEDENT_TOKEN,
    )

    train_dataset, test_dataset = random_split(dataset, lengths=(len(dataset) - 20000, 20000))
    save_subset_into_json(train_dataset, "kotlin_train.json")
    save_subset_into_json(test_dataset, "kotlin_test.json")


if __name__ == "__main__":
    main(
        dataset_load_path='body_signature_pairs.json',
        max_len=512,
    )
