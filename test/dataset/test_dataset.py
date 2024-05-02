from strenum import StrEnum

from src.dataset.dataset import KotlinCodeCompletionDataset

MAX_LEN = 512


class SpecialTokens(StrEnum):
    EOL_TOKEN = "<EOL>"
    INDENT_TOKEN = "<INDENT>"
    DEDENT_TOKEN = "<DEDENT>"


def test_dataset_creation():

    json_filename = "../../body_signature_pairs.json"

    dataset = KotlinCodeCompletionDataset(
        json_with_functions_filename=json_filename,
        max_len=MAX_LEN,
        eol_token=SpecialTokens.EOL_TOKEN,
        indent_token=SpecialTokens.INDENT_TOKEN,
        dedent_token=SpecialTokens.DEDENT_TOKEN,
    )

    print(f"\nDataset length: {len(dataset)}\n")
    print(f"\nFirst element signature: {dataset[0]['signature']}\n")
    print(f"\nFirst element body: {dataset[0]['body']}\n")
