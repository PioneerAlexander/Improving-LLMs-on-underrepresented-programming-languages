from strenum import StrEnum

from src.dataset.KotlinCodeCompletionDataset import KotlinCodeCompletionDataset
from src.dataset.CodeXGLUETestDataset import CodeXGLUETestDataset


MAX_LEN = 512


class SpecialTokens(StrEnum):
    EOL_TOKEN = "<EOL>"
    INDENT_TOKEN = "<INDENT>"
    DEDENT_TOKEN = "<DEDENT>"


def test_python_dataset_reading():
    json_filename = "../../test.jsonl"

    dataset = CodeXGLUETestDataset(
        json_with_functions_filename=json_filename,
    )

    print(f"\nDataset length: {len(dataset)}")
    print(f"First element signature: {dataset[0]['signature']}")
    print(f"First element body: {dataset[0]['body']}")
    print(f"Max length: {dataset.corpus['body'].apply(len).max()}")


def test_kotlin_dataset_creation():
    json_filenames = [
        "../../body_signature_pairs.json",
        "../../kotlin_test.json",
        "../../kotlin_train.json"
    ]
    for json_filename in json_filenames[1:]:
        dataset = KotlinCodeCompletionDataset(
            json_with_functions_filename=json_filename,
            max_len=MAX_LEN,
            eol_token=SpecialTokens.EOL_TOKEN,
            indent_token=SpecialTokens.INDENT_TOKEN,
            dedent_token=SpecialTokens.DEDENT_TOKEN,
            is_subset=True,
        )
        print(f"\nJson filename: {json_filename}")
        print(f"Dataset length: {len(dataset)}")
        print(f"First element signature: {dataset[0]['signature']}\n")
        print(f"First element body: {dataset[0]['body']}")
