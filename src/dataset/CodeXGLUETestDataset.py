"""
    This class preprocesses a CodeXGLUE test dataset which was downloaded from LFS
    https://huggingface.co/datasets/microsoft/codexglue_method_generation/tree/main
    as JSON lines file.
"""
from torch.utils.data import Dataset

from src.dataset.preprocess import read_dataframe_from_json


class CodeXGLUETestDataset(Dataset):
    def __init__(
            self,
            json_with_functions_filename: str,
    ):
        self.json_with_functions_filename = json_with_functions_filename
        self.corpus = self.build_corpus()

    def build_corpus(self):
        df = read_dataframe_from_json(
            self.json_with_functions_filename,
            lines=True  # dataset is stored in JSON lines file
        )
        return df

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, item):
        row = self.corpus.iloc[item]
        return {"signature": row["signature"], "body": row["body"]}
