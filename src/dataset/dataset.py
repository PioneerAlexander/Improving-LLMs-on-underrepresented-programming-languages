"""
    The Kotlin code completion dataset implementation
"""

from torch.utils.data import Dataset

from src.dataset.preprocess import (read_dataframe_from_json, discard_functions_with_long_bodies,
                                    replace_indentation_and_eol_symbols)


class KotlinCodeCompletionDataset(Dataset):

    def __init__(
            self,
            json_with_functions_filename: str,
            max_len: int,
            indent_token: str,
            dedent_token: str,
            eol_token: str
    ):
        self.json_with_functions_filename = json_with_functions_filename

        self.max_len = max_len

        self.indent_token = indent_token
        self.dedent_token = dedent_token
        self.end_of_line_token = eol_token

        self.corpus = self.build_corpus()

    def build_corpus(self):
        # Read dataframe signature body pairs from each parsed function from Kotlin repository
        df = read_dataframe_from_json(filename=self.json_with_functions_filename)

        # Keep only functions where body has at most max_len length
        df = discard_functions_with_long_bodies(max_len=self.max_len, df=df)

        #  For each body replace the indentation and end of line symbols
        #  with special tokens
        df["body"] = df["body"].apply(lambda body: replace_indentation_and_eol_symbols(
            body=body,
            indent_token=self.indent_token,
            dedent_token=self.dedent_token,
            end_of_line_token=self.end_of_line_token,
        ))

        return df

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, item):
        return self.corpus.iloc[item]
