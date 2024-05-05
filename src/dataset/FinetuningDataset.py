"""
    Tokenized dataset for passing to the PEFT Trainer
"""
from typing import List, Dict

from torch.utils.data import Dataset


class FineTuningDataset(Dataset):
    """
        Every element is fully tokenized function code with separated body and signature
    """

    def __init__(self, code_data: List[Dict]):
        self.corpus = code_data

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, item):
        code = self.corpus[item]
        print(code['input_ids'].shape)
        print(code['attention_mask'].shape)
        print(code['labels'].shape)
        return {
            'input_ids': code['input_ids'],
            'attention_mask': code['attention_mask'],
            'labels': code['labels']
        }
