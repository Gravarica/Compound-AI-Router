# src/routing/datasets/router_dataset.py
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class RouterDataset(Dataset):

    def __init__(self, data: List[Dict[str, Any]], tokenizer: PreTrainedTokenizer, max_length: int = 512):

        self.texts = [item['text'] for item in data]
        self.labels = [item['label'] for item in data]
        self.ids = [item['id'] for item in data]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        encoding['labels'] = torch.tensor(label, dtype=torch.long)

        return encoding

    @property
    def label_distribution(self) -> Dict[int, int]:

        import collections
        return dict(collections.Counter(self.labels))