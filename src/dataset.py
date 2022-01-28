import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.config import Config
from src.const import TARGETS, INPUTS


class TextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, config: Config):
        self.df = df.copy()
        self.vocab = config.vocab
        self.sequence_start = self.vocab([config.start_sent])
        self.sequence_end = self.vocab([config.end_sent])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        token_ids = self.vocab(self.df.iloc[index, 0])
        x = self.sequence_start + token_ids + self.sequence_end
        y = token_ids + self.sequence_end
        return x, y


def get_datasets(config: Config):
    train_dataset = TextDataset(config.datasets[0], config)
    valid_dataset = TextDataset(config.datasets[1], config)
    return train_dataset, valid_dataset


def get_dataloaders(config: Config):
    train_dataset, valid_dataset = get_datasets(config)
    pad_token_id = config.vocab([config.pad_token])

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, pad_token_id),
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, pad_token_id),
    )

    return train_dataloader, valid_dataloader


def collate_fn(batch, pad_token_id):
    max_len = max([len(x) for x, _ in batch])
    padded_x = []
    padded_y = []
    for x, y in batch:
        pad_size = max_len - len(x)
        padded_x.append(x + pad_token_id * pad_size)
        padded_y.append(y + pad_token_id * (pad_size + 1))

    return {INPUTS: torch.tensor(padded_x), TARGETS: torch.tensor(padded_y)}
