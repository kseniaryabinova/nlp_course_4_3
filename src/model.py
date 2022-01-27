import torch
from torch import Tensor
from torch import nn

from src.config import Config


class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(
            num_embeddings=len(config.vocab),
            embedding_dim=config.embedding_dim,
            padding_idx=config.vocab.get_default_index(),
        )
        self.backbone = nn.Sequential(
            nn.GRU(
                input_size=config.embedding_dim,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                bias=True,
                batch_first=True,
                dropout=config.dropout,
                bidirectional=config.bidirectional,
            ),
        )
        self.head = nn.Linear(
            in_features=config.hidden_size,
            out_features=len(config.vocab),
            bias=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        x = self.backbone(x)[0]
        x = self.head(x)
        x = torch.permute(x, (0, 2, 1))
        return x
