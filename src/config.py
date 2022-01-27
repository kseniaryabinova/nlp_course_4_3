import typing as tp
import re

from pydantic import BaseModel, validator
import pandas as pd
from torchtext.vocab import Vocab, build_vocab_from_iterator


def _preprocess(dataset_filepath: str):
    df = pd.read_csv(dataset_filepath, header=None, sep='\n')
    df.rename(columns={0: 'text'}, inplace=True)
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].apply(lambda x: re.sub(r",|\.|!|\?", '', x).split(' '))
    return df


class Config(BaseModel):
    seed: int = 25

    batch_size: int = 32
    epochs: int = 20
    lr: float = 0.001

    embedding_dim: int = 50
    hidden_size: int = 50
    num_layers: int = 1
    dropout: float = 0.
    bidirectional: bool = False

    dataset_path: str = 'data/author_quotes.txt'

    start_sent: str = '<sos>'
    end_sent: str = '<eos>'
    pad_token: str = '<pad>'
    unk_token: str = '<unk>'

    datasets: tp.Optional[tp.Tuple[pd.DataFrame, pd.DataFrame]] = None
    vocab: tp.Optional[Vocab] = None

    class Config:
        arbitrary_types_allowed = True

    @validator('datasets', always=True)
    def init_dataset(cls, v, values):
        df = _preprocess(values['dataset_path'])
        return df[:30000], df[30000:]

    @validator('vocab', always=True)
    def init_vocab(cls, v, values):
        df = _preprocess(values['dataset_path'])
        vocab: Vocab = build_vocab_from_iterator(
            iter(df['text']),
            specials=[
                values['pad_token'],
                values['unk_token'],
                values['start_sent'],
                values['end_sent'],
            ]
        )
        vocab.set_default_index(vocab['<unk>'])
        return vocab
