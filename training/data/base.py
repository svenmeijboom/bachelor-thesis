from abc import ABC, abstractmethod
from pathlib import Path
import time
from typing import Iterable, Iterator, List, Optional, Tuple, Union

import pandas as pd

from torch.utils.data import Dataset

from transformers import PreTrainedTokenizer


class BaseDataset(Dataset, ABC):
    docs: List[str] = []
    inputs: List[str] = []
    targets: List[str] = []
    features: List[str] = []

    indices_with_data: List[int] = []
    indices_without_data: List[int] = []

    def __init__(self, files: Iterable[Union[str, Path]], tokenizer: PreTrainedTokenizer,
                 remove_null: bool = False, max_length: Optional[int] = None):
        self.files = list(map(Path, files))
        self.tokenizer = tokenizer
        self.remove_null = remove_null

        self.tokenize_kwargs = {'return_tensors': 'pt', 'padding': True}
        if max_length is not None:
            self.tokenize_kwargs['truncation'] = True
            self.tokenize_kwargs['max_length'] = max_length

        if self.files:
            print(f'Loading {self.__class__.__name__}...', end=' ', flush=True)
            start = time.time()

            self.docs, self.inputs, self.targets, self.features = self.read_data()
            self.prepare_inputs()

            self.indices_with_data = [i for i, t in enumerate(self.targets) if t]
            self.indices_without_data = [i for i, t in enumerate(self.targets) if not t]

            end = time.time()
            print(f'done in {end - start:.1f}s')

    def prepare_inputs(self):
        pass

    def read_data(self):
        return [list(items) for items in zip(*(
            sample
            for file in self.files
            for sample in self.read_csv(file)
        ))]

    def read_csv(self, file: Path) -> Iterator[Tuple[str, ...]]:
        df = pd.read_csv(file, dtype=str).fillna('')

        features = [col for col in df.columns if col not in ('doc_id', 'text')]

        for _, row in df.iterrows():
            for feature in features:
                if row[feature] or not self.remove_null:
                    yield row['doc_id'], row['text'], row[feature], feature

    def __len__(self):
        return len(self.inputs)

    @abstractmethod
    def __getitem__(self, index: int):
        raise NotImplementedError
