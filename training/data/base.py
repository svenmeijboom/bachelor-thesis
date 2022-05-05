from abc import ABC, abstractmethod
import time
from typing import List, Optional

import pandas as pd

from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    def __init__(self, files: List[str], remove_null: bool = False, max_length: Optional[int] = None):
        self.files = files
        self.remove_null = remove_null

        self.tokenize_kwargs = {'return_tensors': 'pt', 'padding': True}
        if max_length is not None:
            self.tokenize_kwargs['truncation'] = True
            self.tokenize_kwargs['max_length'] = max_length

        print(f'Initializing {self.__class__.__name__}...', end=' ', flush=True)
        start_time = time.time()

        self.inputs, self.targets, self.features = self.read_data()
        self.prepare_inputs()

        end_time = time.time()
        print(f'done in {end_time-start_time:.1f}s')

    @abstractmethod
    def prepare_inputs(self):
        raise NotImplementedError

    def read_data(self):
        all_x, all_y, all_f = [], [], []

        for csv_file in self.files:
            x, y, f = self.read_csv(csv_file)
            all_x.extend(x)
            all_y.extend(y)
            all_f.extend(f)

        return all_x, all_y, all_f

    def read_csv(self, csv_file: str):
        df = pd.read_csv(csv_file, dtype=str).fillna('')

        features = [col for col in df.columns if col not in ('id', 'text')]

        x = []
        y = []
        f = []

        for _, row in df.iterrows():
            for feature in features:
                if row[feature] or not self.remove_null:
                    x.append(row['text'])
                    y.append(row[feature])
                    f.append(feature)

        return x, y, f

    def __len__(self):
        return len(self.inputs)

    @abstractmethod
    def __getitem__(self, index: int):
        raise NotImplementedError
