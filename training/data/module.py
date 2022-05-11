from collections import Counter
from pathlib import Path
from typing import List, Optional, Union

from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler

from data.bert import BertDataset
from data.sampler import SWDESampler
from data.t5 import T5Dataset


class SWDEDataModule:
    def __init__(self,
                 model_version: str,
                 max_length: int,
                 train_files: List[Union[str, Path]],
                 val_files: Optional[List[Union[str, Path]]] = None,
                 test_files: Optional[List[Union[str, Path]]] = None,
                 batch_size: int = 32,
                 num_workers: int = 2,
                 remove_null: bool = False):
        super().__init__()
        self.model_version = model_version
        self.max_length = max_length
        self.train_files = train_files
        self.val_files = val_files or []
        self.test_files = test_files or []
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.remove_null = remove_null

        if 'bert' in model_version.lower():
            self.architecture = 'bert'
        elif 't5' in model_version.lower():
            self.architecture = 't5'
        else:
            raise ValueError(f'Unsupported model version: {model_version}')

        self.data_train, self.data_val, self.data_test = self.prepare_datasets()

    def prepare_datasets(self):
        dataset_classes = {
            'bert': BertDataset,
            't5': T5Dataset,
        }

        return tuple(
            dataset_classes[self.architecture](
                files,
                self.model_version,
                max_length=self.max_length,
                remove_null=self.remove_null,
            )
            for files in [self.train_files, self.val_files, self.test_files]
        )

    def train_dataloader(self, sample: bool = True):
        if not sample:
            # Running evaluation on the training data, so we don't want random sampling
            return DataLoader(self.data_train, batch_size=self.batch_size,
                              num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

        sampler = SWDESampler(self.data_train, self.batch_size, replacement=True, remove_null=self.remove_null)

        return DataLoader(self.data_train, batch_size=self.batch_size, sampler=sampler,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def val_dataloader(self, size: Optional[int] = None):
        if size is None:
            return DataLoader(self.data_val, batch_size=self.batch_size,
                              num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

        sampler = SWDESampler(self.data_val, size, replacement=False, remove_null=self.remove_null)

        return DataLoader(self.data_val, batch_size=self.batch_size, sampler=sampler,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True)
