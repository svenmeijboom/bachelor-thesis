import os
from typing import Optional

import torch
from tqdm import tqdm

from information_extraction.dtypes import EvaluationResult
from .base import BaseCallback


class ModelCheckpoint(BaseCallback):
    def __init__(self, filename: str, metric: str = 'loss', mode: str = 'min', restore: bool = False,
                 revert_after: Optional[int] = None):
        self.filename = filename
        self.metric = metric
        self.mode = mode
        self.restore = restore
        self.revert_after = revert_after

        if mode not in ('min', 'max'):
            raise ValueError('`mode` must be one of ("min", "max")')

        dirname = os.path.dirname(filename)
        os.makedirs(dirname, exist_ok=True)

        self.best_score = None
        self.steps_since_improve = 0

    def on_validation_end(self, step_num: int, results: EvaluationResult):
        new_score = results.flattened_metrics[self.metric]

        if (self.best_score is None or
                (new_score < self.best_score and self.mode == 'min') or
                (new_score > self.best_score and self.mode == 'max')):
            tqdm.write(f'Saving new best model to {self.filename}...')

            self.best_score = new_score
            torch.save(self.trainer.model.state_dict(), self.filename)

            self.steps_since_improve = 0
        else:
            self.steps_since_improve += 1

            if self.revert_after is not None and self.steps_since_improve >= self.revert_after:
                tqdm.write(f'Reverting checkpoint: {self.metric} did not improve for {self.revert_after} steps...')
                self.trainer.model.load_state_dict(torch.load(self.filename))
                self.steps_since_improve = 0

    def on_train_end(self):
        if self.restore:
            tqdm.write(f'Restoring best model checkpoint from {self.filename}...')
            self.trainer.model.load_state_dict(torch.load(self.filename))
