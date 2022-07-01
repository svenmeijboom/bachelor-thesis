from typing import Optional

from torch.utils.data import DataLoader
from tqdm import tqdm

from information_extraction.evaluation import Evaluator
from .base import BaseCallback


class ValidationCallback(BaseCallback):
    def __init__(
        self,
        evaluator: Evaluator,
        validation_interval: int,
        segment_loader: Optional[DataLoader] = None,
        document_loader: Optional[DataLoader] = None,
        label: Optional[str] = None,
    ):
        self.evaluator = evaluator
        self.validation_interval = validation_interval
        self.total_steps = 0

        if segment_loader and document_loader:
            raise ValueError('Only one of `segment_loader` and `document_loader` can be provided...')

        if segment_loader is not None:
            self.method = 'segments'
            self.dataloader = segment_loader
        elif document_loader is not None:
            self.method = 'documents'
            self.dataloader = document_loader
        else:
            raise ValueError('At least one of `segment_loader` and `document_loader` must be provided...')

        if label is not None:
            self.dataloader = tqdm(self.dataloader, desc=label, leave=False)

    def on_train_start(self, run_params: dict):
        self.total_steps = run_params['num_steps']
        self.perform_validation(0)

    def on_step_end(self, step_num: int, loss: float):
        if step_num % self.validation_interval == 0:
            self.perform_validation(step_num)

    def perform_validation(self, step_num: int):
        if self.method == 'segments':
            predictions = self.trainer.predict_segments(self.dataloader)
            results = self.evaluator.evaluate_segments(predictions)
        else:
            predictions = self.trainer.predict_documents(self.dataloader)
            results = self.evaluator.evaluate_documents(predictions)

        formatted = ', '.join(f'val/{metric}: {value:.2f}' for metric, value in results.flattened_metrics.items())
        num = str(step_num).rjust(len(str(self.total_steps)))

        tqdm.write(f'[Step {num}] {formatted}')

        for callback in self.trainer.callbacks:
            callback.on_validation_end(step_num, results)
