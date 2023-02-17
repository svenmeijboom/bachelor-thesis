from typing import List

from .base import BaseCallback
from .logging import LoggingCallback
from .model_checkpoint import ModelCheckpoint
from .wandb import WandbCallback
from .early_stopping import EarlyStopping
from .save_predictions import SavePredictions
from .validation import ValidationCallback

from information_extraction.data.metrics import compute_f1, compute_exact
from information_extraction.training.data import SWDEDataModule
from information_extraction.evaluation import Evaluator
from information_extraction.config import WANDB_PROJECT

import wandb


def get_callbacks(dataset: SWDEDataModule, run_name: str, config: wandb.Config) -> List[BaseCallback]:
    monitor_mode = 'min' if config.monitor_metric.endswith('loss') else 'max'

    callbacks = [
        ModelCheckpoint(f'models/{run_name}.state_dict', restore=True,
                        metric=config.monitor_metric, mode=monitor_mode,
                        revert_after=config.get('revert_after')),
        WandbCallback(WANDB_PROJECT, run_name, do_init=False),
        EarlyStopping(patience=config.early_stopping_patience,
                      metric=config.monitor_metric, mode=monitor_mode),
        SavePredictions(documents_dir=f'results/{run_name}', segments_dir=f'results/{run_name}'),
    ]

    evaluator = Evaluator(metrics={'f1': compute_f1, 'em': compute_exact})

    if 'validation_documents' in config:
        val_callback = ValidationCallback(evaluator, config.validation_interval,
                                          document_loader=dataset.val_document_dataloader(config.validation_documents),
                                          label='Validating')
        callbacks.append(val_callback)
    elif 'validation_batches' in config:
        val_callback = ValidationCallback(evaluator, config.validation_interval,
                                          segment_loader=dataset.val_dataloader(
                                              size=config.validation_batches * config.batch_size
                                          ),
                                          label='Validating')
        callbacks.append(val_callback)

    return callbacks
