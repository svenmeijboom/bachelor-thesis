import wandb

from .base import BaseTrainer
from .bert import BertTrainer
from .t5 import T5Trainer


MODELS = {
    'bert': {
        'model_version': 'SpanBERT/spanbert-base-cased',
        'trainer_class': BertTrainer,
    },
    't5': {
        'model_version': 't5-base',
        'trainer_class': T5Trainer,
    }
}


def get_trainer(config: wandb.Config) -> BaseTrainer:
    trainer_kwargs = {
        'learning_rate': config.learning_rate,
        'optimizer': config.optimizer,
        'num_beams': config.get('num_beams'),
    }

    model_config = MODELS[config.model]

    return model_config['trainer_class'](model_config['model_version'], **trainer_kwargs)
