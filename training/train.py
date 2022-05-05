from pathlib import Path

from transformers.data.metrics.squad_metrics import compute_f1, compute_exact

import wandb

from callbacks import EarlyStopping, ModelCheckpoint, WandbCallback
from data.module import SWDEDataModule
from metrics import f1_metric, em_metric
from trainer import BertTrainer, T5Trainer


def main():
    wandb.init()

    config = wandb.config

    models = {
        'bert': {
            'model_version': 'SpanBERT/spanbert-base-cased',
            'trainer_class': BertTrainer,
            'mini_batch_size': 16,
        },
        't5': {
            'model_version': 't5-base',
            'trainer_class': T5Trainer,
            'mini_batch_size': 8,
        }
    }

    model_config = models[config.model]

    if config.use_html:
        artifact_name = 'swde-html'
        data_path = Path('~/Data/SWDE-html/').expanduser()
    else:
        artifact_name = 'swde-text'
        data_path = Path('~/Data/SWDE-text/').expanduser()

    input_artifact = wandb.use_artifact(f'{artifact_name}:latest')
    input_artifact.download(str(data_path))

    # TODO: use larger dataset instead of single file
    train_files = list(data_path.glob(f'train/book/barnesandnoble-{config.context_size}.csv'))
    val_files = list(data_path.glob(f'val/book/abebooks-{config.context_size}.csv'))
    test_files = list(data_path.glob(f'test/book/waterstones-{config.context_size}.csv'))

    print(train_files)
    print(val_files)
    print(test_files)

    dataset = SWDEDataModule(model_config['model_version'],
                             max_length=config.context_size,
                             train_files=train_files,
                             val_files=val_files,
                             test_files=test_files,
                             batch_size=model_config['mini_batch_size'],
                             num_workers=config.num_workers,
                             remove_null=config.remove_null)

    monitor_mode = 'min' if config.monitor_metric == 'loss' else 'max'

    html_slug = 'html' if config.use_html else 'text'
    slug = f'{config.model}-{config.context_size}-{html_slug}'
    callbacks = [
        ModelCheckpoint(f'models/{slug}.state_dict', restore=True,
                        metric=config.monitor_metric, mode=monitor_mode),
        WandbCallback('information_extraction', slug, do_init=False),
        EarlyStopping(patience=config.early_stopping_patience,
                      metric=config.monitor_metric, mode=monitor_mode),
    ]

    trainer_kwargs = {
        'train_loader': dataset.train_dataloader(),
        'val_loader': dataset.val_dataloader(size=150 * config.batch_size),
        'validate_per_steps': config.validation_interval,
        'metrics': {
            'f1': f1_metric,
            'em': em_metric,
        },
        'instance_metrics': {
            'f1': compute_f1,
            'em': compute_exact,
        },
        'learning_rate': config.learning_rate,
        'optimizer': config.optimizer,
        'callbacks': callbacks,
    }

    with model_config['trainer_class'](model_config['model_version'], **trainer_kwargs) as trainer:
        trainer.train(config.num_steps, config.batch_size)
        trainer.perform_evaluation(
            train=dataset.train_dataloader(sample=False),
            val=dataset.val_dataloader(),
            test=dataset.test_dataloader(),
        )

    wandb.finish()


if __name__ == '__main__':
    main()
