from pathlib import Path

import wandb

from callbacks import EarlyStopping, ModelCheckpoint, WandbCallback
from data.module import SWDEDataModule
from metrics import compute_f1, compute_exact, f1_metric, em_metric
from trainer import BertTrainer, T5Trainer


def main():
    wandb.init(job_type='train')

    config = wandb.config

    run_name = config.run_name.format(**dict(config))

    wandb.run.name = run_name

    data_path = Path(f'~/Data/SWDE-{config.representation}').expanduser()
    input_artifact = wandb.use_artifact(f'swde-{config.representation}:latest')
    input_artifact.download(str(data_path))

    train_files = list(data_path.glob(f'train/{config.vertical}/*-{config.context_size}.csv'))
    val_files = list(data_path.glob(f'val/{config.vertical}/*-{config.context_size}.csv'))
    test_files = list(data_path.glob(f'test/{config.vertical}/*-{config.context_size}.csv'))

    print('Files used for training:', [str(f) for f in train_files])
    print('Files used for validation:', [str(f) for f in val_files])
    print('Files used for testing:', [str(f) for f in test_files])

    models = {
        'bert': {
            'model_version': 'SpanBERT/spanbert-base-cased',
            'trainer_class': BertTrainer,
            'mini_batch_size': 16 if config.context_size == 512 else 64,
        },
        't5': {
            'model_version': 't5-base',
            'trainer_class': T5Trainer,
            'mini_batch_size': 8 if config.context_size == 512 else 32,
        }
    }

    model_config = models[config.model]

    dataset = SWDEDataModule(model_config['model_version'],
                             max_length=config.context_size,
                             train_files=train_files,
                             val_files=val_files,
                             test_files=test_files,
                             batch_size=model_config['mini_batch_size'],
                             num_workers=config.num_workers,
                             remove_null=config.remove_null)

    monitor_mode = 'min' if config.monitor_metric == 'loss' else 'max'

    callbacks = [
        ModelCheckpoint(f'models/{run_name}.state_dict', restore=True,
                        metric=config.monitor_metric, mode=monitor_mode),
        WandbCallback('information_extraction', run_name, do_init=False),
        EarlyStopping(patience=config.early_stopping_patience,
                      metric=config.monitor_metric, mode=monitor_mode),
    ]

    trainer_kwargs = {
        'train_loader': dataset.train_dataloader(),
        'val_loader': dataset.val_dataloader(size=config.validation_batches * config.batch_size),
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
