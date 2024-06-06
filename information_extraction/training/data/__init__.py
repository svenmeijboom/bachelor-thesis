import os
from pathlib import Path

import wandb

from .base import BaseDataset
from .module import SWDEDataModule
from ..trainer import MODELS
from information_extraction.config import DEFAULT_GROUND_TRUTH_DIR_CONF


DEFAULT_GROUND_TRUTH_DIR = DEFAULT_GROUND_TRUTH_DIR_CONF


def get_dataset(config: wandb.Config) -> SWDEDataModule:
    if ':' in config.representation:
        representation, tag = config.representation.split(':')
    else:
        representation = config.representation
        tag = 'latest'

    #Old one
    #data_path = Path(f'/vol/csedu-nobackup/other/smeijboom/bachelor-thesis/Data/MovieName-set-{representation}-{tag}').expanduser()
    #input_artifact = wandb.use_artifact(f'MovieName-{representation}-zero-shot:{tag}')

    #SWDE test/val
    data_path = Path(f'/vol/csedu-nobackup/other/smeijboom/bachelor-thesis/Data/MovieName-swde200').expanduser()
    input_artifact = wandb.use_artifact(f'MovieName-swde200:{tag}')

    #WDC test/val
    #data_path = Path(f'/vol/csedu-nobackup/other/smeijboom/bachelor-thesis/Data/MovieName-wdc1000').expanduser()
    #input_artifact = wandb.use_artifact(f'MovieName-wdc1000:{tag}')
    
    input_artifact.download(str(data_path))

    split_mode = config.get('split_mode', 'random')

    if split_mode == 'random':
        website = config.get('website', '*')
        train_files = list(data_path.glob(f'train/{config.vertical}/{website}/*-{config.context_size}.csv'))
        val_files = list(data_path.glob(f'val/{config.vertical}/{website}/*-{config.context_size}.csv'))
        test_files = list(data_path.glob(f'test/{config.vertical}/{website}/*-{config.context_size}.csv'))

        print('Files used for training:', len([str(f) for f in train_files]))
        print('Files used for validation:', len([str(f) for f in val_files]))
        print('Files used for testing:', len([str(f) for f in test_files]))
    elif split_mode == 'zero_shot':
        if 'split_size' not in config or 'split_num' not in config:
            raise ValueError(f'Cannot do zero shot split without `split_size` and `split_num` parameters!')

        websites = os.listdir(data_path / 'train' / config.vertical)
        websites = websites[config.split_num * config.split_size:] + websites[:config.split_num * config.split_size]

        train_websites = websites[2 * config.split_size:]
        val_websites = websites[config.split_size:2 * config.split_size]
        test_websites = websites[:config.split_size]

        print('Websites used for training:', train_websites)
        print('Websites used for validation:', val_websites)
        print('Websites used for testing:', test_websites)

        train_files = [filename for website in train_websites
                       for filename in data_path.glob(f'*/{config.vertical}/{website}/*-{config.context_size}.csv')]
        val_files = [filename for website in val_websites
                     for filename in data_path.glob(f'*/{config.vertical}/{website}/*-{config.context_size}.csv')]
        test_files = [filename for website in test_websites
                      for filename in data_path.glob(f'*/{config.vertical}/{website}/*-{config.context_size}.csv')]
    else:
        raise ValueError(f'Split mode {split_mode} not found!')

    mini_batch_sizes = {
        'bert': 8 if config.context_size == 512 else 32,
        't5': 4 if config.context_size == 512 else 16,
    }

    dataset = SWDEDataModule(MODELS[config.model]['model_version'],
                             max_length=config.context_size,
                             train_files=train_files,
                             val_files=val_files,
                             test_files=test_files,
                             batch_size=mini_batch_sizes[config.model],
                             num_workers=config.num_workers,
                             remove_null=config.remove_null)

    return dataset
