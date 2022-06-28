from collections import namedtuple
import os
from pathlib import Path
import sys
import tempfile

import pandas as pd
import wandb

from sklearn.model_selection import train_test_split

from transformers import T5Tokenizer

from data.module import SWDEDataModule
from data.ground_truths import GroundTruths
from evaluation import Evaluator
from metrics import compute_f1, compute_exact
from trainer import BertTrainer, T5Trainer
from train import get_trainer, perform_evaluation

sys.path.append('../preprocessing')

from feature_extraction.text import TextExtractor
from feature_extraction.html import HtmlExtractor, SimpleHtmlExtractor


Config = namedtuple('Config', ['encode_id', 'encode_class', 'encode_tag_subset', 'split_attributes'])

GENERAL_TAGS = {'div', 'span', 'a', 'p'}


configs = {
    'html-base': Config(False, False, None, False),
    'html-id': Config(True, False, None, False),
    'html-class': Config(False, True, None, False),
    'html-id-class': Config(True, True, None, False),
    'html-id-class-subset': Config(True, True, GENERAL_TAGS, False),
    'html-id-expanded': Config(True, False, None, True),
    'html-class-expanded': Config(False, True, None, True),
    'html-id-class-expanded': Config(True, True, None, True),
    'html-id-class-expanded-subset': Config(True, True, GENERAL_TAGS, True),
}


def get_dataset(config: wandb.Config):
    data_path = Path('~/Data/SWDE-split').expanduser()
    input_artifact = wandb.use_artifact('train-val-test-split:latest')
    input_artifact.download(str(data_path))

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir)

        tokenizer = T5Tokenizer.from_pretrained('t5-base')

        if config.representation == 'text':
            extractor = TextExtractor(data_path, output_path, tokenizer,
                                      max_length=config.context_size, num_workers=config.num_workers)
        elif config.representation == 'html-simple':
            extractor = SimpleHtmlExtractor(data_path, output_path, tokenizer,
                                            max_length=config.context_size, num_workers=config.num_workers)
        else:
            extractor_config = configs[config.representation]
            extractor = HtmlExtractor(data_path, output_path, tokenizer,
                                      parent_depth=config.parent_depth,
                                      encode_id=extractor_config.encode_id,
                                      encode_class=extractor_config.encode_class,
                                      encode_tag_subset=extractor_config.encode_tag_subset,
                                      split_attributes=extractor_config.split_attributes,
                                      max_length=config.context_size, num_workers=config.num_workers)

        dfs = [
            pd.read_csv(extractor.process_directory('train', config.vertical, website))
            for website in os.listdir(data_path / 'train' / config.vertical)
        ]

        df = pd.concat(dfs)

        df_train, df_val_test = train_test_split(df, test_size=0.2)
        df_val, df_test = train_test_split(df_val_test, test_size=0.5)

        df_train.to_csv(output_path / 'train.csv', index=False)
        df_val.to_csv(output_path / 'val.csv', index=False)
        df_test.to_csv(output_path / 'test.csv', index=False)

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
                                 train_files=[output_path / 'train.csv'],
                                 val_files=[output_path / 'val.csv'],
                                 test_files=[output_path / 'test.csv'],
                                 batch_size=model_config['mini_batch_size'],
                                 num_workers=config.num_workers,
                                 remove_null=config.remove_null)

    return dataset


def main():
    wandb.init(job_type='experiment')

    config = wandb.config

    run_name = config.run_name.format(**dict(config))
    wandb.run.name = run_name

    dataset = get_dataset(config)

    with get_trainer(dataset, run_name, config) as trainer:
        trainer.train(config.num_steps, config.batch_size, warmup_steps=config.get('warmup_steps', 0))

        perform_evaluation(trainer, dataset, config)

    wandb.finish()


if __name__ == '__main__':
    main()
