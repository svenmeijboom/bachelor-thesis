import os
from pathlib import Path
import shutil
import tempfile

from sklearn.model_selection import train_test_split

import wandb

TRAIN_SIZE = 0.6
VAL_SIZE = 0.2
TEST_SIZE = 0.2

NAME = 'zero-shot-split'


def normalize_name(name: str):
    """Removes the vertical indicator and the amount of pages in the website"""
    return name.split('(')[0].split('-')[1]


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        wandb.init(
            project='information_extraction',
            name=NAME,
            job_type='preprocess',
        )

        dataset_artifact = wandb.use_artifact('swde-pos:latest')

        dataset_base = Path(dataset_artifact.download(root=os.path.expanduser('~/Data/SWDE-pos/')))
        split_base = Path(tmpdir)

        for vertical in os.listdir(dataset_base):
            if vertical == 'groundtruth' or not os.path.isdir(dataset_base / vertical):
                continue

            websites = os.listdir(dataset_base / vertical)

            train_val_dirs, test_dirs = train_test_split(websites, test_size=TEST_SIZE)
            train_dirs, val_dirs = train_test_split(train_val_dirs, test_size=VAL_SIZE / (1 - TEST_SIZE))

            dir_mapping = {
                'train': train_dirs,
                'val': val_dirs,
                'test': test_dirs,
            }

            for split_name, split_dirs in dir_mapping.items():
                os.makedirs(split_base / split_name / vertical, exist_ok=True)

                for split_dir in split_dirs:

                    shutil.copytree(dataset_base / vertical / split_dir,
                                    split_base / split_name / vertical / normalize_name(split_dir))

        shutil.copytree(dataset_base / 'groundtruth', split_base / 'groundtruth')

        artifact = wandb.Artifact(NAME, type='train-val-test-split',
                                  description='Zero shot train / val / test split of the SWDE dataset',
                                  metadata={'train size': TRAIN_SIZE, 'val size': VAL_SIZE, 'test size': TEST_SIZE})
        artifact.add_dir(str(split_base))

        wandb.log_artifact(artifact)
        wandb.finish()


if __name__ == '__main__':
    main()
