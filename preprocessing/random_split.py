import os
from pathlib import Path
import shutil
import tempfile

from sklearn.model_selection import train_test_split

import wandb

TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1

NAME = 'train-val-test-split'


def normalize_name(name: str):
    """Removes the vertical indicator and the amount of pages in the website"""
    return name.split('(')[0].split('-')[1]


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        wandb.init(
            project='information_extraction',
            name='random-split',
            job_type='preprocess',
        )

        dataset_artifact = wandb.use_artifact('swde-pos:latest')

        dataset_base = Path(dataset_artifact.download(root=os.path.expanduser('~/Data/SWDE-pos/')))
        split_base = Path(tmpdir)

        for vertical in os.listdir(dataset_base):
            if vertical == 'groundtruth' or not os.path.isdir(dataset_base / vertical):
                continue

            all_files = [
                (website, filename)
                for website in os.listdir(dataset_base / vertical)
                for filename in os.listdir(dataset_base / vertical / website)
            ]

            train_val_files, test_files = train_test_split(all_files, test_size=TEST_SIZE)
            train_files, val_files = train_test_split(train_val_files, test_size=VAL_SIZE / (1 - TEST_SIZE))

            split_mapping = {
                'train': train_files,
                'val': val_files,
                'test': test_files,
            }

            for split_name, split_files in split_mapping.items():
                for website, filename in split_files:
                    dest_path = split_base / split_name / vertical / normalize_name(website)

                    os.makedirs(dest_path, exist_ok=True)

                    shutil.copy(
                        dataset_base / vertical / website / filename,
                        dest_path / filename,
                    )

        shutil.copytree(dataset_base / 'groundtruth', split_base / 'groundtruth')

        artifact = wandb.Artifact('random-split', type='train-val-test-split',
                                  description='Fully random train / val / test split of the SWDE dataset',
                                  metadata={'train size': TRAIN_SIZE, 'val size': VAL_SIZE, 'test size': TEST_SIZE})
        artifact.add_dir(str(split_base))

        wandb.log_artifact(artifact)
        wandb.finish()


if __name__ == '__main__':
    main()
