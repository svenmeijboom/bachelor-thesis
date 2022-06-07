import json
import os
from pathlib import Path
import shutil
import tempfile

from sklearn.model_selection import train_test_split

import wandb

TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1


def normalize_name(name: str):
    """Removes the vertical indicator and the amount of pages in the website"""
    return name.split('(')[0].split('-')[1]


def main():
    with open('webke_split_mapping.json') as _file:
        webke_split_mapping = json.load(_file)

    with tempfile.TemporaryDirectory() as tmpdir:
        wandb.init(
            project='information_extraction',
            name='webke-split',
            job_type='preprocess',
        )

        dataset_artifact = wandb.use_artifact('swde:latest')

        dataset_base = Path(dataset_artifact.download(root=os.path.expanduser('~/Data/SWDE/')))
        split_base = Path(tmpdir)

        for vertical in ['movie', 'nbaplayer', 'university']:
            train_val_files = [
                (website, f'{doc_id}.htm')
                for website, doc_id in webke_split_mapping['train'][vertical].values()
            ]
            test_files = [
                (website, f'{doc_id}.htm')
                for website, doc_id in webke_split_mapping['test'][vertical].values()
            ]

            train_files, val_files = train_test_split(train_val_files, test_size=VAL_SIZE / (TRAIN_SIZE + VAL_SIZE),
                                                      stratify=[file[0] for file in train_val_files])

            split_mapping = {
                'train': train_files,
                'val': val_files,
                'test': test_files,
            }

            for split_name, split_files in split_mapping.items():
                for website, filename in split_files:
                    dest_path = split_base / split_name / vertical / normalize_name(website)

                    dest_path.mkdir(exist_ok=True, parents=True)
                    os.makedirs(dest_path, exist_ok=True)

                    shutil.copy(
                        dataset_base / vertical / website / filename,
                        dest_path / filename,
                    )

            dest_path = split_base / 'groundtruth'
            dest_path.mkdir(exist_ok=True)

            shutil.copytree(dataset_base / 'groundtruth' / vertical, dest_path / vertical)

        artifact = wandb.Artifact('webke-split', type='train-val-test-split',
                                  description='Train / val / test split of the SWDE dataset matching the preprocessed WebKE data',
                                  metadata={'train size': TRAIN_SIZE, 'val size': VAL_SIZE, 'test size': TEST_SIZE})
        artifact.add_dir(str(split_base))

        wandb.log_artifact(artifact)
        wandb.finish()


if __name__ == '__main__':
    main()
