from collections import namedtuple
import os
from pathlib import Path
from typing import Dict, List

from sklearn.model_selection import train_test_split

from information_extraction.data import DOMAINS

SplitFile = namedtuple('SplitFile', ['vertical', 'website', 'filename'])


def get_random_split(input_dir: Path, train_size: float) -> Dict[str, List[SplitFile]]:
    split = {'train': [], 'val': [], 'test': []}

    for vertical in DOMAINS:
        all_files = [
            SplitFile(vertical, website, filename)
            for website in os.listdir(input_dir / vertical)
            for filename in os.listdir(input_dir / vertical / website)
        ]

        train_files, val_test_files = train_test_split(all_files, train_size=train_size,
                                                       stratify=[file.website for file in all_files])
        val_files, test_files = train_test_split(val_test_files, test_size=0.5,
                                                 stratify=[file.website for file in val_test_files])

        split['train'].extend(train_files)
        split['val'].extend(val_files)
        split['test'].extend(test_files)

    return split


def get_webke_split(webke_split_mapping: dict, train_size: float) -> Dict[str, List[SplitFile]]:
    split = {'train': [], 'val': [], 'test': []}

    for vertical in ['movie', 'nbaplayer', 'university']:
        train_val_files = [
            SplitFile(vertical, website, f'{doc_id}.htm')
            for website, doc_id in webke_split_mapping['train'][vertical].values()
        ]
        test_files = [
            SplitFile(vertical, website, f'{doc_id}.htm')
            for website, doc_id in webke_split_mapping['test'][vertical].values()
        ]

        train_files, val_files = train_test_split(train_val_files, train_size=2 * train_size / (train_size + 1),
                                                  stratify=[file.website for file in train_val_files])

        split['train'].extend(train_files)
        split['val'].extend(val_files)
        split['test'].extend(test_files)

    return split


def get_zero_shot_split(input_dir: Path, train_size: float) -> Dict[str, List[SplitFile]]:
    split = {'train': [], 'val': [], 'test': []}

    for vertical in DOMAINS:
        websites = os.listdir(input_dir / vertical)

        train_websites, val_test_websites = train_test_split(websites, train_size=train_size)
        val_websites, test_websites = train_test_split(val_test_websites, test_size=0.5)

        split['train'].extend(
            SplitFile(vertical, website, filename)
            for website in train_websites
            for filename in os.listdir(input_dir / vertical / website)
        )

        split['val'].extend(
            SplitFile(vertical, website, filename)
            for website in val_websites
            for filename in os.listdir(input_dir / vertical / website)
        )

        split['test'].extend(
            SplitFile(vertical, website, filename)
            for website in test_websites
            for filename in os.listdir(input_dir / vertical / website)
        )

    return split

