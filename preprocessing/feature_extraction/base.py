from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import StringIO
import os
from pathlib import Path
import re
from typing import List, Optional

from lxml import etree
import lxml.html
from lxml.html.clean import Cleaner

import pandas as pd
from tqdm import tqdm

from transformers import PreTrainedTokenizer
from transformers.data.metrics.squad_metrics import normalize_answer


DOMAINS = {
    'auto': ['engine', 'fuel_economy', 'model', 'price'],
    'book': ['author', 'isbn_13', 'publication_date', 'publisher', 'title'],
    'camera': ['manufacturer', 'model', 'price'],
    'job': ['company', 'date_posted', 'location', 'title'],
    'movie': ['director', 'genre', 'mpaa_rating', 'title'],
    'nbaplayer': ['height', 'name', 'team', 'weight'],
    'restaurant': ['address', 'cuisine', 'name', 'phone'],
    'university': ['name', 'phone', 'website', 'type'],
}


class BaseExtractor(ABC):
    output_format = '{split}/{vertical}/{website}/features-{max_length}.csv'

    def __init__(self, input_path: Path, output_path: Path, tokenizer: PreTrainedTokenizer,
                 max_length: int, domains: Optional[dict] = None, num_workers: int = 1):
        self.input_path = input_path
        self.output_path = output_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.domains = domains or DOMAINS
        self.num_workers = num_workers if num_workers > 0 else None

        self.num_too_large = 0

    @abstractmethod
    def feature_representation(self, elem: etree.Element) -> str:
        raise NotImplementedError

    @staticmethod
    def text_representation(elem: etree.Element) -> str:
        return re.sub(r'\s+', ' ', ' '.join(elem.itertext()))

    def process_dataset(self):
        for split in ['train', 'val', 'test']:
            for vertical in os.listdir(self.input_path / split):
                for website in os.listdir(self.input_path / split / vertical):
                    self.process_directory(split, vertical, website)

    def process_directory(self, split: str, vertical: str, website: str) -> Path:
        ground_truths = self.get_ground_truths(vertical, website)

        base_dir = self.input_path / split / vertical / website
        files = [
            filename
            for filename in os.listdir(base_dir)
            if filename.endswith('.htm')
        ]

        self.num_too_large = 0

        extracted_features = []
        with tqdm(desc=f'{split}/{vertical}-{website}', total=len(files)) as pbar:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                future_to_id = {}

                for filename in files:
                    file_base = filename[:-4]  # Stripping off the .htm
                    future = executor.submit(
                        self.get_features_from_file,
                        base_dir / filename,
                        ground_truths[file_base],
                    )
                    future.add_done_callback(lambda _: pbar.update())
                    future_to_id[future] = file_base

                for future in as_completed(future_to_id):
                    page_id = future_to_id[future]
                    try:
                        features = future.result()

                        for feature in features:
                            feature['id'] = page_id

                        extracted_features.extend(features)

                    except Exception as e:
                        print(f'{vertical}/{website}/{page_id}.htm raised exception: {e}')

        df_features = pd.DataFrame(extracted_features)

        destination = self.output_path / self.output_format.format(
            split=split,
            vertical=vertical,
            website=website,
            max_length=self.max_length,
        )

        os.makedirs(os.path.dirname(destination), exist_ok=True)

        df_features.to_csv(destination, index=False)

        if self.num_too_large > 0:
            num_total = self.num_too_large + len(df_features)
            print(f'Warning: {vertical}-{website} has {self.num_too_large}/{num_total} '
                  f'({self.num_too_large / num_total * 100:.2f}%) unsplittable texts that are '
                  f'longer than {self.max_length} tokens')

        return destination

    def get_features_from_file(self, filename: str, ground_truth: dict) -> List[dict]:
        with open(filename) as _file:
            html = _file.read()

        cleaned_html = self.clean_html(html)

        tree = lxml.html.parse(StringIO(cleaned_html))

        cleaner = Cleaner(style=True)
        cleaned_tree = cleaner.clean_html(tree.find('body'))

        return self.extract_features(cleaned_tree, ground_truth)

    def extract_features(self, elem: etree.Element, ground_truth: dict) -> List[dict]:
        representation = self.feature_representation(elem)

        if not representation:
            return []

        if len(self.tokenizer.tokenize(representation)) >= self.max_length - 1:
            if not len(elem):
                # This element is too large to fit in the document, but cannot
                # be split into sub-elements
                self.num_too_large += 1
                return []

            # We split the element that is too large into its children
            features = []
            for child in elem:
                features.extend(self.extract_features(child, ground_truth))

            return features

        result = {
            'text': representation,
        }

        normalized_text = normalize_answer(self.text_representation(elem))

        for key, value in ground_truth.items():
            # TODO: look at how webformer did this
            if normalize_answer(value) in normalized_text:
                result[key] = value
            else:
                result[key] = None

        return [result]

    def read_ground_truth_file(self, vertical: str, website: str, attribute: str):
        filename = f'{vertical}-{website}-{attribute}.txt'
        with open(self.input_path / 'groundtruth' / vertical / filename) as _file:
            entries = _file.read().splitlines()[2:]

        data = {}

        for entry in entries:
            parts = entry.split('\t')
            data[parts[0]] = parts[2:]

        return data

    def get_ground_truths(self, vertical: str, website: str) -> dict:
        ground_truths = defaultdict(dict)

        for attribute in self.domains[vertical]:
            sub_data = self.read_ground_truth_file(vertical, website, attribute)

            for key, values in sub_data.items():
                # TODO: ensure multiple values are allowed
                ground_truths[key][attribute] = values[0]

        return ground_truths

    @staticmethod
    def clean_html(html: str) -> str:
        # https://stackoverflow.com/a/25920392
        html = re.sub(u'[^\u0020-\uD7FF\u0009\u000A\u000D\uE000-\uFFFD\U00010000-\U0010FFFF]+', '', html)

        return html
