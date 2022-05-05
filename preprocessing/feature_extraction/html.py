from lxml import etree
from pathlib import Path
from typing import List, Optional

from transformers import PreTrainedTokenizer

from feature_extraction.base import BaseExtractor


class HtmlExtractor(BaseExtractor):
    output_format = '{split}/{vertical}/{website}-{max_length}.csv'

    def __init__(self, input_path: Path, output_path: Path, tokenizer: PreTrainedTokenizer,
                 max_length: int, parent_depth: int = 1, domains: Optional[dict] = None,
                 num_workers: int = 1):
        super().__init__(input_path, output_path, tokenizer, max_length, domains, num_workers)
        self.parent_depth = parent_depth

    def get_parents(self, elem: etree.Element, depth: int = 0) -> List[str]:
        if depth >= self.parent_depth:
            return []
        elif elem is None:
            return []

        parents = self.get_parents(elem.getparent(), depth=depth + 1)
        parents.append(elem.tag)

        return parents

    def feature_representation(self, elem: etree.Element) -> str:
        texts = []
        tails = []

        for el in elem.iter():
            parents = '/'.join(self.get_parents(el))

            prefix = f'node={parents}'

            if value := el.get('id'):
                prefix += f' id={value}'
            if value := el.get('class'):
                prefix += ' ' + ' '.join(f'class={name}' for name in value.split())

            text = el.text and el.text.strip()
            tail = el.tail and el.tail.strip()

            if text:
                texts.append(f'{prefix} {text}')
            if tail:
                tails.insert(0, f'{prefix} {tail}')

        return ' '.join(texts + tails)
