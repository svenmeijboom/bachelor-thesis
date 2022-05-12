from lxml import etree
from pathlib import Path
import re
from typing import Iterable, List, Optional

from transformers import PreTrainedTokenizer

from feature_extraction.base import BaseExtractor


class HtmlExtractor(BaseExtractor):
    def __init__(self, input_path: Path, output_path: Path, tokenizer: PreTrainedTokenizer,
                 max_length: int, parent_depth: int = 1, encode_class: bool = True, encode_id: bool = True,
                 encode_tag_subset: Optional[Iterable[str]] = None, split_attributes: bool = False,
                 domains: Optional[dict] = None, num_workers: int = 1):
        super().__init__(input_path, output_path, tokenizer, max_length, domains, num_workers)
        self.parent_depth = parent_depth
        self.encode_class = encode_class
        self.encode_id = encode_id
        self.encode_tag_subset = encode_tag_subset
        self.split_attributes = split_attributes

    def get_parent_prefix(self, elem: etree.Element, depth: int = 0) -> List[str]:
        if depth >= self.parent_depth:
            return []
        elif elem is None:
            return []

        parent_prefix = self.get_parent_prefix(elem.getparent(), depth=depth + 1)

        current_parent = [
            f'parent{depth}tag {elem.tag}',
        ]

        if self.encode_tag_subset is None or elem.tag in self.encode_tag_subset:
            if (value := elem.get('id')) is not None and self.encode_id:
                if self.split_attributes:
                    parts = self.split_attribute_value(value)
                else:
                    parts = [value]
                current_parent.extend(f'parent{depth}id {part}' for part in parts)

            if (value := elem.get('class')) is not None and self.encode_class:
                if self.split_attributes:
                    parts = self.split_attribute_value(value)
                else:
                    parts = value.split()
                current_parent.extend(f'parent{depth}class {part}' for part in parts)

        parent_prefix.extend(current_parent)

        return parent_prefix

    def feature_representation(self, elem: etree.Element) -> str:
        texts = []
        tails = []

        for el in elem.iter():
            text = el.text and el.text.strip()
            tail = el.tail and el.tail.strip()

            if text:
                prefix = ' '.join(self.get_parent_prefix(el))
                texts.append(f'{prefix} {text}')
            if tail:
                prefix = ' '.join(self.get_parent_prefix(el.getparent()))
                tails.insert(0, f'{prefix} {tail}')

        return ' '.join(texts + tails)

    @staticmethod
    def split_attribute_value(value: str) -> List[str]:
        def split_dashes(s: str) -> List[str]:
            return re.split(r'[_\- ]', s)

        def split_camel(s: str) -> List[str]:
            parts = []
            current = []

            for i in range(len(s) - 1):
                current.append(s[i])
                if s[i].islower() and s[i + 1].isupper():
                    parts.append(''.join(current))
                    current = []

            current.append(s[-1])
            parts.append(''.join(current))

            return parts

        parts = {value}

        for transform in (split_dashes, split_camel):
            parts = set(y for x in parts for y in transform(x) if y)

        return list(parts)
