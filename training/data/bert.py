from collections import namedtuple
from typing import List, Optional

import numpy as np

from transformers import BertTokenizer

from data.base import BaseDataset

BertBatch = namedtuple('BertBatch', ['inputs', 'targets', 'features',
                                     'input_ids', 'attention_mask', 'token_type_ids',
                                     'start_positions', 'end_positions'])


class BertDataset(BaseDataset):
    input_ids: np.array
    attention_mask: np.array
    token_type_ids: np.array
    start_positions: np.array
    end_positions: np.array

    def __init__(self, files: List[str], bert_version: str, remove_null: bool = False, max_length: Optional[int] = None):
        self.tokenizer = BertTokenizer.from_pretrained(bert_version)
        super().__init__(files, remove_null, max_length)

    def prepare_inputs(self):
        tokenized_inputs = self.tokenizer(self.features, self.inputs,
                                          **self.tokenize_kwargs)

        self.input_ids = tokenized_inputs.input_ids
        self.attention_mask = tokenized_inputs.attention_mask
        self.token_type_ids = tokenized_inputs.token_type_ids

        self.start_positions, self.end_positions = self.get_answer_indices()

    def get_answer_indices(self):
        start_indices = []
        end_indices = []
        for tokenized_input, target in zip(self.input_ids, self.targets):
            if not target:
                start_indices.append(0)
                end_indices.append(0)
                continue

            tokenized_target = self.tokenizer(target, return_tensors='pt', add_special_tokens=False).input_ids

            tokenized_target = tokenized_target.flatten()
            tokenized_input = tokenized_input.flatten()

            start, end = 0, 0
            for i in range(0, len(tokenized_input) - len(tokenized_target) + 1):
                if np.array_equal(tokenized_input[i:i + len(tokenized_target)], tokenized_target):
                    start, end = i, i + len(tokenized_target) - 1
                    break

            start_indices.append(start)
            end_indices.append(end)

        return np.array(start_indices, dtype=np.int64), np.array(end_indices, dtype=np.int64)

    def __getitem__(self, item: int) -> BertBatch:
        return BertBatch(
            self.inputs[item],
            self.targets[item],
            self.features[item],
            self.input_ids[item],
            self.attention_mask[item],
            self.token_type_ids[item],
            self.start_positions[item],
            self.end_positions[item],
        )
