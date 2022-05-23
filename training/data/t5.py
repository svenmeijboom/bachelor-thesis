from collections import namedtuple
from typing import List

import numpy as np

from data.base import BaseDataset

T5Batch = namedtuple('T5Batch', ['docs', 'inputs', 'targets', 'features',
                                 'input_ids', 'attention_mask', 'decoder_input_ids',
                                 'decoder_attention_mask', 'target_labels'])


class T5Dataset(BaseDataset):
    def __getitem__(self, idx: List[int]) -> T5Batch:
        docs = [self.docs[i] for i in idx]
        inputs = [self.inputs[i] for i in idx]
        targets = [self.targets[i] for i in idx]
        features = [self.features[i] for i in idx]

        formatted_inputs = [
            f'attribute: {feature} context: {text}'
            for text, feature in zip(inputs, features)
        ]

        encoded_inputs = self.tokenizer(formatted_inputs, **self.tokenize_kwargs)
        encoded_targets = self.tokenizer(targets, **self.tokenize_kwargs)

        return T5Batch(
            docs,
            inputs,
            targets,
            features,
            encoded_inputs.input_ids,
            encoded_inputs.attention_mask,
            encoded_targets.input_ids[:, :-1],
            encoded_targets.attention_mask[:, :-1],
            encoded_targets.input_ids[:, 1:],
        )
