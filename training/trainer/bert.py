from typing import Iterable, Tuple

from transformers import BertTokenizer, BertForQuestionAnswering

import torch

from data.bert import BertBatch
from trainer.base import BaseTrainer


class BertTrainer(BaseTrainer):
    architecture = 'bert'

    def __init__(
        self,
        model_version: str,
        *args,
        **kwargs,
    ):
        model = BertForQuestionAnswering.from_pretrained(model_version)
        tokenizer = BertTokenizer.from_pretrained(model_version)

        super().__init__(
            model,
            tokenizer,
            *args,
            **kwargs
        )

    def forward(self, batch: BertBatch):
        return self.model(
            input_ids=batch.input_ids.to(self.device),
            attention_mask=batch.attention_mask.to(self.device),
            token_type_ids=batch.token_type_ids.to(self.device),
            start_positions=batch.start_positions.to(self.device),
            end_positions=batch.end_positions.to(self.device),
        )

    def train_step(self, batch: BertBatch) -> torch.Tensor:
        prediction = self.forward(batch)

        return prediction.loss

    def evaluate_step(self, batch: BertBatch) -> Tuple[float, Iterable[str], Iterable[float]]:
        with torch.no_grad():
            outputs = self.forward(batch)

            start_outputs = outputs.start_logits.max(axis=1)
            end_outputs = outputs.end_logits.max(axis=1)

            start_indices = start_outputs.indices.tolist()
            end_indices = end_outputs.indices.tolist()

            scores = (start_outputs.values * end_outputs.values).tolist()

        predictions = []
        for i, (start, end) in enumerate(zip(start_indices, end_indices)):
            if start == end == 0:
                predictions.append('')
            else:
                predict_answer_tokens = batch.input_ids[i, start:end + 1]
                predictions.append(self.tokenizer.decode(predict_answer_tokens, skip_special_tokens=True))

        return float(outputs.loss), predictions, scores
