from typing import Iterable, Tuple

from transformers import T5Tokenizer, T5ForConditionalGeneration

import torch
from torch.nn import CrossEntropyLoss

from data.t5 import T5Batch
from trainer.base import BaseTrainer


class T5Trainer(BaseTrainer):
    architecture = 't5'

    def __init__(
        self,
        model_version: str,
        *args,
        **kwargs,
    ):
        model = T5ForConditionalGeneration.from_pretrained(model_version)
        tokenizer = T5Tokenizer.from_pretrained(model_version)

        self.loss_fn = CrossEntropyLoss()

        super().__init__(
            model,
            tokenizer,
            *args,
            **kwargs
        )

    def forward(self, batch: T5Batch):
        return self.model(
            input_ids=batch.input_ids.to(self.device),
            attention_mask=batch.attention_mask.to(self.device),
            decoder_input_ids=batch.decoder_input_ids.to(self.device),
            decoder_attention_mask=batch.decoder_attention_mask.to(self.device),
        )

    def train_step(self, batch: T5Batch) -> torch.Tensor:
        prediction = self.forward(batch)

        logits = prediction.logits
        target_labels = batch.target_labels.to(self.device)
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), target_labels.flatten())

        return loss

    def evaluate_step(self, batch: T5Batch) -> Tuple[float, Iterable[str]]:
        with torch.no_grad():
            prediction = self.forward(batch)

            logits = prediction.logits

            target_labels = batch.target_labels.to(self.device)
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), target_labels.flatten())

            generated_outputs = self.model.generate(batch.input_ids.to(self.device))
            decoded_outputs = self.tokenizer.batch_decode(generated_outputs)

        predictions = [s.replace('</s>', '').replace('<pad>', '').strip() for s in decoded_outputs]

        return float(loss), predictions
