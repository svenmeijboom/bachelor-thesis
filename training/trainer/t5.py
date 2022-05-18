from typing import Iterable, Tuple

from transformers import T5Tokenizer, T5ForConditionalGeneration

import torch
from torch.nn import functional as F
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

    def evaluate_step(self, batch: T5Batch) -> Tuple[float, Iterable[str], Iterable[float]]:
        with torch.no_grad():
            prediction = self.forward(batch)

            logits = prediction.logits

            target_labels = batch.target_labels.to(self.device)
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), target_labels.flatten())

            outputs = self.model.generate(batch.input_ids.to(self.device),
                                          return_dict_in_generate=True, output_scores=True)

            # Compute the probability of each token in the decoded sequences
            stacked_scores = F.softmax(torch.stack(outputs.scores).permute(1, 0, 2), dim=2).double()
            token_scores = torch.take_along_dim(stacked_scores, outputs.sequences[:, 1:, None], dim=2).squeeze()

            # Average the probabilities over each sequence to obtain score per sequence
            token_mask = outputs.sequences[:, 1:].gt(1)
            sums = torch.where(token_mask, token_scores, 0.).sum(dim=1)
            lengths = token_mask.sum(dim=1)

            scores = (sums / lengths).tolist()

        predictions = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

        return float(loss), predictions, scores
