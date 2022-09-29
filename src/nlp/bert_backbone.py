"""BERT shared backbone wrapper.

We expose two outputs:
  - sequence_output: (B, T, H) for token-level tasks (NER)
  - pooled_output:   (B, H)    for sentence-level tasks (sentiment, topic)
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import BertModel


@dataclass
class BertOutputs:
    sequence_output: torch.Tensor   # (B, T, H)
    pooled_output: torch.Tensor     # (B, H)
    attention_mask: torch.Tensor    # (B, T) -- propagated for token tasks


class BertBackbone(nn.Module):
    def __init__(self, pretrained: str = "bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained)
        self.hidden_size = self.bert.config.hidden_size

    def freeze(self, freeze: bool = True) -> None:
        for p in self.bert.parameters():
            p.requires_grad = not freeze

    def forward(self, batch) -> BertOutputs:
        out = self.bert(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
        )
        return BertOutputs(
            sequence_output=out.last_hidden_state,
            pooled_output=out.pooler_output,
            attention_mask=batch["attention_mask"],
        )
