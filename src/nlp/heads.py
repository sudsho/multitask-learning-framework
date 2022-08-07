"""Task heads on top of BertBackbone."""

import torch
import torch.nn as nn


class SentenceClassifierHead(nn.Module):
    """Used for sentiment and topic. Consumes pooled_output."""

    def __init__(self, hidden_size: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, bert_out) -> torch.Tensor:
        x = self.dropout(bert_out.pooled_output)
        return self.fc(x)


class TokenClassifierHead(nn.Module):
    """Used for NER. Consumes sequence_output."""

    def __init__(self, hidden_size: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, bert_out) -> torch.Tensor:
        x = self.dropout(bert_out.sequence_output)
        return self.fc(x)   # (B, T, C)
