"""NLP demo data.

A toy multi-task dataset for the BERT MTL setup. The same input sentence
carries three labels:
  - sentiment   (binary)
  - topic       (4-class)
  - ner_tags    (per-token, 9 BIO tags)

In a real pipeline you would load AG News for topic, SST-2 for sentiment, and
CoNLL-2003 for NER and zip them up with task masking. We keep things simple
here so the demo runs offline.
"""

from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset


# Toy NER tag scheme.
NER_TAGS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
NER_TAG_TO_ID = {t: i for i, t in enumerate(NER_TAGS)}
NUM_NER = len(NER_TAGS)
PAD_NER_ID = -100   # ignored by CrossEntropyLoss(ignore_index=-100)


_TOY = [
    ("Apple released a new iPhone in California today",
     1, 2, ["B-ORG", "O", "O", "O", "O", "O", "B-LOC", "O"]),
    ("The movie was absolutely terrible and boring",
     0, 1, ["O", "O", "O", "O", "O", "O", "O"]),
    ("Microsoft and Google compete in cloud services",
     1, 2, ["B-ORG", "O", "B-ORG", "O", "O", "O", "O"]),
    ("I really enjoyed the new pizza place downtown",
     1, 0, ["O", "O", "O", "O", "O", "O", "O", "O"]),
    ("Manchester United won the match in London",
     1, 3, ["B-ORG", "I-ORG", "O", "O", "O", "O", "B-LOC"]),
    ("That restaurant has the worst service ever",
     0, 0, ["O", "O", "O", "O", "O", "O", "O"]),
    ("The senate passed a new bill in Washington",
     0, 2, ["O", "O", "O", "O", "O", "O", "O", "B-LOC"]),
    ("Lebron James scored 40 points last night",
     1, 3, ["B-PER", "I-PER", "O", "O", "O", "O", "O"]),
]


class ToyMTLNLPDataset(Dataset):
    def __init__(self, tokenizer, max_seq_len: int = 32, repeat: int = 32):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples = _TOY * repeat

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text, sent, topic, ner_tags = self.examples[idx]
        words = text.split()

        enc = self.tokenizer(
            words,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )

        word_ids = enc.word_ids(0)
        labels: List[int] = []
        for wid in word_ids:
            if wid is None or wid >= len(ner_tags):
                labels.append(PAD_NER_ID)
            else:
                labels.append(NER_TAG_TO_ID[ner_tags[wid]])

        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "token_type_ids": enc.get("token_type_ids", torch.zeros_like(enc["input_ids"]))[0],
            "sentiment": torch.tensor(sent, dtype=torch.long),
            "topic": torch.tensor(topic, dtype=torch.long),
            "ner": torch.tensor(labels, dtype=torch.long),
        }


def collate(batch):
    out = {}
    for k in batch[0]:
        out[k] = torch.stack([b[k] for b in batch])
    inputs = {
        "input_ids": out["input_ids"],
        "attention_mask": out["attention_mask"],
        "token_type_ids": out["token_type_ids"],
    }
    targets = {
        "sentiment": out["sentiment"],
        "topic": out["topic"],
        "ner": out["ner"],
    }
    return {"inputs": inputs, "targets": targets}
