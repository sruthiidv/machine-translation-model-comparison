"""Small word-level tokenizer for the from-scratch Transformer."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


class SimpleTokenizer:
    def __init__(self, token_to_id: dict[str, int]):
        self.token_to_id = token_to_id
        self.id_to_token = {value: key for key, value in token_to_id.items()}
        self.pad_id = token_to_id["<pad>"]
        self.unk_id = token_to_id["<unk>"]
        self.bos_id = token_to_id["<bos>"]
        self.eos_id = token_to_id["<eos>"]

    @classmethod
    def train(cls, texts: list[str], vocab_size: int = 16000, min_frequency: int = 2) -> "SimpleTokenizer":
        counts = Counter(token for text in texts for token in text.split())
        vocabulary = SPECIAL_TOKENS + [
            token for token, count in counts.most_common() if count >= min_frequency and token not in SPECIAL_TOKENS
        ][: max(0, vocab_size - len(SPECIAL_TOKENS))]
        return cls({token: index for index, token in enumerate(vocabulary)})

    def encode(self, text: str, max_length: int | None = None) -> list[int]:
        values = [self.bos_id] + [self.token_to_id.get(token, self.unk_id) for token in text.split()] + [self.eos_id]
        return values[:max_length] if max_length else values

    def decode(self, ids: list[int]) -> str:
        ignored = {self.pad_id, self.bos_id, self.eos_id}
        return " ".join(self.id_to_token.get(index, "<unk>") for index in ids if index not in ignored)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.token_to_id, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SimpleTokenizer":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))
