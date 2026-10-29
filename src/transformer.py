"""From-scratch encoder-decoder Transformer and batching utilities."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = 512):
        super().__init__()
        position = torch.arange(max_length).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.encoding[:, : values.size(1)]


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self, source_vocab_size: int, target_vocab_size: int, d_model: int = 256,
        nhead: int = 4, num_encoder_layers: int = 3, num_decoder_layers: int = 3,
        dim_feedforward: int = 1024, dropout: float = 0.1, max_length: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.source_embedding = nn.Embedding(source_vocab_size, d_model)
        self.target_embedding = nn.Embedding(target_vocab_size, d_model)
        self.position = PositionalEncoding(d_model, max_length)
        self.transformer = nn.Transformer(
            d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout,
            batch_first=True,
        )
        self.output = nn.Linear(d_model, target_vocab_size)

    def forward(
        self, source: torch.Tensor, target_input: torch.Tensor,
        source_padding_mask: torch.Tensor | None = None,
        target_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        source_emb = self.position(self.source_embedding(source) * math.sqrt(self.d_model))
        target_emb = self.position(self.target_embedding(target_input) * math.sqrt(self.d_model))
        mask = nn.Transformer.generate_square_subsequent_mask(target_input.size(1), device=source.device)
        values = self.transformer(
            source_emb, target_emb, tgt_mask=mask,
            src_key_padding_mask=source_padding_mask,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=source_padding_mask,
        )
        return self.output(values)

    @torch.no_grad()
    def translate(self, source: torch.Tensor, bos_id: int, eos_id: int, max_length: int = 96,
                 do_sample: bool = False, temperature: float = 1.0) -> torch.Tensor:
        result = torch.full((source.size(0), 1), bos_id, dtype=torch.long, device=source.device)
        for _ in range(max_length - 1):
            logits = self(source, result)
            next_logits = logits[:, -1] / max(temperature, 1e-5)
            next_token = (torch.multinomial(torch.softmax(next_logits, dim=-1), 1)
                          if do_sample else next_logits.argmax(dim=-1, keepdim=True))
            result = torch.cat([result, next_token], dim=1)
            if bool((next_token == eos_id).all()):
                break
        return result


@dataclass
class Batch:
    source: torch.Tensor
    target_input: torch.Tensor
    target_output: torch.Tensor


def pad_sequences(sequences: list[list[int]], pad_id: int) -> torch.Tensor:
    width = max(len(sequence) for sequence in sequences)
    return torch.tensor([sequence + [pad_id] * (width - len(sequence)) for sequence in sequences], dtype=torch.long)


def make_batch(pairs: list[tuple[list[int], list[int]]], source_pad: int, target_pad: int) -> Batch:
    source = pad_sequences([pair[0] for pair in pairs], source_pad)
    target = pad_sequences([pair[1] for pair in pairs], target_pad)
    return Batch(source, target[:, :-1], target[:, 1:])
