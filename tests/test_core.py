import torch
import pytest

from src.evaluate import score
from src.transformer import Seq2SeqTransformer, make_batch


def test_target_shift_and_padding():
    batch = make_batch([([2, 4, 3], [2, 7, 8, 3]), ([2, 5, 3], [2, 9, 3])], 0, 0)
    assert batch.target_input.tolist()[0] == [2, 7, 8]
    assert batch.target_output.tolist()[0] == [7, 8, 3]
    assert batch.target_output.tolist()[1] == [9, 3, 0]


def test_causal_mask_blocks_future_positions():
    mask = torch.nn.Transformer.generate_square_subsequent_mask(4)
    assert torch.isneginf(mask[0, 1])
    assert mask[2, 1] == 0


def test_transformer_forward_padding_shapes():
    model = Seq2SeqTransformer(20, 20, d_model=16, nhead=4, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=32)
    source = torch.tensor([[2, 4, 3], [2, 0, 0]])
    target = torch.tensor([[2, 5], [2, 0]])
    logits = model(source, target, source.eq(0), target.eq(0))
    assert logits.shape == (2, 2, 20)


def test_metric_alignment():
    result = score([{"target": "bonjour tout le monde", "prediction": "bonjour tout le monde"}])
    assert result["n"] == 1
    assert result["bleu"] == pytest.approx(100)
