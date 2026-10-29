"""Training and inference for the custom Transformer."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import read_split
from .tokenization import SimpleTokenizer
from .transformer import Seq2SeqTransformer, make_batch
from .utils import get_device, set_seed, write_json


def _batches(records: list[dict[str, Any]], source_tok: SimpleTokenizer, target_tok: SimpleTokenizer, batch_size: int):
    pairs = [(source_tok.encode(x["source"]), target_tok.encode(x["target"])) for x in records]
    return DataLoader(pairs, batch_size=batch_size, shuffle=False, collate_fn=lambda x: make_batch(x, source_tok.pad_id, target_tok.pad_id))


def train(config: dict[str, Any], data_dir: str, output_dir: str, smoke: bool = False) -> dict[str, Any]:
    set_seed(config["seed"])
    device = get_device()
    train_records, valid_records = read_split(data_dir, "train"), read_split(data_dir, "validation")
    if smoke:
        train_records, valid_records = train_records[:32], valid_records[:8]
    source_tok = SimpleTokenizer.train([x["source"] for x in train_records], config["transformer"]["vocab_size"], config["transformer"]["min_frequency"])
    target_tok = SimpleTokenizer.train([x["target"] for x in train_records], config["transformer"]["vocab_size"], config["transformer"]["min_frequency"])
    model_cfg = config["transformer"]
    model = Seq2SeqTransformer(len(source_tok.token_to_id), len(target_tok.token_to_id), **{
        key: model_cfg[key] for key in ("d_model", "nhead", "num_encoder_layers", "num_decoder_layers", "dim_feedforward", "dropout")
    }).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_cfg["learning_rate"])
    criterion = nn.CrossEntropyLoss(ignore_index=target_tok.pad_id)
    checkpoint_dir = Path(output_dir) / "models"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    source_tok.save(checkpoint_dir / "source_tokenizer.json")
    target_tok.save(checkpoint_dir / "target_tokenizer.json")
    data_manifest = json.loads((Path(data_dir) / "preprocessing_stats.json").read_text(encoding="utf-8"))
    fingerprint_path = checkpoint_dir / "data_fingerprint.txt"
    existing_checkpoint = checkpoint_dir / "best.pt"
    if existing_checkpoint.exists() and fingerprint_path.exists() and not smoke and fingerprint_path.read_text(encoding="utf-8") == data_manifest["split_fingerprint"]:
        checkpoint = torch.load(existing_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        history_path = Path(output_dir) / "training_history" / "transformer.json"
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else {}
        return {"model": model, "source_tokenizer": source_tok, "target_tokenizer": target_tok,
                "history": history.get("history", []), "training_seconds": history.get("training_seconds", 0.0),
                "parameters": sum(p.numel() for p in model.parameters()),
                "resumed": True}
    history, best_loss, wait = [], float("inf"), 0
    epochs = 1 if smoke else model_cfg["epochs"]
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in _batches(train_records, source_tok, target_tok, model_cfg["batch_size"]):
            batch = make_batch(
                [(x.tolist(), y.tolist()) for x, y in zip(batch.source, torch.cat([batch.target_input[:, :1], batch.target_output], dim=1))],
                source_tok.pad_id, target_tok.pad_id,
            )
            source, target_input, target_output = (x.to(device) for x in (batch.source, batch.target_input, batch.target_output))
            logits = model(source, target_input, source.eq(source_tok.pad_id), target_input.eq(target_tok.pad_id))
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_output.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), model_cfg["max_grad_norm"])
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch in _batches(valid_records, source_tok, target_tok, model_cfg["batch_size"]):
                source, target_input, target_output = (x.to(device) for x in (batch.source, batch.target_input, batch.target_output))
                logits = model(source, target_input, source.eq(source_tok.pad_id), target_input.eq(target_tok.pad_id))
                valid_loss += criterion(logits.reshape(-1, logits.size(-1)), target_output.reshape(-1)).item()
        row = {"epoch": epoch, "train_loss": train_loss, "validation_loss": valid_loss}
        history.append(row)
        if valid_loss < best_loss:
            best_loss, wait = valid_loss, 0
            torch.save({"model": model.state_dict(), "config": model_cfg}, checkpoint_dir / "best.pt")
            fingerprint_path.write_text(data_manifest["split_fingerprint"], encoding="utf-8")
        else:
            wait += 1
            if wait >= model_cfg["patience"]:
                break
    output = {"history": history, "training_seconds": time.perf_counter() - started, "parameters": sum(p.numel() for p in model.parameters())}
    checkpoint = torch.load(existing_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    write_json(Path(output_dir) / "training_history" / "transformer.json", output)
    return {"model": model, "source_tokenizer": source_tok, "target_tokenizer": target_tok, **output}


def generate(config: dict[str, Any], data_dir: str, output_dir: str, smoke: bool = False) -> list[dict[str, Any]]:
    artifacts = train(config, data_dir, output_dir, smoke=smoke)
    model, source_tok, target_tok = artifacts["model"], artifacts["source_tokenizer"], artifacts["target_tokenizer"]
    model.eval()
    records = read_split(data_dir, "test")
    if smoke:
        records = records[:16]
    predictions = []
    started = time.perf_counter()
    for record in records:
        source = torch.tensor([source_tok.encode(record["source"])], device=next(model.parameters()).device)
        generated = model.translate(source, target_tok.bos_id, target_tok.eos_id, config["evaluation"]["max_new_tokens"])[0].tolist()
        predictions.append({**record, "model": "basic_transformer", "prediction": target_tok.decode(generated), "raw_output": generated})
    path = Path(output_dir) / "predictions" / "basic_transformer.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in predictions) + "\n", encoding="utf-8")
    history_path = Path(output_dir) / "training_history" / "transformer.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else {}
    history.update({"inference_seconds_including_tokenization": time.perf_counter() - started,
                    "inference_sentences": len(predictions),
                    "seconds_per_sentence": (time.perf_counter() - started) / len(predictions),
                    "sentences_per_second": len(predictions) / (time.perf_counter() - started),
                    "parameters_updated": sum(p.numel() for p in model.parameters() if p.requires_grad),
                    "adaptation": "from-scratch training"})
    write_json(history_path, history)
    return predictions


def generate_sampling(config: dict[str, Any], data_dir: str, output_dir: str, temperature: float, seed: int, smoke: bool = False) -> list[dict[str, Any]]:
    torch.manual_seed(seed)
    artifacts = train(config, data_dir, output_dir, smoke=smoke)
    model, source_tok, target_tok = artifacts["model"].eval(), artifacts["source_tokenizer"], artifacts["target_tokenizer"]
    records = read_split(data_dir, "test")[: config["evaluation"]["sampling_subset_size"]]
    if smoke:
        records = records[:16]
    predictions = []
    for record in records:
        source = torch.tensor([source_tok.encode(record["source"])], device=next(model.parameters()).device)
        generated = model.translate(source, target_tok.bos_id, target_tok.eos_id,
                                     config["evaluation"]["max_new_tokens"], True, temperature)[0].tolist()
        predictions.append({**record, "model": "basic_transformer", "prediction": target_tok.decode(generated),
                            "raw_output": generated, "temperature": temperature, "seed": seed})
    path = Path(output_dir) / "temperature"
    path.mkdir(parents=True, exist_ok=True)
    (path / f"basic_transformer_t{temperature}_seed{seed}.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in predictions) + "\n", encoding="utf-8")
    return predictions
