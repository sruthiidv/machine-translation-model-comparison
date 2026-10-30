"""Small, explicit FP32 mT5 fine-tuning and resumable generation."""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .data import read_split
from .utils import available_gib, get_device, set_seed, write_json


def _require_transformers():
    try:
        from transformers import AutoTokenizer, DataCollatorForSeq2Seq, T5ForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt to fine-tune mT5.") from exc
    return AutoTokenizer, DataCollatorForSeq2Seq, T5ForConditionalGeneration


def _finite_model(model: torch.nn.Module) -> bool:
    return all(torch.isfinite(parameter).all().item() for parameter in model.parameters())


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _compatible(checkpoint: Path, data_fingerprint: str, cfg: dict[str, Any], history_path: Path) -> bool:
    fingerprint = checkpoint / "data_fingerprint.txt"
    if not (checkpoint / "config.json").exists() or not fingerprint.exists() or not history_path.exists():
        return False
    history = json.loads(history_path.read_text(encoding="utf-8"))
    return (history.get("status") == "completed" and history.get("finite", False)
            and fingerprint.read_text(encoding="utf-8").strip() == data_fingerprint
            and history.get("model_name") == cfg["model_name"])


def _tokenize(tokenizer, records: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for row in records:
        encoded = tokenizer(f"translate English to French: {row['source']}",
                            max_length=cfg["max_source_tokens"], truncation=True)
        labels = tokenizer(text_target=row["target"], max_length=cfg["max_target_tokens"], truncation=True)
        output.append({"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"],
                       "labels": labels["input_ids"]})
    return output


def train(config: dict[str, Any], data_dir: str, output_dir: str, smoke: bool = False) -> dict[str, Any]:
    tokenizer_cls, collator_cls, model_cls = _require_transformers()
    set_seed(config["seed"])
    device = get_device()
    cfg = config["mt5"]
    output = Path(output_dir)
    checkpoint = output / "models" / "mt5"
    history_path = output / "training_history" / "mt5.json"
    manifest = json.loads((Path(data_dir) / "preprocessing_stats.json").read_text(encoding="utf-8"))
    if not smoke and cfg.get("resume_valid_checkpoint", True) and _compatible(checkpoint, manifest["split_fingerprint"], cfg, history_path):
        return json.loads(history_path.read_text(encoding="utf-8")) | {"resumed": True}
    if history_path.exists() and not smoke:
        failed_dir = output / "failed_runs"
        failed_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(history_path, failed_dir / f"mt5_previous_{int(time.time())}.json")
    if not smoke and available_gib(output) < float(cfg.get("min_free_gib", 4.0)):
        raise RuntimeError(f"Only {available_gib(output):.2f} GiB free; mT5 requires at least {cfg['min_free_gib']:.1f} GiB before training.")
    token = os.environ.get("HF_TOKEN") or None
    tokenizer = tokenizer_cls.from_pretrained(cfg["model_name"], token=token)
    model = model_cls.from_pretrained(cfg["model_name"], token=token).float().to(device)
    if not _finite_model(model):
        raise RuntimeError("Pretrained mT5 weights are non-finite; refusing to train.")
    train_rows, valid_rows = read_split(data_dir, "train"), read_split(data_dir, "validation")
    if smoke:
        train_rows, valid_rows = train_rows[:8], valid_rows[:4]
    collator = collator_cls(tokenizer=tokenizer, model=model)
    train_loader = DataLoader(_tokenize(tokenizer, train_rows, cfg), batch_size=cfg["batch_size"], shuffle=True,
                              collate_fn=collator)
    valid_loader = DataLoader(_tokenize(tokenizer, valid_rows, cfg), batch_size=cfg["batch_size"], shuffle=False,
                              collate_fn=collator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    max_steps = len(train_loader) * (1 if smoke else cfg["epochs"])
    max_steps = min(max_steps, int(cfg.get("max_train_steps", max_steps)))
    accumulation = max(1, int(cfg["gradient_accumulation_steps"]))
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    for batch in train_loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        loss = model(**batch).loss
        if not torch.isfinite(loss):
            write_json(history_path, {"status": "failed", "failure": "non-finite loss", "finite": False,
                                      "model_name": cfg["model_name"], "preserved": True})
            raise RuntimeError("mT5 produced a non-finite loss; run stopped before saving a checkpoint.")
        (loss / accumulation).backward()
        if (step + 1) % accumulation == 0 or step + 1 == max_steps:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                write_json(history_path, {"status": "failed", "failure": "non-finite gradient", "finite": False,
                                          "model_name": cfg["model_name"], "preserved": True})
                raise RuntimeError("mT5 produced a non-finite gradient; run stopped before saving a checkpoint.")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        step += 1
        if step >= max_steps:
            break
    model.eval()
    losses = []
    with torch.no_grad():
        for index, batch in enumerate(valid_loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            value = model(**batch).loss
            if not torch.isfinite(value):
                raise RuntimeError("mT5 validation loss is non-finite; refusing to export this model.")
            losses.append(float(value))
            if index + 1 >= int(cfg.get("validation_steps", 50)):
                break
    validation_loss = sum(losses) / max(1, len(losses))
    if not torch.isfinite(torch.tensor(validation_loss)):
        raise RuntimeError("mT5 validation loss is non-finite.")
    if not smoke and available_gib(output) < float(cfg.get("min_free_gib", 4.0)):
        raise RuntimeError(f"Only {available_gib(output):.2f} GiB free before saving mT5; refusing to risk a partial export.")
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)
    (checkpoint / "data_fingerprint.txt").write_text(manifest["split_fingerprint"], encoding="utf-8")
    result = {"status": "completed", "finite": True, "model_name": cfg["model_name"],
              "training_seconds": time.perf_counter() - started, "validation_loss": validation_loss,
              "steps": step, "parameters": sum(p.numel() for p in model.parameters()),
              "parameters_updated": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "device": str(device), "adaptation": "full fine-tuning", "checkpoint_type": "model-only",
              "optimizer_state_saved": False,
              "note": "Model-only checkpoints cannot provide exact optimizer-state resumption."}
    write_json(history_path, result)
    del optimizer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _generate(config, data_dir, output_dir, model, tokenizer, model_key="mt5", sampling=False,
              temperature=1.0, seed=42, smoke=False):
    device = get_device()
    path = (Path(output_dir) / "temperature" / f"{model_key}_t{temperature}_seed{seed}.jsonl"
            if sampling else Path(output_dir) / "predictions" / f"{model_key}.jsonl")
    existing = {row["example_id"]: row for row in _records(path)} if path.exists() else {}
    records = read_split(data_dir, "test")
    if smoke:
        records = records[:16]
    elif sampling:
        records = records[:config["evaluation"]["sampling_subset_size"]]
    results = []
    started = time.perf_counter()
    for record in records:
        if record["example_id"] in existing:
            results.append(existing[record["example_id"]])
            continue
        inputs = tokenizer(f"translate English to French: {record['source']}", return_tensors="pt",
                           truncation=True).to(device)
        if sampling:
            torch.manual_seed(seed)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=config["evaluation"]["max_new_tokens"],
                                       do_sample=sampling, temperature=temperature, top_p=config["evaluation"]["sampling_top_p"])
        results.append({**record, "model": model_key, "prediction": tokenizer.decode(generated[0], skip_special_tokens=True),
                        "raw_output": generated[0].tolist(), "temperature": temperature if sampling else None})
        path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in results) + "\n", encoding="utf-8")
    history_path = Path(output_dir) / "training_history" / f"{model_key}.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else {}
    elapsed = time.perf_counter() - started
    if not sampling:
        history.update({"inference_seconds_including_tokenization": elapsed, "inference_sentences": len(results),
                        "seconds_per_sentence": elapsed / max(1, len(results)),
                        "sentences_per_second": len(results) / max(elapsed, 1e-9)})
        write_json(history_path, history)
    return results


def generate(config, data_dir, output_dir, smoke=False):
    tokenizer_cls, _, model_cls = _require_transformers()
    checkpoint = Path(output_dir) / "models" / "mt5"
    history = train(config, data_dir, output_dir, smoke)
    device = get_device()
    tokenizer = tokenizer_cls.from_pretrained(checkpoint)
    model = model_cls.from_pretrained(checkpoint).float().to(device).eval()
    result = _generate(config, data_dir, output_dir, model, tokenizer, smoke=smoke)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def generate_sampling(config, data_dir, output_dir, temperature, seed, smoke=False):
    tokenizer_cls, _, model_cls = _require_transformers()
    train(config, data_dir, output_dir, smoke)
    device = get_device()
    checkpoint = Path(output_dir) / "models" / "mt5"
    tokenizer = tokenizer_cls.from_pretrained(checkpoint)
    model = model_cls.from_pretrained(checkpoint).float().to(device).eval()
    result = _generate(config, data_dir, output_dir, model, tokenizer, sampling=True,
                       temperature=temperature, seed=seed, smoke=smoke)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
