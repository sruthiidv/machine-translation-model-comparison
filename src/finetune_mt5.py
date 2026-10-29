"""Memory-conscious mT5 fine-tuning with validation checkpoint selection."""
from __future__ import annotations

import json
import inspect
import os
import time
from pathlib import Path
from typing import Any

import torch

from .data import read_split
from .utils import get_device, set_seed, write_json


def _require_transformers():
    try:
        from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, T5ForConditionalGeneration, T5Tokenizer
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt to fine-tune mT5.") from exc
    return DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, T5ForConditionalGeneration, T5Tokenizer


def train(config: dict[str, Any], data_dir: str, output_dir: str, smoke: bool = False) -> dict[str, Any]:
    if not os.environ.get("HF_TOKEN"):
        # Public mT5 can work without a token, but the explicit check makes auth behavior clear.
        token = None
    else:
        token = os.environ["HF_TOKEN"]
    collator_cls, trainer_cls, args_cls, model_cls, tokenizer_cls = _require_transformers()
    set_seed(config["seed"])
    device = get_device()
    cfg = config["mt5"]
    model_name = cfg["model_name"]
    tokenizer = tokenizer_cls.from_pretrained(model_name, token=token)
    model = model_cls.from_pretrained(model_name, token=token)
    train_records, valid_records = read_split(data_dir, "train"), read_split(data_dir, "validation")
    if smoke:
        train_records, valid_records = train_records[:8], valid_records[:4]

    def tokenize(batch):
        source = [f"translate English to French: {x}" for x in batch["source"]]
        encoded = tokenizer(source, max_length=cfg["max_source_tokens"], truncation=True)
        labels = tokenizer(text_target=batch["target"], max_length=cfg["max_target_tokens"], truncation=True)
        encoded["labels"] = labels["input_ids"]
        return encoded

    from datasets import Dataset
    train_ds = Dataset.from_list(train_records).map(tokenize, batched=True, remove_columns=["example_id", "source", "target"])
    valid_ds = Dataset.from_list(valid_records).map(tokenize, batched=True, remove_columns=["example_id", "source", "target"])
    checkpoint_dir = Path(output_dir) / "models" / "mt5"
    data_manifest = json.loads((Path(data_dir) / "preprocessing_stats.json").read_text(encoding="utf-8"))
    fingerprint_path = checkpoint_dir / "data_fingerprint.txt"
    if checkpoint_dir.exists() and (checkpoint_dir / "config.json").exists() and fingerprint_path.exists() and not smoke and fingerprint_path.read_text(encoding="utf-8") == data_manifest["split_fingerprint"]:
        history_path = Path(output_dir) / "training_history" / "mt5.json"
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else {}
        return {"model": model_name, "training_seconds": history.get("training_seconds", 0.0), "parameters": sum(p.numel() for p in model.parameters()),
                "parameters_updated": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "device": str(device), "adaptation": "full fine-tuning", "resumed": True}
    trainer_args = dict(
        output_dir=str(checkpoint_dir), save_strategy="epoch",
        learning_rate=cfg["learning_rate"], per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"], gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=1 if smoke else cfg["epochs"], predict_with_generate=True,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        fp16=device.type == "cuda" and torch.cuda.get_device_capability()[0] < 9,
        bf16=device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 9,
        report_to="none", save_total_limit=2,
    )
    args_parameters = inspect.signature(args_cls).parameters
    trainer_args["evaluation_strategy" if "evaluation_strategy" in args_parameters else "eval_strategy"] = "epoch"
    args = args_cls(**trainer_args)
    started = time.perf_counter()
    collator = collator_cls(tokenizer=tokenizer, model=model)
    trainer_kwargs = {"model": model, "args": args, "train_dataset": train_ds, "eval_dataset": valid_ds,
                      "data_collator": collator}
    trainer_kwargs["processing_class" if "processing_class" in inspect.signature(trainer_cls).parameters else "tokenizer"] = tokenizer
    trainer = trainer_cls(**trainer_kwargs)
    trainer.train()
    trainer.save_model(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    result = {
        "model": model_name, "training_seconds": time.perf_counter() - started,
        "parameters": sum(p.numel() for p in model.parameters()),
        "parameters_updated": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "device": str(device), "adaptation": "full fine-tuning",
    }
    write_json(Path(output_dir) / "training_history" / "mt5.json", result)
    fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint_path.write_text(data_manifest["split_fingerprint"], encoding="utf-8")
    return result


def generate(config: dict[str, Any], data_dir: str, output_dir: str, smoke: bool = False) -> list[dict[str, Any]]:
    _, _, _, model_cls, tokenizer_cls = _require_transformers()
    checkpoint = Path(output_dir) / "models" / "mt5"
    if not checkpoint.exists():
        train(config, data_dir, output_dir, smoke)
    device = get_device()
    tokenizer = tokenizer_cls.from_pretrained(checkpoint)
    model = model_cls.from_pretrained(checkpoint).to(device).eval()
    records = read_split(data_dir, "test")
    if smoke:
        records = records[:16]
    predictions = []
    started = time.perf_counter()
    for record in records:
        inputs = tokenizer(f"translate English to French: {record['source']}", return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=config["evaluation"]["max_new_tokens"], do_sample=False)
        predictions.append({**record, "model": "mt5", "prediction": tokenizer.decode(generated[0], skip_special_tokens=True), "raw_output": generated[0].tolist()})
    path = Path(output_dir) / "predictions" / "mt5.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in predictions) + "\n", encoding="utf-8")
    history_path = Path(output_dir) / "training_history" / "mt5.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else {}
    elapsed = time.perf_counter() - started
    history.update({"inference_seconds_including_tokenization": elapsed, "inference_sentences": len(predictions),
                    "seconds_per_sentence": elapsed / len(predictions), "sentences_per_second": len(predictions) / elapsed})
    write_json(history_path, history)
    return predictions


def generate_sampling(config: dict[str, Any], data_dir: str, output_dir: str, temperature: float, seed: int, smoke: bool = False) -> list[dict[str, Any]]:
    _, _, _, model_cls, tokenizer_cls = _require_transformers()
    checkpoint = Path(output_dir) / "models" / "mt5"
    if not checkpoint.exists():
        train(config, data_dir, output_dir, smoke)
    device = get_device()
    tokenizer, model = tokenizer_cls.from_pretrained(checkpoint), model_cls.from_pretrained(checkpoint).to(device).eval()
    records = read_split(data_dir, "test")[: config["evaluation"]["sampling_subset_size"]]
    if smoke:
        records = records[:16]
    torch.manual_seed(seed)
    predictions = []
    for record in records:
        inputs = tokenizer(f"translate English to French: {record['source']}", return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=config["evaluation"]["max_new_tokens"],
                                       do_sample=True, temperature=temperature, top_p=0.95)
        predictions.append({**record, "model": "mt5", "prediction": tokenizer.decode(generated[0], skip_special_tokens=True),
                            "raw_output": generated[0].tolist(), "temperature": temperature, "seed": seed})
    path = Path(output_dir) / "temperature"
    path.mkdir(parents=True, exist_ok=True)
    (path / f"mt5_t{temperature}_seed{seed}.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in predictions) + "\n", encoding="utf-8")
    return predictions
