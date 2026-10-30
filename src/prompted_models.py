"""Few-shot inference for LLaMA and Gemma using their native chat templates."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from .data import read_split
from .utils import get_device, write_json


def check_model_access(config: dict[str, Any]) -> dict[str, str]:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for the gated LLaMA and Gemma models. Set it in the environment or Colab Secrets.")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    revisions = {}
    for model_name in config["prompted"]["models"].values():
        try:
            revisions[model_name] = api.model_info(model_name).sha
        except Exception as exc:
            raise RuntimeError(f"Cannot access {model_name}. Accept its Hugging Face license and check HF_TOKEN.") from exc
    return revisions


def _load(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    token = os.environ.get("HF_TOKEN")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        model = AutoModelForCausalLM.from_pretrained(model_name, token=token, torch_dtype="auto", device_map="auto")
    except Exception as exc:
        raise RuntimeError(f"Unable to load {model_name}. Check HF_TOKEN, license acceptance, and GPU memory.") from exc
    return tokenizer, model


def _prompt(tokenizer, examples: list[dict[str, str]], source: str) -> str:
    messages = [{"role": "system", "content": "Translate English to French. Return only the French translation."}]
    for example in examples:
        messages.extend([
            {"role": "user", "content": example["source"]},
            {"role": "assistant", "content": example["target"]},
        ])
    messages.append({"role": "user", "content": source})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_model(config: dict[str, Any], data_dir: str, output_dir: str, model_key: str, smoke: bool = False, sampling: bool = False, temperature: float = 0.7, seed: int = 42) -> list[dict[str, Any]]:
    model_name = config["prompted"]["models"][model_key]
    tokenizer, model = _load(model_name)
    device = get_device()
    train_records = read_split(data_dir, "train")
    test_records = read_split(data_dir, "test")
    if smoke:
        train_records, test_records = train_records[:3], test_records[:16]
    elif sampling:
        test_records = test_records[:config["evaluation"]["sampling_subset_size"]]
    examples = train_records[:config["prompted"]["few_shot_count"]]
    path = (Path(output_dir) / "temperature" / f"{model_key}_t{temperature}_seed{seed}.jsonl"
            if sampling else Path(output_dir) / "predictions" / f"{model_key}.jsonl")
    existing = {json.loads(line)["example_id"]: json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines() if line.strip()} if path.exists() else {}
    results = []
    started = time.perf_counter()
    for record in test_records:
        if record["example_id"] in existing:
            results.append(existing[record["example_id"]])
            continue
        prompt = _prompt(tokenizer, examples, record["source"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=config["prompted"]["max_input_tokens"]).to(device)
        if sampling:
            torch.manual_seed(seed)
        with torch.no_grad():
            generated = model.generate(
                **inputs, max_new_tokens=config["prompted"]["max_new_tokens"], do_sample=sampling,
                temperature=temperature if sampling else 1.0, top_p=config["evaluation"]["sampling_top_p"] if sampling else 1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[0, inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
        prediction = raw.strip().splitlines()[0].strip() if raw.strip() else ""
        results.append({**record, "model": model_key, "prediction": prediction, "raw_output": raw, "temperature": temperature if sampling else None})
    elapsed = time.perf_counter() - started
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in results) + "\n", encoding="utf-8")
    if not sampling:
        write_json(Path(output_dir) / "training_history" / f"{model_key}.json", {"training_seconds": "N/A — prompting only", "inference_seconds_including_tokenization": elapsed, "inference_sentences": len(results), "seconds_per_sentence": elapsed / len(results), "sentences_per_second": len(results) / elapsed, "parameters": sum(p.numel() for p in model.parameters()), "parameters_updated": 0, "adaptation": "few-shot prompting", "model_name": model_name})
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results
