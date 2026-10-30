"""Shared configuration, reproducibility, metadata, and device helpers."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preferred: str = "auto") -> torch.device:
    if preferred == "cpu" or (preferred == "auto" and not torch.cuda.is_available()):
        return torch.device("cpu")
    if preferred in {"cuda", "auto"} and torch.cuda.is_available():
        return torch.device("cuda")
    raise RuntimeError(f"Requested device {preferred!r} is unavailable.")


def fingerprint_records(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def collect_metadata(config: dict[str, Any], run_id: str, device: torch.device) -> dict[str, Any]:
    versions = {}
    for package in ("datasets", "transformers", "sacrebleu", "pandas"):
        try:
            module = __import__(package)
            versions[package] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[package] = "not-installed"
    return {
        "run_id": run_id,
        "seed": config["seed"],
        "device": str(device),
        "cuda": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "git_revision": git_revision(),
        "hf_token_present": bool(os.environ.get("HF_TOKEN")),
        "dependency_versions": {"torch": torch.__version__, **versions},
        "config": config,
    }


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def available_gib(path: str | Path) -> float:
    return shutil.disk_usage(Path(path)).free / (1024 ** 3)


def validate_run(output_dir: str | Path, config: dict[str, Any], smoke: bool = False,
                 stage: str = "all") -> dict[str, Any]:
    """Validate persisted evidence; a successful process alone is not sufficient."""
    output = Path(output_dir)
    errors: list[str] = []
    required = [output / "data" / "preprocessing_stats.json"]
    if stage in {"train", "generate", "evaluate", "temperature", "ethics", "report", "all"}:
        required.extend([output / "models" / "best.pt", output / "training_history" / "transformer.json"])
    if stage in {"generate", "evaluate", "temperature", "all"} and not smoke:
        required.extend([output / "models" / "mt5" / "config.json",
                         output / "training_history" / "mt5.json"])
        for model in ("basic_transformer", "mt5", "llama", "gemma"):
            required.append(output / "predictions" / f"{model}.jsonl")
    if stage in {"evaluate", "all"}:
        required.extend([output / "metrics.csv", output / "plots" / "comparison.png",
                         output / "ethical_review.jsonl"])
    if stage in {"temperature", "all"} and not smoke:
        required.extend([output / "temperature" / "aggregated_metrics.csv",
                         output / "plots" / "temperature.png"])
    for path in required:
        if not path.exists():
            errors.append(f"missing required artifact: {path}")
    expected_ids = []
    test_path = output / "data" / "test.jsonl"
    if test_path.exists():
        expected_ids = [json.loads(line)["example_id"] for line in test_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if stage in {"generate", "evaluate", "temperature", "all"} and expected_ids:
        for model in ("basic_transformer", "mt5", "llama", "gemma"):
            path = output / "predictions" / f"{model}.jsonl"
            if path.exists():
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                if [row.get("example_id") for row in rows] != expected_ids[:len(rows)] or len(rows) != len(expected_ids):
                    errors.append(f"prediction coverage mismatch for {model}: {len(rows)}/{len(expected_ids)}")
    metrics_path = output / "metrics.csv"
    if metrics_path.exists() and stage in {"evaluate", "all"}:
        import pandas as pd
        metrics = pd.read_csv(metrics_path)
        if metrics.empty or not metrics[["bleu", "chrf"]].map(lambda value: value == value and value not in (float("inf"), float("-inf"))).all().all():
            errors.append("metrics are empty or non-finite")
    return {"ok": not errors, "errors": errors, "checked_at": __import__("datetime").datetime.now().isoformat()}
