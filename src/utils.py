"""Shared configuration, reproducibility, metadata, and device helpers."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
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
