"""Download, normalize, split, fingerprint, and persist the OPUS Books data."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset

from .utils import fingerprint_records, set_seed, write_json


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def _extract_pair(row: dict[str, Any], source: str, target: str) -> tuple[str, str]:
    translation = row["translation"]
    return normalize_text(translation[source]), normalize_text(translation[target])


def prepare_dataset(config: dict[str, Any], output_dir: str | Path, smoke: bool = False) -> dict[str, Any]:
    """Create stable JSONL split files and manifests.

    The dataset's upstream train split is shuffled deterministically, then sampled without
    overlap. Training length limits are explicit; validation/test are retained and truncation
    is reported to avoid silently changing the benchmark.
    """
    set_seed(config["seed"])
    data_cfg = config["dataset"]
    output = Path(output_dir) / "data"
    output.mkdir(parents=True, exist_ok=True)
    raw = load_dataset(data_cfg["name"], data_cfg["config"], split="train")
    seen_sources: set[str] = set()
    cleaned: list[dict[str, str]] = []
    stats = Counter()
    for row in raw:
        source, target = _extract_pair(row, data_cfg["source_language"], data_cfg["target_language"])
        if not source or not target:
            stats["empty_removed"] += 1
            continue
        if source in seen_sources:
            stats["duplicate_source_removed"] += 1
            continue
        seen_sources.add(source)
        cleaned.append({"source": source, "target": target})
    cleaned.sort(key=lambda item: hashlib.sha256(item["source"].encode("utf-8")).hexdigest())
    sizes = {key: int(data_cfg[f"{key}_size"]) for key in ("train", "validation", "test")}
    if smoke:
        sizes = {key: min(value, {"train": 64, "validation": 16, "test": 16}[key]) for key, value in sizes.items()}
    train_candidates = [
        item for item in cleaned
        if len(item["source"].split()) <= data_cfg["max_source_tokens"]
        and len(item["target"].split()) <= data_cfg["max_target_tokens"]
    ]
    stats["length_filtered_for_training"] = len(cleaned) - len(train_candidates)
    remaining = [item for item in cleaned if item not in train_candidates[:sizes["train"]]]
    if len(train_candidates) < sizes["train"] or len(remaining) < sizes["validation"] + sizes["test"]:
        raise RuntimeError(f"Not enough usable pairs after training length filtering for requested split sizes: {sizes}.")
    cursor = 0
    splits: dict[str, list[dict[str, Any]]] = {}
    for split_name in ("train", "validation", "test"):
        pool = train_candidates if split_name == "train" else remaining
        split_records = []
        for index, pair in enumerate(pool[cursor : cursor + sizes[split_name]]):
            record = {
                "example_id": f"{split_name}-{index:05d}",
                "source": pair["source"],
                "target": pair["target"],
            }
            if split_name == "test":
                record["source_truncated"] = len(pair["source"].split()) > data_cfg["max_source_tokens"]
                record["target_truncated"] = len(pair["target"].split()) > data_cfg["max_target_tokens"]
            split_records.append(record)
        splits[split_name] = split_records
        cursor += sizes[split_name]
        if split_name == "train":
            cursor = 0
        (output / f"{split_name}.jsonl").write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in split_records) + "\n",
            encoding="utf-8",
        )
    for split_name, records in splits.items():
        write_json(output / f"{split_name}_manifest.json", [item["example_id"] for item in records])
    stats.update({"raw_pairs": len(raw), "usable_pairs": len(cleaned)})
    stats["test_source_truncated"] = sum(bool(x.get("source_truncated")) for x in splits["test"])
    stats["test_target_truncated"] = sum(bool(x.get("target_truncated")) for x in splits["test"])
    manifest = {
        "dataset": data_cfg["name"],
        "config": data_cfg["config"],
        "license_note": "OPUS Books is derived from books; verify upstream dataset/card licensing before redistribution.",
        "domain": "literary/book text",
        "limitations": "Domain mismatch, translation quality variation, and possible duplicate/near-duplicate passages.",
        "seed": config["seed"],
        "sizes": {key: len(value) for key, value in splits.items()},
        "fingerprint": fingerprint_records([item for records in splits.values() for item in records]),
        "split_fingerprint": fingerprint_records(
            [{"split": split, "example_id": item["example_id"], "source": item["source"], "target": item["target"]}
             for split, records in splits.items() for item in records]
        ),
        "statistics": dict(stats),
        "length_policy": data_cfg["length_policy"],
    }
    write_json(output / "preprocessing_stats.json", manifest)
    return {"splits": splits, "manifest": manifest, "directory": str(output)}


def read_split(data_dir: str | Path, split: str) -> list[dict[str, Any]]:
    path = Path(data_dir) / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
