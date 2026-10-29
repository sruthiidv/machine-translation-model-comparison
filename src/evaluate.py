"""Metrics, prediction alignment, and reproducible comparison tables."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import sacrebleu



def score(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions:
        raise ValueError("Cannot score empty predictions.")
    references = [row["target"] for row in predictions]
    hypotheses = [row["prediction"] for row in predictions]
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    return {"bleu": bleu.score, "chrf": chrf.score, "n": len(predictions), "bleu_signature": str(bleu), "chrf_signature": str(chrf)}


def evaluate_directory(output_dir: str | Path, models: list[str] | None = None) -> pd.DataFrame:
    output = Path(output_dir)
    files = sorted(path for path in (output / "predictions").glob("*.jsonl") if path.name != "all_predictions.jsonl")
    if models:
        files = [path for path in files if path.stem in models]
    rows = []
    all_predictions = []
    for path in files:
        predictions = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        metrics = score(predictions)
        all_predictions.extend(predictions)
        history_name = "transformer" if path.stem == "basic_transformer" else path.stem
        history_path = output / "training_history" / f"{history_name}.json"
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else {}
        rows.append({"model": path.stem, **metrics,
                     "training_seconds": history.get("training_seconds"),
                     "inference_seconds": history.get("inference_seconds_including_tokenization"),
                     "seconds_per_sentence": history.get("seconds_per_sentence"),
                     "sentences_per_second": history.get("sentences_per_second"),
                     "parameters": history.get("parameters"),
                     "parameters_updated": history.get("parameters_updated"),
                     "adaptation": history.get("adaptation")})
    frame = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "metrics.csv", index=False)
    (output / "predictions" / "all_predictions.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in all_predictions) + "\n", encoding="utf-8")
    return frame


def benchmark_inference(fn, records: list[dict[str, Any]], warmup: int = 1) -> dict[str, float]:
    for row in records[:warmup]:
        fn(row)
    started = time.perf_counter()
    for row in records:
        fn(row)
    elapsed = time.perf_counter() - started
    return {"total_seconds": elapsed, "seconds_per_sentence": elapsed / len(records), "sentences_per_second": len(records) / elapsed}
