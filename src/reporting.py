"""Build evidence-based reports from saved experiment artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def generate_reports(output_dir: str | Path, reports_dir: str | Path = "reports") -> None:
    output = Path(output_dir)
    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    history_dir = output / "training_history"
    histories = {path.stem: _read_json(path) for path in history_dir.glob("*.json")}
    if metrics.empty:
        comparison = "No full-run metrics are available yet. Run the full pipeline before making performance claims."
    else:
        quality = metrics.sort_values(["bleu", "chrf"], ascending=False).iloc[0]
        efficiency = metrics.dropna(subset=["seconds_per_sentence"]).sort_values("seconds_per_sentence").iloc[0] if "seconds_per_sentence" in metrics and metrics["seconds_per_sentence"].notna().any() else None
        comparison = (
            f"Measured highest BLEU model: **{quality['model']}** (BLEU {quality['bleu']:.3f}, chrF {quality['chrf']:.3f}). "
            + (f"Measured fastest inference model: **{efficiency['model']}** ({efficiency['seconds_per_sentence']:.4f} s/sentence)." if efficiency is not None else "Inference timing is not available.")
        )
    (reports / "model_comparison.md").write_text(
        "# Model comparison\n\n"
        "This report is generated from saved artifacts, not placeholder values.\n\n"
        f"## Result status\n\n{comparison}\n\n"
        "Quality and efficiency are separate criteria. Differences are confounded by parameter count, pretraining, "
        "tokenization, precision, and adaptation: the custom Transformer is trained from scratch, mT5 is fine-tuned, "
        "and LLaMA/Gemma use few-shot prompting. Therefore this is not an architecture-only comparison.\n\n"
        "## Saved metrics\n\n"
        + (metrics.to_markdown(index=False) if not metrics.empty else "Metrics pending.") + "\n\n"
        "## Training evidence\n\n"
        + json.dumps(histories, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (reports / "ethical_analysis.md").write_text(
        "# Responsible-AI analysis\n\n"
        "The benchmark outputs are available in `results/predictions/` and the review queue is "
        "`results/ethical_review.jsonl`. Labels begin as `pending`; no human judgment is inferred from automatic metrics.\n\n"
        "Review actual examples for bias, hallucinated additions, cultural misrepresentation, sensitive-content "
        "mistranslation, names/numbers, gender ambiguity, idioms, and omissions. Mark observed errors only when a "
        "bilingual reviewer records supporting evidence. Potential risks include domain mismatch, fluent hallucination, "
        "and terminology drift. Mitigations include bilingual review, representative data, terminology constraints, "
        "targeted challenge cases kept outside benchmark scores, and escalation for high-stakes use.\n",
        encoding="utf-8",
    )
