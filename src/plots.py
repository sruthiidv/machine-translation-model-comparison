"""Plots generated only from saved metrics/history."""
from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd


def make_plots(output_dir: str) -> None:
    output = Path(output_dir)
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.csv"
    if metrics_path.exists():
        frame = pd.read_csv(metrics_path)
        if not frame.empty:
            frame.set_index("model")[["bleu", "chrf"]].plot(kind="bar", title="Translation metrics")
            plt.tight_layout()
            plt.savefig(plot_dir / "comparison.png", dpi=150)
            plt.close()
    temp_path = output / "temperature" / "metrics.csv"
    if temp_path.exists():
        frame = pd.read_csv(temp_path)
        if not frame.empty:
            frame.groupby(["model", "temperature"])["bleu"].mean().unstack(1).plot(marker="o", title="Sampling BLEU by temperature")
            plt.tight_layout()
            plt.savefig(plot_dir / "temperature.png", dpi=150)
            plt.close()
    history_path = output / "training_history" / "transformer.json"
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8")).get("history", [])
        if history:
            frame = pd.DataFrame(history)
            frame.set_index("epoch")[["train_loss", "validation_loss"]].plot(marker="o", title="Transformer training history")
            plt.tight_layout()
            plt.savefig(plot_dir / "training_transformer.png", dpi=150)
            plt.close()
