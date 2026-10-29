"""Separate sampled-generation experiment; temperature is never used for greedy scores."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import sacrebleu

from .finetune_mt5 import generate_sampling as generate_mt5_sampling
from .prompted_models import generate_model
from .train_transformer import generate_sampling as generate_transformer_sampling


def run(config, data_dir: str, output_dir: str, smoke: bool = False) -> pd.DataFrame:
    temperatures = config["evaluation"]["sampling_temperatures"]
    seeds = config["evaluation"]["sampling_seeds"]
    rows = []
    for model_key in ["basic_transformer", "mt5", "llama", "gemma"]:
        for temperature in temperatures:
            for seed in seeds:
                if model_key == "basic_transformer":
                    predictions = generate_transformer_sampling(config, data_dir, output_dir, temperature, seed, smoke)
                elif model_key == "mt5":
                    predictions = generate_mt5_sampling(config, data_dir, output_dir, temperature, seed, smoke)
                else:
                    predictions = generate_model(config, data_dir, output_dir, model_key, smoke, True, temperature, seed)
                hypotheses, references = [x["prediction"] for x in predictions], [x["target"] for x in predictions]
                rows.append({"model": model_key, "temperature": temperature, "seed": seed, "bleu": sacrebleu.corpus_bleu(hypotheses, [references]).score, "chrf": sacrebleu.corpus_chrf(hypotheses, [references]).score})
    frame = pd.DataFrame(rows)
    temperature_dir = Path(output_dir) / "temperature"
    frame.to_csv(temperature_dir / "metrics.csv", index=False)
    if not frame.empty:
        aggregate = frame.groupby(["model", "temperature"]).agg(
            bleu_mean=("bleu", "mean"), bleu_std=("bleu", "std"),
            chrf_mean=("chrf", "mean"), chrf_std=("chrf", "std"),
            seeds=("seed", "count"),
        ).reset_index()
        aggregate.to_csv(temperature_dir / "aggregated_metrics.csv", index=False)
        (temperature_dir / "settings.json").write_text(json.dumps({
            "subset_size": config["evaluation"]["sampling_subset_size"],
            "temperatures": temperatures, "seeds": seeds, "sampling_enabled": True,
            "top_p": config["evaluation"]["sampling_top_p"],
            "note": "The subset is the first N records in the persisted ordered test split; it is separate from main scores.",
        }, indent=2), encoding="utf-8")
    return frame
