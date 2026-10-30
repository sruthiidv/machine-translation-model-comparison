#!/usr/bin/env python
"""Single entry point for preparation, training, generation, evaluation, and analysis."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Keep both `python scripts/run_pipeline.py` and `python -m scripts.run_pipeline`
# usable in Colab and from a fresh checkout.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import prepare_dataset
from src.evaluate import evaluate_directory
from src.ethical_analysis import create_review_artifacts
from src.finetune_mt5 import generate as generate_mt5, train as train_mt5
from src.plots import make_plots
from src.prompted_models import check_model_access, generate_model
from src.reporting import generate_reports
from src.train_transformer import generate as generate_transformer, train as train_transformer
from src.utils import collect_metadata, get_device, load_config, validate_run, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--stage", choices=["prepare", "train", "generate", "evaluate", "temperature", "ethics", "report", "all"], default="all")
    parser.add_argument("--smoke", action="store_true", help="Small CPU-friendly run; never use its results as final assignment results.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir or (config["output"]["smoke_dir"] if args.smoke else config["output"]["full_dir"]))
    data_dir = output_dir / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.smoke and args.stage in {"train", "generate", "temperature", "all"} and not __import__("torch").cuda.is_available():
        raise RuntimeError("The full experiment requires a CUDA GPU. Use --smoke for CPU validation or run the Colab notebook on a GPU.")
    if not args.smoke and args.stage in {"train", "generate", "temperature", "all"}:
        access_revisions = check_model_access(config)
    else:
        access_revisions = {}
    if args.stage in {"prepare", "all"}:
        prepare_dataset(config, output_dir, smoke=args.smoke)
    if not (data_dir / "train.jsonl").exists():
        raise RuntimeError(f"Prepared data not found at {data_dir}; run --stage prepare first.")
    if args.stage in {"train", "all"}:
        train_transformer(config, str(data_dir), str(output_dir), args.smoke)
        if not args.smoke:
            train_mt5(config, str(data_dir), str(output_dir), args.smoke)
    if args.stage in {"generate", "all"}:
        generate_transformer(config, str(data_dir), str(output_dir), args.smoke)
        if not args.smoke:
            generate_mt5(config, str(data_dir), str(output_dir), args.smoke)
            for model_key in ("llama", "gemma"):
                generate_model(config, str(data_dir), str(output_dir), model_key, args.smoke)
    if args.stage in {"evaluate", "all"}:
        evaluate_directory(output_dir)
        create_review_artifacts(str(output_dir))
        make_plots(str(output_dir))
        if not args.smoke:
            generate_reports(output_dir, args.reports_dir)
    if args.stage in {"temperature", "all"} and not args.smoke:
        from src.temperature_analysis import run
        run(config, str(data_dir), str(output_dir), args.smoke)
        make_plots(str(output_dir))
        if not args.smoke:
            generate_reports(output_dir, args.reports_dir)
    if args.stage == "ethics":
        create_review_artifacts(str(output_dir))
        generate_reports(output_dir, args.reports_dir)
    if args.stage == "report":
        if args.smoke:
            raise RuntimeError("Smoke outputs cannot be promoted to final reports.")
        generate_reports(output_dir, args.reports_dir)
    validation = validate_run(output_dir, config, smoke=args.smoke, stage=args.stage)
    if not validation["ok"]:
        raise RuntimeError("Stage validation failed:\n- " + "\n- ".join(validation["errors"]))
    metadata = collect_metadata(config, output_dir.name, get_device())
    metadata["model_revisions"] = access_revisions
    metadata["stage_status"] = validation
    metadata_path = output_dir / "run_metadata.json"
    previous = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    previous.update(metadata)
    write_json(metadata_path, previous)
    print(json.dumps({"output_dir": str(output_dir), "stage": args.stage, "smoke": args.smoke,
                      "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
