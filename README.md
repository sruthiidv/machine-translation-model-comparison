# Comparative Machine Translation Using Transformer, mT5, LLaMA, and Gemma

## Problem and objectives

This project compares English-to-French translation on the public
[Helsinki-NLP/opus_books](https://huggingface.co/datasets/Helsinki-NLP/opus_books) `en-fr` configuration:

1. a PyTorch encoder-decoder Transformer trained from scratch;
2. `google/mt5-small` fine-tuned on the training split;
3. `meta-llama/Llama-3.2-1B-Instruct` with fixed training-only few-shot prompting; and
4. `google/gemma-3-1b-it` with the same prompting protocol.

The objective is a reproducible educational comparison, not a claim that architecture alone explains quality.
Model size, pretraining, tokenization, precision, and adaptation are confounded.

## Dataset and preprocessing

OPUS Books is literary/book-domain parallel data. The dataset card and upstream license information are authoritative;
check them before redistributing data. The pipeline normalizes whitespace only, preserving punctuation, accents, and
capitalization; removes empty pairs and duplicate normalized sources; hashes and orders records deterministically;
and assigns disjoint train/validation/test manifests with seed 42. Defaults are 10,000/1,000/500 pairs. The
training tokenizer is fit only on training text. Training length limits are explicit; validation/test records are
retained and truncation flags are reported rather than silently removing hard examples. Actual counts and a SHA-256
fingerprint are saved in `data/preprocessing_stats.json`.

## Installation and access

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Set `HF_TOKEN` in the environment or Colab Secrets for gated LLaMA/Gemma access, and accept each model's Hugging
Face license. The code never embeds a token and never substitutes another model. A GPU with at least roughly
16 GB VRAM is recommended for the full sequential run; CPU is supported for smoke tests and the small Transformer.

## Exact execution

```bash
python scripts/run_pipeline.py --smoke --stage all
python scripts/run_pipeline.py --stage prepare
python scripts/run_pipeline.py --stage train
python scripts/run_pipeline.py --stage generate
python scripts/run_pipeline.py --stage evaluate
python scripts/run_pipeline.py --stage temperature
python scripts/run_pipeline.py --stage report
```

Use `--output-dir results/my_run` to create an isolated run. Smoke output is intentionally separate and must never be
presented as final assignment results. Full checkpoints are ignored by Git; small metrics, plots, and review samples
can be retained. Add `--reports-dir results/reports` when reports should be persisted beside a Drive-backed run.
Stages are resumable at the data-fingerprint/checkpoint level, and training scripts select validation-best checkpoints.

## Repository structure

`src/data.py` prepares stable splits; `src/tokenization.py` and `src/transformer.py` implement the custom model;
`src/finetune_mt5.py` fine-tunes mT5; `src/prompted_models.py` handles native chat templates and new-token-only
decoding; `src/evaluate.py`, `src/temperature_analysis.py`, `src/ethical_analysis.py`, and `src/plots.py` produce
analysis outputs. `scripts/run_pipeline.py` is the single entry point. The Colab notebook calls these same modules.

## Colab

Open `notebooks/assignment3_colab.ipynb`, select a GPU runtime, install requirements, add `HF_TOKEN` to Colab
Secrets, and execute cells in order. The notebook runs preparation, training, generation, evaluation, sampling, ethics,
and final comparison without duplicating implementation. Models are loaded sequentially to limit memory.

## Outputs and evaluation

The full run writes `results/metrics.csv`, `run_metadata.json`, predictions, training histories, temperature
outputs/metrics, plots, and a review queue. Metrics use corpus SacreBLEU and chrF with saved signatures. Main
comparison decoding is deterministic greedy; sampling temperatures are not interpreted under greedy decoding.
Inference timing excludes model loading and includes tokenization, as recorded in each training-history JSON. The
same ordered test set is used for every model. CUDA timing is synchronized where applicable.

## Required sampling experiment

`--stage temperature` is separate from the main comparison and uses temperatures 0.3, 0.7, and 1.0, three fixed
seeds, the first 50 records of the persisted ordered test split, enabled sampling, and saved outputs plus aggregate
BLEU/chrF mean and standard deviation. It must
not replace or alter deterministic benchmark scores.

## Responsible AI

`ethical_review.jsonl` contains actual model outputs for bilingual human review with labels initially marked `pending`.
Reviewers should record evidence for bias, hallucinated additions, cultural misrepresentation, sensitive-content
mistranslation, names/numbers, gender ambiguity, idioms, and omissions. Potential risks are not reported as observed
errors. A supplemental challenge set may be supplied separately and is excluded from benchmark scores. Mitigations
include bilingual review, terminology constraints, representative data, targeted challenge cases, and escalation for
high-stakes use.

## Results status and conclusion

Before a real full run, results are **pending**. Do not copy smoke metrics into an assignment report. The `evaluate`,
`temperature`, or explicit `report` stage updates `reports/model_comparison.md` and `reports/ethical_analysis.md` from
genuine saved artifacts. The report identifies the highest measured quality and fastest measured inference model
separately; no evidence-based winner is named before the full experiment completes.
