# Comparative Machine Translation

This project compares English-to-French translation using:

1. a basic encoder-decoder Transformer trained from scratch;
2. `google/mt5-small` fine-tuned with supervised parallel data;
3. `meta-llama/Llama-3.2-1B-Instruct` with fixed few-shot prompting; and
4. `google/gemma-3-1b-it` with the same prompting protocol.

This is an educational comparison. Model size, pretraining, tokenization, precision, and adaptation method differ,
so the results are not an architecture-only comparison.

## Dataset

The project uses the public [Helsinki-NLP/opus_books](https://huggingface.co/datasets/Helsinki-NLP/opus_books)
English-French (`en-fr`) configuration. OPUS Books is literary/book-domain parallel text; consult the upstream
dataset card and license before redistributing data.

Preprocessing:

- whitespace normalization;
- empty-pair and duplicate-source removal;
- deterministic ordering with seed 42;
- disjoint train/validation/test manifests;
- tokenizer fitting on training text only; and
- explicit length limits with truncation flags retained for validation/test examples.

The configured split is 10,000 training pairs, 1,000 validation pairs, and 500 test pairs. The split fingerprint and
preprocessing statistics are saved in [`results/data/preprocessing_stats.json`](results/data/preprocessing_stats.json).

## Models and evaluation

All four models translate the same ordered 500-example test set. The pipeline records:

- corpus SacreBLEU and chrF;
- training or fine-tuning time where applicable;
- inference time including tokenization;
- total and trainable parameter counts;
- deterministic main predictions;
- enabled-sampling temperature analysis; and
- representative outputs for bilingual ethical review.

The main comparison uses deterministic greedy decoding. Temperature sampling is a separate experiment and must not
replace the main benchmark.

## Measured benchmark results

The captured full Colab run produced these main results:

| Model | BLEU | chrF | Parameters | Seconds/sentence |
|---|---:|---:|---:|---:|
| Basic Transformer | 0.41 | 9.44 | 15.3M | 0.1164 |
| mT5-small | 0.01 | 1.73 | 300.2M | 0.0624 |
| LLaMA | 13.10 | 39.65 | 1.24B | 1.0164 |
| Gemma | **15.59** | **42.22** | 999.9M | 1.7936 |

Gemma achieved the highest measured BLEU and chrF. The short mT5 run completed with finite loss, but its very low
score and repeated `<extra_id_0>` outputs show that it did not learn a useful translation model under the modest
assignment compute budget. The basic Transformer also produced many `<unk>` tokens. These limitations are reported
instead of being hidden.

## Repository layout

- [`notebooks/assignment3_colab.ipynb`](notebooks/assignment3_colab.ipynb) — one resumable Colab workflow.
- [`scripts/run_pipeline.py`](scripts/run_pipeline.py) — preparation, training, generation, evaluation, and reporting.
- [`src/`](src/) — dataset, model, inference, evaluation, plotting, reporting, and ethics modules.
- [`configs/default.yaml`](configs/default.yaml) — reproducible dataset, model, and evaluation settings.
- [`results/data/`](results/data/) — persisted split files, manifests, and preprocessing statistics.
- [`results/predictions/`](results/predictions/) — four model outputs and the combined output file.
- [`results/metrics.csv`](results/metrics.csv) — main BLEU/chrF/timing/parameter comparison.
- [`results/plots/`](results/plots/) — generated comparison and Transformer training plots.

Large model weights and optimizer checkpoints are intentionally excluded from GitHub. They are retained separately in
the Colab/Drive archive when needed.

## Installation

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
```

LLaMA and Gemma require a Hugging Face token and accepted model licenses. Set `HF_TOKEN` in the environment or use
Colab Secrets. Never commit tokens.

## Execution

For a local CPU smoke test:

```bash
python scripts/run_pipeline.py --smoke --stage all
```

For the full experiment on a GPU:

```bash
python -m scripts.run_pipeline --stage prepare --output-dir results
python -m scripts.run_pipeline --stage train --output-dir results
python -m scripts.run_pipeline --stage generate --output-dir results
python -m scripts.run_pipeline --stage evaluate --output-dir results --reports-dir results/reports
python -m scripts.run_pipeline --stage temperature --output-dir results --reports-dir results/reports
python -m scripts.run_pipeline --stage report --output-dir results --reports-dir results/reports
```

The runner validates checkpoint presence, finite metrics, and exact prediction coverage. A zero exit code alone is
not considered successful. Predictions are written incrementally so an interrupted generation stage can resume.
Models are loaded sequentially and GPU memory is released between models.

The mT5 run uses FP32, small batches, gradient accumulation, finite loss/gradient checks, a short configurable
training budget, and model-only `safetensors` checkpoints. Model-only checkpoints cannot provide exact optimizer-state
resumption. Previous failed-run metadata is preserved separately; no failed checkpoints or unrelated files are
deleted automatically.

## Responsible AI

The ethics output is a review queue containing source text, reference translations, model translations, and a
`pending` human-review label. It is not a fabricated human evaluation. Reviewers should examine hallucinations,
omissions, names and numbers, gender ambiguity, idioms, cultural misrepresentation, bias, and sensitive-content
mistranslation. Potential risks should only be marked as observed errors when supported by bilingual review.

Recommended mitigations include representative evaluation data, bilingual review, terminology constraints, targeted
challenge examples outside benchmark scores, and escalation for high-stakes use.

## Submission notes

Commit the source code, notebook, README, lightweight metrics, plots, reports, manifests, and generated text samples.
Do not commit model weights, optimizer state, tokens, caches, or the large ZIP archive. The assignment also requires
a separate student-specific Word document converted to PDF containing the student name, registration number, assignment
title, dataset link, and GitHub repository link; submit that PDF through the LMS.
