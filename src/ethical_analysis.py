"""Create a factual human-review table and rubric without inventing judgments."""
from __future__ import annotations

import json
from pathlib import Path


def create_review_artifacts(output_dir: str, challenge_path: str | None = None) -> None:
    output = Path(output_dir)
    prediction_path = output / "predictions" / "all_predictions.jsonl"
    rows = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines() if line.strip()] if prediction_path.exists() else []
    review = output / "ethical_review.jsonl"
    review.write_text("\n".join(json.dumps({**row, "human_label": "pending", "review_notes": ""}, ensure_ascii=False) for row in rows[:100]) + "\n", encoding="utf-8")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/ethical_analysis.md").write_text(
        "# Responsible-AI analysis\n\n"
        "The generated table is a review queue, not evidence of human evaluation. Reviewers should label "
        "adequacy, omissions, hallucinated additions, names/numbers, gender ambiguity, idioms, cultural "
        "misrepresentation, and sensitive-content mistranslation. Record evidence in `ethical_review.jsonl`.\n\n"
        "## Potential risks and mitigations\n"
        "- Bias and cultural misrepresentation: use representative bilingual review and domain terminology lists.\n"
        "- Hallucinated additions or omissions: compare source/reference and require human approval for high-stakes use.\n"
        "- Names, numbers, gender, idioms, and sensitive content: targeted challenge cases, constrained decoding, and escalation.\n"
        "- Never treat benchmark scores as a safety evaluation. A supplemental challenge set is kept outside benchmark metrics.\n",
        encoding="utf-8",
    )
