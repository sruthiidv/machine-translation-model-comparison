# Responsible-AI analysis

The generated table is a review queue, not evidence of human evaluation. Reviewers should label adequacy, omissions, hallucinated additions, names/numbers, gender ambiguity, idioms, cultural misrepresentation, and sensitive-content mistranslation. Record evidence in `ethical_review.jsonl`.

## Potential risks and mitigations
- Bias and cultural misrepresentation: use representative bilingual review and domain terminology lists.
- Hallucinated additions or omissions: compare source/reference and require human approval for high-stakes use.
- Names, numbers, gender, idioms, and sensitive content: targeted challenge cases, constrained decoding, and escalation.
- Never treat benchmark scores as a safety evaluation. A supplemental challenge set is kept outside benchmark metrics.
