# Evaluation & scoring

## Gold set
Per item: question + reference answer + SOURCE-SPAN labels (doc id + char offsets, not
chunk ids, so they survive `chunk_size` tuning). Only `verified: true` items score models.
Target ~200-300 items, partitioned into DISJOINT calibration / tuning / final splits so
tuning never leaks into the leaderboard number.

## Two scoring layers
- **Retrieval quality** (recall@k, MRR by span overlap) validates the EMBEDDING. It is
  constant across generation models under pinned retrieval, so it is NOT a model-ranking
  axis.
- **Generation quality** ranks models: reference-based answer correctness (objective) plus
  Ragas faithfulness / relevance via a gated LLM judge.

## Gated judge
The judge counts only after Ukrainian calibration: Spearman rho >= 0.6 against ~50-80 human
ratings (including adversarial fluent-but-wrong answers). Below the bar, it is demoted to a
diagnostic and objective scores carry the ranking.

## Ranking
Average-rank headline (scale-invariant, comparable to lang-uk) plus a configurable
weighted-blend view, over a Pareto table (quality vs speed vs VRAM), with confidence
intervals. Tier-1 screen metrics and Tier-2 private metrics are never mixed in one rank;
rank flips within overlapping CIs are marked unresolved.

Full detail: [the design spec](../design.md).
