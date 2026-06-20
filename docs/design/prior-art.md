# Prior art (Ukrainian LLM evaluation)

Three efforts we studied and reuse. Net: they reinforce the wedge (public Ukrainian
rankings don't transfer, and even the best one mixes 0-shot and 3-shot), so we consume them
as a prior plus ready-made data and keep the private corpus eval as the decider.

- **MamayLM v2 (INSAIT)** — Ukrainian models (12B / 27B, Gemma-3-based), OpenAI-compatible
  API. Candidate models; the 27B is a possible local judge.
  <https://models.mamay.ai>
- **lm-evaluation-harness-uk (INSAIT)** — EleutherAI fork with Ukrainian tasks (ZNO, MMLU-uk,
  etc.); used as the Tier-1 public screen.
  <https://github.com/insait-institute/lm-evaluation-harness-uk>
- **lang-uk Ukrainian LLM Leaderboard** — task suite + average-rank method; treats
  Summarization + Q&A (Belebele-uk, SQuAD-uk) as a RAG proxy.
  <https://huggingface.co/spaces/lang-uk/ukrainian-llm-leaderboard>

What we reuse: SQuAD-uk / Belebele-uk as the gold-set base and RAG seed; lm-evaluation-
harness-uk as the public screen; average-rank aggregation; MamayLM / Lapa / Gemma as
candidate seeds.

Full detail: [the design spec](../design.md).
