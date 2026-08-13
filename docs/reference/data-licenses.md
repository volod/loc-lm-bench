# Data Licenses and Attribution

The repository's own code is [MIT](../../LICENSE). The DATA it ships, downloads, or points at is
not: each fixture and task source keeps its upstream terms, and those terms travel with anything you
derive from them. This page is the long form of the
[README summary](../../README.md#data-licenses).

Three categories, and the difference between them decides what you owe:

| Category | What it means | Where it lives |
| --- | --- | --- |
| Committed, upstream-derived | Vendored in the repo, derived from a licensed source | [`samples/goldsets/`](../../samples/goldsets/) |
| Not vendored, fetched at run time | The repo names a dataset; the records download to `$DATA_DIR` | public screen, knowledge-cutoff benchmark |
| Repo-authored | Written for this project, covered by the repo license | the remaining `samples/` fixtures |

Your own corpora are a fourth case the repo never touches: restricted material stays local by
design, and no lane sends it off-host without an explicit consent flag
(`FRONTIER_EGRESS_CONSENT=1`, `FRONTIER_MAX_USD=<cap>`).

## Committed fixtures derived from upstream data

### UA-SQuAD post-edited v1 (the default fixture)

[`samples/goldsets/ua_squad_postedited_v1/`](../../samples/goldsets/ua_squad_postedited_v1/) is the
250-item development fixture that `make ingest-uk-squad` and most quickstart flows start from. It
derives from [`FIdo-AI/ua-squad`](https://huggingface.co/datasets/FIdo-AI/ua-squad), pinned to an
exact revision and file checksum in
[`source.json`](../../samples/goldsets/ua_squad_postedited_v1/source.json).

The upstream dataset-card metadata is MIT-marked, but the card also carries a derivative-text note:
SQuAD-derived text inherits [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The
fixture applies the stricter reading, so `goldset.jsonl` and `corpus/**` in that directory are
CC BY-SA 4.0 -- attribution AND share-alike -- per the local
[fixture license](../../samples/goldsets/ua_squad_postedited_v1/LICENSE.md). Preserve the
[SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) attribution in anything you redistribute from
it, including derived chunks, indexes, and exported fine-tuning records.

### Chain-context UA v1 (source excerpts)

[`samples/goldsets/chain_context_uk_v1/`](../../samples/goldsets/chain_context_uk_v1/) is the
20-chain context-policy fixture. It keeps only the exact excerpts cited by human-verified chains
from a copyrighted 2024 Ukrainian publication, attributed in
[`source.json`](../../samples/goldsets/chain_context_uk_v1/source.json). No license to the complete
publication is included with this repository, and copyright stays with the named rightsholders --
see the local [notice](../../samples/goldsets/chain_context_uk_v1/LICENSE.md). Redistributing these
excerpts is your call to make in your jurisdiction.

## Task sources the repo names but does not vendor

### Tier-1 public screen

The public screen runs stock `lm-eval` Ukrainian tasks; it vendors no task records, so the terms
below apply to the data your harness downloads, not to anything in this repository. Check them
before publishing or redistributing results that embed task text:

| Task id | Upstream dataset | Terms |
| --- | --- | --- |
| `belebele_ukr_Cyrl` | [Belebele](https://huggingface.co/datasets/facebook/belebele) | CC BY-SA 4.0 |
| `arc_uk` | [ARC](https://huggingface.co/datasets/allenai/ai2_arc) | CC BY-SA 4.0 |
| `hellaswag_uk` | [HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag) | MIT |
| `m_mmlu_uk` | [MMLU](https://huggingface.co/datasets/cais/mmlu) | MIT |
| `global_piqa_prompted_ukr_cyrl` | [PIQA](https://huggingface.co/datasets/piqa) | license-unknown on the dataset card -- check before redistributing |

The MCQ tasks need token logprobs and therefore a vLLM endpoint; the generation task is the
Ollama-compatible lane. Task ids are overridable per harness build, so a local override changes
which terms apply -- re-check this table when you pass `--tasks`.

### Knowledge-cutoff event set

`make bench-knowledge-cutoff` does not vendor its event set either. It loads
[`apoorvumang/knowledge-cutoff-benchmark`](https://huggingface.co/datasets/apoorvumang/knowledge-cutoff-benchmark),
whose dataset card marks the data CC BY 4.0, and RESOLVES a moving Hugging Face revision to an exact
commit that the run bundle records -- so a published cutoff estimate names the data it was measured
on.

The method and dataset choice are inspired by Apoorv Saxena's
[`knowledge-cutoff`](https://github.com/apoorvumang/knowledge-cutoff) project; no upstream
application source is copied. Preserve that attribution for downloaded or redistributed data.

The Ukrainian bilingual workflow translates the SAME pinned items and freezes a human-reviewed
bundle. A frozen translation is a derivative of the CC BY 4.0 source and carries its attribution
obligation with it. See the
[bilingual workflow](../impl/current/knowledge-cutoff.md#ukrainian-bilingual-calibration-workflow).

## Repo-authored fixtures

Everything else committed under [`samples/`](../../samples/) is repo-authored unless a local
`README.md`, `LICENSE.md`, `source.json`, or `provenance.json` in that directory says otherwise --
the tutorial gold sets (`ip_regulation_uk`), the planted retrieval fixtures
(`apostrophe_variants_uk`), the corpus-hygiene corpora (`conflicts_uk_v1`,
`duplicate_chunks_uk_v1`, `near_duplicate_chunks_uk_v1`, `intra_document_repeats_uk_v1`), the
category-suite seeds, and the synthetic text-analysis bundle. Committed fixtures must be
deterministic, independently attributable, structurally validated, and usable without network
access.

Check the directory before you assume: the two exceptions above both sit in `samples/goldsets/`
beside fixtures that are fully repo-authored.

## Redistribution checklist

Before publishing a corpus, an index, a gold set, or a run bundle derived from any of the above:

1. **Find the source.** A derived artifact inherits terms from whatever it was built on --
   `source.json` / `provenance.json` next to the fixture, or the dataset card for a fetched set.
2. **Carry the notice.** Ship the attribution and license notice with the derivative, not only with
   the original. Share-alike sources (CC BY-SA 4.0) also constrain the license you may publish the
   derivative under.
3. **Check what the artifact embeds.** Chunk stores, gold sets, exported SFT/DPO records, adapters
   trained on them, and answer logs can all contain source text.
4. **Keep restricted corpora local.** Imported external-service drafts and closed-service answer
   logs are scored on-host; nothing leaves without the explicit egress consent flag and spend cap.
5. **Weights are separate.** Model licenses are their own question --
   see [model families](model-families.md#licenses-and-gating).

## Related

- [Model families, tiers, and licenses](model-families.md) -- the terms on the weights.
- [Data prep guide](../guides/data-prep/data-prep.md) -- how corpora and gold sets are built.
- [External AI service artifacts](../guides/data-prep/external-ai-service-artifacts.md) -- what may
  and may not be sent to a closed service, and how imports are grounded.
- [`samples/README.md`](../../samples/README.md) -- what each committed fixture directory is for.
