# Committed gold-set fixtures

- `ua_squad_postedited_v1/` is the default stable public development fixture: 250 verified
  canonical items, matching corpus documents, pinned source metadata, and data-license notice.
- `ip_regulation_uk/` is the compact prompt-system tutorial fixture: 8 verified items over the
  committed IP regulation corpus, split into `tuning` and `final` for prompt-selection examples.
- `chain_context_uk_v1/` is the compact context-policy fixture: 20 human-verified two-step chains
  and the 36 exact attributed source spans they cite.
- `apostrophe_variants_uk/` is the planted mixed-apostrophe-variant retrieval fixture: 60 entries
  across four documents "converted" with four different apostrophes, measuring what
  apostrophe-variant tokenization buys when index and query disagree ([its
  README](apostrophe_variants_uk/README.md)).

Committed fixtures must be deterministic, independently attributable, structurally validated,
and usable without network access. Runtime downloads, generated drafts, private corpora, and
manual-review working files belong under `$DATA_DIR`, not in this directory.
