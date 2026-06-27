# Committed gold-set fixtures

- `ua_squad_postedited_v1/` is the default stable public development fixture: 250 verified
  canonical items, matching corpus documents, pinned source metadata, and data-license notice.
- `ip_regulation_uk/` is the compact prompt-system tutorial fixture: 8 verified items over the
  committed IP regulation corpus, split into `tuning` and `final` for prompt-selection examples.

Committed fixtures must be deterministic, independently attributable, structurally validated,
and usable without network access. Runtime downloads, generated drafts, private corpora, and
manual-review working files belong under `$DATA_DIR`, not in this directory.
