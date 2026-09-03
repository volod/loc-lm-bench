# Gold Item Contract And Splits

Part of the [Data prep](../data-prep.md) area of the
[current implementation index](../../current.md).

Gold rows and chains are registered artifact contracts (`llb.gold-item` at
`2.0.0`, `llb.gold-chain` at `1.0.0`): each row carries its own `schema_id`/`schema_version`, and a
row written before the registry existed is migrated on load. See
[data-prep contracts](../artifact-contracts/data-prep-contracts.md).

## Gold Item Contract

`src/llb/goldset/schema.py` defines `GoldItem` and `SourceSpan`.

```text
id
lang
question
reference_answer
source_doc_id
source_spans[{doc_id, char_start, char_end, text}]
provenance
verified
split
```

The important design choice is span identity. Labels use document-relative character offsets rather
than chunk ids, because chunking strategy is an experiment variable. `load_goldset` and
`dump_goldset` read and write UTF-8 JSONL.

## Splits And Validation

`src/llb/goldset/splits.py` assigns deterministic disjoint
`calibration`, `tuning`, and `final` splits. `src/llb/goldset/validate.py` checks ids, split
labels, corpus references, and exact span text.

```bash
make validate-goldset
```

The default target validates `samples/goldsets/ua_squad_postedited_v1/goldset.jsonl` against its
sibling `corpus/`.

## Committed Fixture

`samples/goldsets/ua_squad_postedited_v1/` is the default development fixture:

- `goldset.jsonl`: 250 verified Ukrainian QA items;
- `corpus/`: one exact source document per item;
- `source.json`: upstream identity, revision, source digest, selection rule, and license context.

The fixture is intentionally small and stable. It is suitable for smoke checks, retrieval
comparisons, and development regressions. It is not a substitute for evaluating a private target
corpus.
