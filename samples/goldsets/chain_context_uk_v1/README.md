# Verified Ukrainian chain-context fixture

This fixture contains 20 human-accepted, `verified=true` two-step chains for context-policy
benchmarking. All rows use the `final` split and pass exact-span chain validation.

The corpus is intentionally compact. It contains only the 36 distinct source spans cited by the
reviewed chains, with offsets remapped during promotion. It does not contain the complete source
publication. See `source.json` and `LICENSE.md` for attribution and reuse constraints.

Validate the fixture from the repository root:

```bash
make validate-goldset \
  CHAINS=samples/goldsets/chain_context_uk_v1/chains.jsonl \
  CORPUS=samples/goldsets/chain_context_uk_v1/corpus
```

`fixture_manifest.json` records the chain count, minimum gate, original document digest, and
compact-corpus statistics.
