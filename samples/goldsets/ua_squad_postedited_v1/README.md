# UA-SQuAD post-edited development fixture

This directory is the repository's stable, human-reviewed Ukrainian RAG development gold set.
It contains 250 canonical items and 250 matching source documents. Tests and `make demo-eval`
use these committed files; they do not download or regenerate a gold set.

## Provenance and verification basis

- Upstream dataset: `FIdo-AI/ua-squad`
- Upstream revision: `943ef27daea65e400350ef1875d07c7e97288177`
- Source split/file: validation / `val.json`
- Selection: first grounded QA per distinct context in upstream order
- Upstream curation statement: Ukrainian translations/adaptations were post-edited and answer
  spans were aligned and checked for malformed entries.

Items use `provenance: public-reused` and `verified: true`. The pinned 250-item selection was
reviewed by a human and passes local source-span validation. The upstream post-editing claim is
supporting provenance, not a substitute for this project's review. This remains a public-data
development fixture and must not be presented as a private-corpus benchmark.

`source.json` pins the source digest and selection rule. Regeneration is intentionally strict:

    python -m llb.prep.published_goldset \
      --source /path/to/val.json \
      --out-dir samples/goldsets/ua_squad_postedited_v1

The runtime importer can reproduce the same reviewed bundle under `$DATA_DIR` from the pinned
Hub revision without modifying this committed directory:

    make ingest-uk-squad GOLDSET_MODE=development GOLDSET_N=250

## License and attribution

The upstream card identifies the data as a Ukrainian derivative of Stanford SQuAD and notes
CC BY-SA 4.0 obligations for redistributed text. This fixture is redistributed under those
data terms; the repository's software license does not replace them.

- UA-SQuAD: https://huggingface.co/datasets/FIdo-AI/ua-squad
- `ua_datasets`: https://github.com/fido-ai/ua-datasets
- SQuAD: https://rajpurkar.github.io/SQuAD-explorer/
- CC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/

If this fixture is redistributed or used in published work, preserve this attribution and
follow the upstream dataset card's citation instructions.
