# apostrophe_variants_uk -- planted MIXED-apostrophe-variant retrieval fixture

A committed micro-goldset for the question the two real corpora cannot answer: what does
apostrophe-variant tokenization buy when the index and the query disagree about WHICH apostrophe
was typed? Both quickstart corpora are internally consistent about their variant, so index and
query split identically and every recall number is flat before and after the fix
([RAG core](../../../docs/impl/current/rag-core.md#apostrophe-variant-tokenization-evidence)).
This fixture plants the mismatch instead.

Layout:

- `corpus/` -- one registry of 60 near-identical record entries, split across four documents that
  were "converted" by different tools. Each document is internally consistent about its apostrophe
  except the copy-pasted appendix, which alternates between two sources -- so the CORPUS mixes
  variants the way a re-ingested edition or a pasted appendix does.
- `goldset.jsonl` -- 60 `final` items, one per entry, each with an offset-exact source span.

| document | apostrophe | entries | plays the part of |
| --- | --- | ---: | --- |
| `reyestr-osnovnyy.md` | `'` U+0027 | 15 | text typed in an editor |
| `reyestr-perevydannya.md` | `’` U+2019 | 15 | a PDF converter's typographic apostrophe |
| `reyestr-arkhiv.md` | `ʼ` U+02BC | 15 | another converter's modifier letter |
| `dodatok-zmishanyy.md` | `’` / `` ` `` alternating | 8 / 7 | an appendix pasted from two sources |

Every entry differs from every other ONLY in one apostrophe-bearing subject noun
(`Об'єкт обліку: пам'ятка.`); the rest of the entry is byte-identical boilerplate, and the record
number the question asks for never appears in the question. Every question is typed with the
keyboard apostrophe `'`, which is what an operator actually types, so 45 of the 60 items face a
variant mismatch and the other 15 are the same-variant control.

Because the subject noun is the only discriminating token, an index that cannot match it across
the variant boundary has nothing else to rank on -- which is what makes the delta measurable
instead of masked by other query terms.

## What the fixture measures

`tests/llb/rag/test_apostrophe_variant_fixture.py` scores the lexical lane over the committed
corpus twice: with the shipped tokenizer and with the pre-fix v1 one (apostrophe variants were not
in-word characters, unification ran after tokenizing). The numbers below are the assertion, not
documentation of an incidental fact.

| entry variant | items | v1 BM25 candidates for the subject term | v1 lexical recall@10 | v2 |
| --- | ---: | ---: | ---: | ---: |
| `'` U+0027 (control) | 15 | 15 | 1.000 | 1.000 |
| `ʼ` U+02BC | 15 | 15 | 1.000 | 1.000 |
| `’` U+2019 | 23 | 0 | 0.000 | 1.000 |
| `` ` `` U+0060 | 7 | 0 | 0.000 | 1.000 |

Only the PUNCTUATION-class variants were unreachable before the fix: U+02BC is a Unicode modifier
LETTER, so `\w` already kept `памʼятка` whole, while U+2019 and the grave accent split it into two
half-words that a keyboard-typed query could never match. The v1 misses are total -- 0 of 30 --
so nothing is recovered by boilerplate ties either.

The heavy hybrid comparison over the same fixture (real e5-base embeddings, `recursive` 800/120,
k=10, exact duplicate collapse: 180 chunks -> 80 indexed) is recorded in
[RAG core](../../../docs/impl/current/rag-core.md#apostrophe-variant-tokenization-evidence).

Regenerating: the fixture is deterministic (no randomness); edit and re-run the generation snippet
recorded in the git history of this directory if the layout must change. Span offsets are
validated by `load_goldset`, which rejects any drift between `text` and the char offsets.
