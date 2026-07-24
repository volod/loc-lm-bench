# near_duplicate_chunks_uk_v1 -- planted NEAR-duplicate chunk fixture

Three Ukrainian service regulations shaped like converted PDFs. Unlike the exact-duplicate sibling
fixture [`duplicate_chunks_uk_v1`](../duplicate_chunks_uk_v1/README.md), their shared furniture is
repeated with the small differences a PDF conversion actually produces, so each repeated block
lands on a DIFFERENT duplicate tier
([RAG core](../../../docs/impl/current/rag-core.md#duplicate-chunk-collapse)).

| repeated block | how the copies differ | merged by |
| --- | --- | --- |
| `Загальні положення` | nasos == ventylyator byte for byte; kompresor differs in case and doubled spaces | `exact` merges 2, `normalized` merges 3 |
| `Застереження` | nasos == ventylyator byte for byte; kompresor writes `зобов’язань` with U+2019 | `exact` and `normalized` both merge 2 |
| `Ставка збору` | one number per document (15 / 7 / 22 percent) -- a GENUINE content difference | only `masked` merges, and it is wrong to |
| `Колонтитул` | page number per document (1 / 4 / 9 of 12) -- pure page furniture | only `masked` merges, and it is right to |
| `Технічні характеристики` | distinct per document | never merged |

Chunked with `heading` at `size=400` the corpus yields 15 chunks, and the tiers nest:

| tier | groups | collapsed | indexed |
| --- | ---: | ---: | ---: |
| `exact` | 2 | 2 | 13 |
| `normalized` | 2 | 3 | 12 |
| `masked` | 4 | 7 | 8 |

Two of those rows are the point of the fixture:

- The `Застереження` block pins a LIMITATION of reusing the corpus-conflict `hash` tier's
  normalizer: it extracts word tokens before unifying apostrophe variants, so a typographic
  apostrophe (U+2019) splits `зобов’язань` into two tokens and the copy does NOT normalize onto its
  U+0027 twin. Apostrophe-variant furniture is therefore residue the `normalized` tier cannot take.
- The `Ставка збору` block pins the COST of the `masked` tier: digit masking cannot tell a page
  number from a rate, so adopting it on a corpus with numeric facts merges three different rates
  into one indexed passage.

The numbers above are the assertion, not documentation of an incidental fact: changing these files
changes what `tests/llb/rag/test_duplicate_tiers.py` measures.
