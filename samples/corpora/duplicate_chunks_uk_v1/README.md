# duplicate_chunks_uk_v1 -- planted duplicate-chunk fixture

Three Ukrainian equipment manuals shaped like converted PDFs: each one has its own technical
section, and each one repeats the SAME page furniture -- a legal preamble, a mode-legend table,
and a service-contact block. That is the structure that makes a converted-PDF index spend a large
share of its budget on text it already holds, and makes identical chunks tie exactly at retrieval
time. See [duplicate chunk collapse](../../../docs/impl/current/rag-core.md#duplicate-chunk-collapse)
for the behavior under test.

| document | unique content | repeats |
| --- | --- | --- |
| `manual-nasos.md` | pump NV-200 ratings | preamble, mode legend, service contacts |
| `manual-kompresor.md` | compressor KG-45 ratings | preamble, mode legend, service contacts |
| `manual-ventylyator.md` | fan VP-12 ratings | preamble, mode legend, service contacts |

Chunked with `heading` at `size=400`, the corpus yields 12 chunks in 3 identical groups of 3 plus
3 distinct technical sections: 9 of 12 chunks (75.0%) are byte-identical to another chunk, and
collapsing them leaves 6 indexed chunks. Every document keeps a distinct section, so a gold span
labeled on any copy -- including a copy inside a collapsed group -- must still be retrievable
after collapse; `tests/llb/rag/test_duplicates.py` pins exactly that.

The numbers above are the assertion, not documentation of an incidental fact: changing these
files changes what the duplicate tests measure.
