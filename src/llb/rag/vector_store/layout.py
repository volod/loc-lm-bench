"""The file names one built vector-store generation is published under.

A leaf module on purpose: the artifact-contract gate must name these members without importing
the chunker, the encoder, or a vector backend, and `build.py` re-exports them so every existing
importer keeps its own spelling.
"""

CHUNKS_FILE = "chunks.jsonl"  # the INDEXED units (children in parent_child mode)

PARENTS_FILE = "parents.jsonl"  # the parent docstore (parent_child mode only)

META_FILE = "store_meta.json"

LEXICAL_FILE = "lexical_index.json"  # BM25 postings beside the vector index (hybrid mode)
