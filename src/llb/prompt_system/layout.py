"""The file names one prepared prompt-system package is published under.

A leaf module on purpose: the artifact-contract descriptor must name these members without
importing the corpus reader, the tokenizer, or the candidate grid, and `pipeline.py` re-exports
them so every existing importer keeps its own spelling.
"""

ANTHOLOGY_FILE = "anthology.json"
METADATA_FILE = "doc_metadata.json"
MAPPING_FILE = "graph_rag_mapping.json"
CANDIDATES_FILE = "candidates.json"
MANIFEST_FILE = "manifest.json"
