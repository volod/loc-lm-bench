"""Fine-tune the pinned text embedder on the operator's own corpus.

The general multilingual encoder is chosen by bake-off, not by assumption
(`llb.rag.embedding_bakeoff`), but every candidate in that bake-off was trained on general web
text: a corpus whose domain terms the encoder never saw loses recall no roster entry can win back.
This lane closes that gap locally -- export (question, gold-chunk) pairs from the TUNING split,
train contrastively against hard lexical negatives, and hand the result back to the same bake-off
so the uplift is measured on the held-out final split by the standard source-span metric, never
claimed.
"""
