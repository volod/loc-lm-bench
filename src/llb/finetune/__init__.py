"""Local adapter fine-tuning workflow.

The package is intentionally file-driven: dataset export, adapter manifests, contamination
guards, and orchestration state are all plain JSON/JSONL so CI can exercise the complete control
plane without CUDA training dependencies.
"""
