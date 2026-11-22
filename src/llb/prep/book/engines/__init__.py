"""Engine adapters for the block ledger.

Importing this package imports no recognition library. Paddle, RapidOCR, MinerU, and Docling run
in their own environments; the vLLM engines run in their own container.
"""

from llb.prep.book.engines.protocol import EngineRejected, EngineUnavailable
from llb.prep.book.engines.registry import AdapterRegistry, default_adapter_registry

__all__ = [
    "AdapterRegistry",
    "EngineRejected",
    "EngineUnavailable",
    "default_adapter_registry",
]
