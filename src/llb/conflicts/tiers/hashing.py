"""Deterministic hashing primitives shared by the hash and lexical tiers.

Everything here is stable across processes and platforms: `hashlib` rather than Python's salted
`hash()`, so a shingle computed today matches one computed on another host tomorrow.
"""

import hashlib
import json

from llb.core.contracts.common import JsonObject

CONTENT_HASH_HEX_LENGTH = 16


def finding_id(finding: JsonObject) -> str:
    """Stable identity for one finding, independent of JSON object key order.

    It lives here rather than beside the resolution policy because the audit's decision-group
    sidecar and the resolution plan must address the same row by the same id.
    """
    encoded = json.dumps(finding, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:CONTENT_HASH_HEX_LENGTH]


def sha256_text(text: str) -> str:
    """Hex sha256 of `text` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_hash64(value: str) -> int:
    """A stable unsigned 64-bit hash of `value` (blake2b truncated)."""
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")
