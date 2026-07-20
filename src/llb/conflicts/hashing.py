"""Deterministic hashing primitives shared by the hash and lexical tiers.

Everything here is stable across processes and platforms: `hashlib` rather than Python's salted
`hash()`, so a shingle computed today matches one computed on another host tomorrow.
"""

import hashlib


def sha256_text(text: str) -> str:
    """Hex sha256 of `text` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_hash64(value: str) -> int:
    """A stable unsigned 64-bit hash of `value` (blake2b truncated)."""
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")
