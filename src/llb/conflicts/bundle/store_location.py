"""Portable location of the store a conflict audit read.

The store identity says *which* store supplied a bundle's semantic inputs. This module records
where that immutable store directory sat below ``DATA_DIR`` so a later stage replay can find it on
the same host, or at the same data-relative layout on another host, without persisting a
host-specific absolute path.

Locations outside ``DATA_DIR`` are deliberately omitted. They remain addressable through the
replay command's explicit ``--store`` fallback, while a bundle never leaks or hardcodes an
operator-specific absolute directory.
"""

from dataclasses import dataclass
from pathlib import Path

from llb.core.contracts.common import JsonObject
from llb.core.paths import resolve_data_dir

STORE_DATA_DIR_RELATIVE_KEY = "store_data_dir_relative"
StoreFingerprintCache = dict[tuple[str, str], dict[str, str] | None]


@dataclass(frozen=True)
class RecordedStoreLocation:
    """A validated DATA_DIR-relative store reference and its resolved local path."""

    relative: str
    path: Path

    @property
    def display(self) -> str:
        """A portable operator-facing spelling that never exposes the host's data root."""
        return "$DATA_DIR" if self.relative == "." else f"$DATA_DIR/{self.relative}"


def store_location_payload(
    store_dir: Path | str | None, *, data_dir: Path | str | None = None
) -> JsonObject:
    """Record ``store_dir`` relative to DATA_DIR, or nothing when it is outside that root."""
    if store_dir is None:
        return {}
    root = resolve_data_dir(data_dir).resolve()
    resolved = Path(store_dir).expanduser().resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        return {}
    return {STORE_DATA_DIR_RELATIVE_KEY: relative.as_posix()}


def recorded_store_location(
    tree_meta: JsonObject, *, data_dir: Path | str | None = None
) -> RecordedStoreLocation | None:
    """Resolve a bundle's validated DATA_DIR-relative location, or None when it has none.

    A malformed or escaping path is an invalid bundle field, not an absent one. Refusing it keeps
    an untrusted bundle from turning a replay into an arbitrary filesystem read.
    """
    raw = tree_meta.get(STORE_DATA_DIR_RELATIVE_KEY)
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{STORE_DATA_DIR_RELATIVE_KEY} must be a non-empty relative path")
    relative = Path(raw)
    if relative.is_absolute():
        raise ValueError(f"{STORE_DATA_DIR_RELATIVE_KEY} must not be absolute")
    root = resolve_data_dir(data_dir).resolve()
    resolved = (root / relative).resolve()
    try:
        normalized = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{STORE_DATA_DIR_RELATIVE_KEY} escapes DATA_DIR") from exc
    return RecordedStoreLocation(relative=normalized.as_posix(), path=resolved)


def resolve_store_placement(
    summary: JsonObject,
    *,
    fallback_store: Path | None,
    data_dir: Path | None,
    cache: StoreFingerprintCache | None = None,
) -> JsonObject | None:
    """Place a bundle against an explicit store, or its recorded store when no flag is supplied.

    The explicit store deliberately wins for compatibility and for operator-directed comparison.
    Without it, a recorded reference is resolved exactly. An invalid or gone recorded reference
    produces a non-comparable reading rather than being mistaken for an identity mismatch.
    """
    from llb.conflicts.bundle.store_identity import StoreChange, identity_entry
    from llb.conflicts.store_access import store_doc_fingerprints, store_doc_fingerprints_at

    resolved_cache = cache if cache is not None else {}
    if fallback_store is not None:
        key = ("explicit", str(fallback_store.resolve()))
        if key not in resolved_cache:
            resolved_cache[key] = store_doc_fingerprints(fallback_store)
        fingerprints = resolved_cache[key]
        assert fingerprints is not None
        return identity_entry(
            summary,
            fingerprints,
            label="the explicit `--store`",
            reference="--store",
            reference_source="explicit",
        )

    tree_meta = summary.get("tree")
    tree = tree_meta if isinstance(tree_meta, dict) else {}
    try:
        recorded = recorded_store_location(tree, data_dir=data_dir)
    except ValueError as exc:
        payload = StoreChange(
            comparable=False,
            changed=False,
            detail=f"invalid recorded store location: {exc}; no identity comparison was made",
        ).payload()
        payload["reference_source"] = "bundle"
        return payload

    if recorded is not None:
        key = ("recorded", str(recorded.path))
        if key not in resolved_cache:
            resolved_cache[key] = store_doc_fingerprints_at(recorded.path)
        fingerprints = resolved_cache[key]
        if fingerprints is None:
            payload = StoreChange(
                comparable=False,
                changed=False,
                detail=(
                    f"recorded store location `{recorded.display}` is gone: no store metadata "
                    "exists there; no identity comparison was made"
                ),
            ).payload()
            payload.update({"reference": recorded.display, "reference_source": "bundle"})
            return payload
        return identity_entry(
            summary,
            fingerprints,
            label=f"the recorded store at `{recorded.display}`",
            reference=recorded.display,
            reference_source="bundle",
        )

    return None
