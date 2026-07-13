"""User-facing adapter lookup."""

from pathlib import Path

from llb.core.paths import resolve_project_path
from llb.finetune.registry.io import load_registry
from llb.finetune.registry.model import MIN_ADAPTER_ID_PREFIX, AdapterEntry


def resolve_adapter(entries: dict[str, AdapterEntry], adapter: str) -> AdapterEntry:
    """Resolve an adapter id, unique id prefix, label, or directory."""
    wanted = str(adapter).strip()
    if wanted in entries:
        return entries[wanted]
    by_label = [entry for entry in entries.values() if entry.adapter_label == wanted]
    if len(by_label) == 1:
        return by_label[0]
    by_dir = [entry for entry in entries.values() if _same_dir(entry, wanted)]
    if len(by_dir) == 1:
        return by_dir[0]
    if len(wanted) >= MIN_ADAPTER_ID_PREFIX:
        prefixed = [entry for entry in entries.values() if entry.adapter_id.startswith(wanted)]
        if len(prefixed) == 1:
            return prefixed[0]
        if len(prefixed) > 1:
            raise ValueError(
                f"adapter prefix {wanted!r} is ambiguous: "
                f"{', '.join(sorted(entry.short_id for entry in prefixed))}"
            )
    raise ValueError(
        f"adapter {wanted!r} is not registered; `llb list-adapters` shows the registered ids"
    )


def find_by_digest(registry: Path | str, adapter_digest: str) -> AdapterEntry | None:
    """Return the registry entry for a trained digest, if one exists."""
    return load_registry(registry).get(str(adapter_digest))


def _same_dir(entry: AdapterEntry, candidate: str) -> bool:
    try:
        return entry.resolved_dir == resolve_project_path(candidate)
    except (OSError, ValueError):
        return False
