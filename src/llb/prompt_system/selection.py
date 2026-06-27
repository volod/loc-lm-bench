"""Load a reviewed prompt-system package for benchmark runs.

The prepare/review CLI writes one `candidates.json` plus a run manifest. `run-eval` only needs the
selected candidate's rendered prompts and the provenance block that makes the score traceable.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from llb.prompt_system.manifest import PromptSystemProvenance, template_digest
from llb.prompt_system.pipeline import CANDIDATES_FILE, MANIFEST_FILE, METHOD
from llb.prompt_system.review import PromptCandidate, load_candidates
from llb.prompt_system.template import PromptPackage


@dataclass(slots=True)
class SelectedPromptPackage:
    """A selected prompt package plus manifest-ready provenance."""

    package: PromptPackage
    provenance: PromptSystemProvenance
    run_dir: Path


def resolve_prompt_package(
    data_dir: Path | str,
    prompt_system_id: str,
    prompt_package: Path | str | None = None,
) -> SelectedPromptPackage:
    """Find `prompt_system_id` in a supplied run dir/file or under `$DATA_DIR/prompt-system/*`."""
    for run_dir, candidates_path in _candidate_locations(
        data_dir, prompt_system_id, prompt_package
    ):
        candidates = load_candidates(candidates_path)
        candidate = next((c for c in candidates if c.prompt_system_id == prompt_system_id), None)
        if candidate is None:
            continue
        return SelectedPromptPackage(
            package=candidate.package(),
            provenance=_provenance(run_dir, candidate),
            run_dir=run_dir,
        )
    where = str(prompt_package) if prompt_package is not None else str(Path(data_dir) / METHOD)
    raise FileNotFoundError(f"prompt system {prompt_system_id!r} not found under {where}")


def prompt_system_id_from_package_path(path: Path | str | None) -> str | None:
    """Support `--prompt-package <run_dir>/<id>` as a compact selector."""
    if path is None:
        return None
    value = Path(path)
    if value.exists():
        return None
    return value.name if value.parent.exists() else None


def _candidate_locations(
    data_dir: Path | str,
    prompt_system_id: str,
    prompt_package: Path | str | None,
) -> list[tuple[Path, Path]]:
    if prompt_package is not None:
        value = Path(prompt_package)
        if value.is_file():
            return [(value.parent, value)]
        if value.is_dir():
            return [(value, value / CANDIDATES_FILE)]
        if value.parent.is_dir() and value.name == prompt_system_id:
            return [(value.parent, value.parent / CANDIDATES_FILE)]
        return [(value, value / CANDIDATES_FILE)]
    root = Path(data_dir) / METHOD
    return [(path.parent, path) for path in sorted(root.glob(f"*/{CANDIDATES_FILE}"), reverse=True)]


def _provenance(run_dir: Path, candidate: PromptCandidate) -> PromptSystemProvenance:
    manifest = _read_manifest(run_dir)
    return {
        "prompt_system_id": candidate.prompt_system_id,
        "corpus_digest": str(manifest["corpus_digest"]),
        "mapping_digest": str(manifest["mapping_digest"]),
        "template_revision": template_digest(candidate.fields),
        "tokenizer": str(manifest["tokenizer"]),
        "context_window": _int_value(manifest["context_window"]),
        "prompt_budget_tokens": _int_value(manifest["prompt_budget_tokens"]),
    }


def _read_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / MANIFEST_FILE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load prompt-system manifest: {path}") from exc
    required = (
        "corpus_digest",
        "mapping_digest",
        "tokenizer",
        "context_window",
        "prompt_budget_tokens",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"prompt-system manifest missing keys {missing}: {path}")
    return cast(dict[str, Any], payload)


def _int_value(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"expected int-compatible value, got {type(value).__name__}")
