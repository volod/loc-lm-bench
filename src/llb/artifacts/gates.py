"""Compatibility gates run before an artifact is acted on rather than after.

A store build reads a corpus and publishes a generation fingerprinted against it; a review session
opens a ledger a person then decides in. Both are expensive and both are hard to undo, so the
question "can this build read what it is about to act on" is asked once, at the door. A record
from a FUTURE major is the case that matters: it validates as JSON, it looks like the family it
names, and every field a newer writer added is silently invisible to this reader.
"""

import json
import logging
from pathlib import Path

from llb.artifacts.default_registry import DEFAULT_REGISTRY
from llb.artifacts.errors import ArtifactContractError, MissingIdentityError
from llb.core.contracts.common import JsonObject

_LOG = logging.getLogger(__name__)

CORPUS_MANIFEST_SCHEMA_ID = "llb.corpus-manifest"
GOLD_ITEM_SCHEMA_ID = "llb.gold-item"
ONTOLOGY_PROVENANCE_SCHEMA_ID = "llb.ontology-provenance"
WORKSHEET_ROW_SCHEMA_ID = "llb.verification-worksheet-row"


class ArtifactCompatibilityError(ArtifactContractError):
    """A named artifact this build cannot read stands between an operator and their next step."""


def refuse_unreadable_document(path: Path, schema_id: str) -> None:
    """Refuse a JSON document of a known family this build cannot read.

    An absent file is not a refusal: many members are optional, and the caller that requires one
    says so itself. A present file that cannot resolve is, because the alternative is acting on a
    record whose newer half this reader cannot see.
    """
    if not path.is_file():
        return
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(f"{path}: cannot read {schema_id} record: {exc}") from exc
    if not isinstance(record, dict):
        raise ArtifactCompatibilityError(f"{path}: {schema_id} record must be an object")
    _resolve(record, schema_id, path)


def refuse_unreadable_corpus(corpus_root: Path | str) -> None:
    """The gate a store build passes: the corpus manifest and the applied conflict overlay.

    Both are folded into the store's corpus fingerprint, so a generation built from a manifest or
    an overlay this build only half understands is a generation nobody can reproduce.
    """
    from llb.conflicts.resolution.overlay import applied_overlay_path, load_applied_overlay
    from llb.prep.corpus.fingerprints import CORPUS_MANIFEST

    root = Path(corpus_root)
    refuse_unreadable_document(root / CORPUS_MANIFEST, CORPUS_MANIFEST_SCHEMA_ID)
    overlay_path = applied_overlay_path(root)
    if not overlay_path.is_file():
        return
    try:
        load_applied_overlay(root)
    except (ArtifactContractError, ValueError) as exc:
        raise ArtifactCompatibilityError(f"{overlay_path}: {exc}") from exc


def refuse_unreadable_review(path: Path | str) -> None:
    """The gate a review session passes before a person is shown anything to decide.

    Only the members whose family is unambiguous from the path are checked; the review registry's
    own signature detection stays the authority on WHICH ledger this is.
    """
    target = Path(path)
    bundle = target if target.is_dir() else target.parent
    refuse_unreadable_document(bundle / "provenance.json", ONTOLOGY_PROVENANCE_SCHEMA_ID)


def _resolve(record: JsonObject, schema_id: str, path: Path) -> None:
    try:
        DEFAULT_REGISTRY.resolve(_identified(record, schema_id), source=str(path))
    except MissingIdentityError:
        _LOG.debug("[artifacts] %s carries no identity and no legacy version is declared", path)
    except ArtifactContractError as exc:
        raise ArtifactCompatibilityError(str(exc)) from exc


def _identified(record: JsonObject, schema_id: str) -> JsonObject:
    """The record with the identity a pre-contract file leaves for its reader to supply."""
    if record.get("schema_id") is not None:
        return record
    legacy = DEFAULT_REGISTRY.definition(schema_id).legacy_version
    if legacy is None:
        return record
    return {**record, "schema_id": schema_id, "schema_version": legacy}
