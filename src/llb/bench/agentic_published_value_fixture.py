"""The committed evidence a published agentic number is resolved against, and its integrity rules.

A published value is only as checkable as the file it cites. The repo therefore carries the cited
run aggregates THEMSELVES -- one verbatim copy per artifact under `agentic_published_aggregates/`,
mirroring its DATA_DIR-relative path -- plus a manifest pinning each copy by content digest.

That is the whole point of the copy. A digest alone pins a file only on a host that HAS the file: on
CI, a fresh clone, or this host after a `.data` cleanup, a digest recorded beside a hand-cut summary
is a claim about a file nobody present can read, so a summary and a pin fabricated together were
accepted there. With the bytes committed, the pin is verified on EVERY host against the copy that
carries them, the published value is read out of those bytes rather than out of a hand-shaped
excerpt, and a host that still has the run root falsifies the copy against the run.

What this does not claim to do is stop a determined fabricator: a hand-written aggregate re-pinned
to its own digest passes on a host with no run root, because nothing there can contradict it. What
it removes is the gap between the evidence and the claim -- the bytes are present and reviewable as
an ordinary diff, so fabricating a published number now means fabricating a whole analysis file in
the open rather than a hash of an absent one.

A signed manifest was the alternative. It was rejected: a key committed beside the manifest signs
whatever the fabricator wants, and a key held outside the repo authenticates the operator who ran
the regeneration rather than the run, while adding key distribution to a check that must work in a
fresh clone with nothing configured.

Growth is bounded rather than unlimited, since these aggregates accumulate with every study that
adopts the seam: a per-artifact byte cap, a total budget across the committed evidence, and a
regeneration that PRUNES every copy no published value still cites -- so the cost tracks the number
of CITED artifacts, not the number of runs.

The tree is SHARED, so "still cites" is the union over every design that publishes resolvable
values, not the citations of whichever design a refresh was handed. That registry and the refresh
that walks it live in `llb.bench.agentic_published_value_registry`; this module refuses to write a
manifest that omits an artifact the registry says is cited, so the prune here can never retire a
registered study's evidence.
"""

from collections.abc import Collection, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import cast

# The manifest, project-root relative: one entry per cited artifact, pinning its committed copy.
PROVENANCE_FIXTURE = Path("samples/benchmarks/agentic_published_crossover_aggregates.json")
# The committed copies themselves, each at the manifest key's own DATA_DIR-relative path.
COMMITTED_AGGREGATE_DIR = Path("samples/benchmarks/agentic_published_aggregates")
FIXTURE_SCHEMA_VERSION = 3

# The artifact pin: sha256 over the aggregate's bytes as written, algorithm-prefixed so a later
# change of algorithm is a visible fixture change rather than a silently re-read hex string.
DIGEST_ALGORITHM = "sha256"
_DIGEST = re.compile(rf"^{DIGEST_ALGORITHM}:[0-9a-f]{{64}}$")

# The growth policy. These aggregates are per-study JSON analyses of a completed run, so one is a
# few tens of KiB; the caps are the point at which committing the evidence stops being the cheap
# option and the study should cite a narrower analysis artifact instead.
MAX_AGGREGATE_BYTES = 256 * 1024
MAX_COMMITTED_BYTES = 2 * 1024 * 1024

REGENERATE = "regenerate it with make bench-agentic-published-provenance"


@dataclass(frozen=True)
class CommittedAggregate:
    """One cited run aggregate as the repo carries it: the pin, and the bytes it pins, parsed."""

    digest: str
    payload: dict[str, object]


def artifact_digest(path: Path) -> str:
    """The content digest that pins one run aggregate, computed over its bytes as written."""
    return digest_bytes(path.read_bytes())


def digest_bytes(raw: bytes) -> str:
    """The same pin over bytes in hand, so a copy and its digest come out of one read."""
    return f"{DIGEST_ALGORITHM}:{hashlib.sha256(raw).hexdigest()}"


def is_contained_artifact(artifact: str) -> bool:
    """Whether an artifact key stays inside the tree it addresses, in DATA_DIR and in the repo."""
    return bool(artifact) and not Path(artifact).is_absolute() and ".." not in Path(artifact).parts


def committed_aggregate_path(root: Path, artifact: str) -> Path:
    """Where the repo carries its verbatim copy of one cited artifact."""
    if not is_contained_artifact(artifact):
        raise ValueError(
            f"{PROVENANCE_FIXTURE}: the artifact key {artifact!r} escapes "
            f"{COMMITTED_AGGREGATE_DIR}, so no committed copy can address it"
        )
    return root / COMMITTED_AGGREGATE_DIR / artifact


def committed_evidence_bytes(root: Path) -> int:
    """Everything the repo carries under the committed-aggregate tree, for the growth budget."""
    directory = root / COMMITTED_AGGREGATE_DIR
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def load_provenance_fixture(root: Path) -> dict[str, CommittedAggregate]:
    """The committed aggregates, each verified against the pin the manifest records for it."""
    path = root / PROVENANCE_FIXTURE
    if not path.is_file():
        raise ValueError(f"the committed provenance fixture {PROVENANCE_FIXTURE} does not exist")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != FIXTURE_SCHEMA_VERSION:
        raise ValueError(
            f"{PROVENANCE_FIXTURE} must declare schema_version {FIXTURE_SCHEMA_VERSION}"
        )
    entries = cast(dict[str, dict[str, object]], payload["aggregates"])
    return {artifact: _committed(root, artifact, entry) for artifact, entry in entries.items()}


def write_provenance_fixture(
    root: Path,
    aggregates: dict[str, bytes],
    *,
    cited_by: Mapping[str, Collection[str]] | None = None,
) -> Path:
    """Commit each cited aggregate verbatim, pin it, and drop every copy nothing cites any more.

    Bytes in, never a re-serialized payload: a copy written from a parsed object would digest to
    something the run artifact never produced, which is the one way a pin and the file it pins could
    disagree by construction.

    `cited_by` is the registry's whole citation map (artifact -> the designs that cite it). Passing
    it makes this write a UNION rather than one caller's view of the tree; leaving it out states
    that the caller speaks for no registry, which is what a synthetic set in a test does.
    """
    _refuse_partial_union(aggregates, cited_by)
    _check_growth_budget(aggregates)
    for artifact, raw in aggregates.items():
        committed = committed_aggregate_path(root, artifact)
        committed.parent.mkdir(parents=True, exist_ok=True)
        committed.write_bytes(raw)
    _prune_uncited_copies(root, set(aggregates))
    path = root / PROVENANCE_FIXTURE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_manifest(aggregates), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _manifest(aggregates: dict[str, bytes]) -> dict[str, object]:
    """The manifest as written, sorted so a regeneration diffs to what actually moved."""
    return {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "note": (
            f"content pins for the run aggregates the published agentic numbers were measured in; "
            f"each pinned file is carried verbatim under {COMMITTED_AGGREGATE_DIR.name}/ at the "
            "same DATA_DIR-relative path, so the pin is checkable on a host that never ran the "
            "study; regenerate with make bench-agentic-published-provenance rather than by hand"
        ),
        "aggregates": {
            artifact: {"digest": digest_bytes(aggregates[artifact])}
            for artifact in sorted(aggregates)
        },
    }


def _refuse_partial_union(
    aggregates: dict[str, bytes], cited_by: Mapping[str, Collection[str]] | None
) -> None:
    """The manifest is the union of every registered design's citations, or it is not written.

    The prune below is what makes this worth refusing over: it deletes every copy the incoming set
    omits, and the deletion reads as a clean prune rather than as evidence loss. So a caller holding
    ONE design's citations must not be able to retire another registered study's aggregates by
    regenerating -- the whole set is offered, or nothing is written.
    """
    if cited_by is None:
        return
    dropped = sorted(artifact for artifact in cited_by if artifact not in aggregates)
    if not dropped:
        return
    named = "; ".join(
        f"{artifact} (cited by {', '.join(sorted(cited_by[artifact]))})" for artifact in dropped
    )
    raise ValueError(
        f"{PROVENANCE_FIXTURE}: regenerating from {len(aggregates)} aggregates would drop the "
        f"committed copy of {named} -- a registered design still resolves published values through "
        "it, so the refresh must offer the union over every registered design rather than one "
        "design's citations"
    )


def _check_growth_budget(aggregates: dict[str, bytes]) -> None:
    """Refuse evidence the repo should not carry, naming the cap rather than growing past it."""
    for artifact, raw in sorted(aggregates.items()):
        if len(raw) > MAX_AGGREGATE_BYTES:
            raise ValueError(
                f"the run aggregate {artifact} is {len(raw)} bytes, over the "
                f"{MAX_AGGREGATE_BYTES}-byte per-artifact cap on committed evidence -- publish the "
                "value out of a narrower analysis artifact of that run rather than committing this "
                "one"
            )
    total = sum(len(raw) for raw in aggregates.values())
    if total > MAX_COMMITTED_BYTES:
        raise ValueError(
            f"the {len(aggregates)} cited aggregates total {total} bytes, over the "
            f"{MAX_COMMITTED_BYTES}-byte budget for committed provenance evidence -- retire a "
            "study's published values or cite narrower analysis artifacts"
        )


def _prune_uncited_copies(root: Path, cited: set[str]) -> None:
    """Drop the copies of artifacts no published value cites, so the tree tracks the citations."""
    directory = root / COMMITTED_AGGREGATE_DIR
    if not directory.is_dir():
        return
    kept = {committed_aggregate_path(root, artifact) for artifact in cited}
    for path in sorted(directory.rglob("*"), reverse=True):
        if path.is_file() and path not in kept:
            path.unlink()
        elif path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def _committed(root: Path, artifact: str, entry: object) -> CommittedAggregate:
    """One manifest entry, read back out of the bytes the repo carries for it."""
    digest = _pin(artifact, entry)
    path = committed_aggregate_path(root, artifact)
    if not path.is_file():
        raise ValueError(
            f"{PROVENANCE_FIXTURE}: the manifest pins {artifact!r} at {digest}, but the repo "
            f"carries no copy of it under {COMMITTED_AGGREGATE_DIR}, so on a host without that run "
            f"the pin is a claim about a file nobody present can read -- {REGENERATE}"
        )
    found = artifact_digest(path)
    if found != digest:
        raise ValueError(
            f"{PROVENANCE_FIXTURE}: the manifest pins {artifact!r} at {digest} while the committed "
            f"copy digests to {found} -- one of the two was edited by hand, so the committed "
            f"evidence no longer stands for the run it names; {REGENERATE}"
        )
    return CommittedAggregate(
        digest=digest, payload=cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    )


def _pin(artifact: str, entry: object) -> str:
    """The recorded pin, refusing an entry that pins nothing rather than reading the copy anyway."""
    digest = entry.get("digest") if isinstance(entry, dict) else None
    if digest is None:
        raise ValueError(
            f"{PROVENANCE_FIXTURE}: the entry for {artifact!r} records no content digest of the "
            f"aggregate it stands for, so nothing ties the committed copy to the cited run -- "
            f"{REGENERATE}"
        )
    if not isinstance(digest, str) or not _DIGEST.match(digest):
        raise ValueError(
            f"{PROVENANCE_FIXTURE}: the entry for {artifact!r} must pin its aggregate with a "
            f"{DIGEST_ALGORITHM}:<hex> content digest, got {digest!r}"
        )
    return digest
