"""Write and read the machine-readable sidecar of a retrieval comparison.

Every comparison command prints a Markdown report for a person and writes a JSON sidecar for
everything else -- a board, a later sweep, an external analysis. The sidecar is wrapped in its
registered envelope here, in one place, so each command says only which KIND of reading it took
and which command took it, and a reader can tell a graph-fusion sweep from an embedding bake-off
without opening the body.

The envelope is versioned; the body is not modelled. See `llb.core.contracts.retrieval.comparison`
for why that boundary is where it is.
"""

import json
from pathlib import Path
from typing import Any

from llb.artifacts.records import decode, encode
from llb.core.contracts.retrieval.comparison import (
    RETRIEVAL_COMPARISON_SCHEMA_ID,
    SIDECAR_KINDS,
)

SIDECAR_CONTRACT_VERSION = "1.0.0"


def write_sidecar(path: Path | str, kind: str, produced_by: str, report: Any) -> Path:
    """Write one comparison sidecar with its identity, kind, and producing command."""
    if kind not in SIDECAR_KINDS:
        raise ValueError(
            f"unknown comparison sidecar kind {kind!r}; expected one of {SIDECAR_KINDS}"
        )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = encode(
        RETRIEVAL_COMPARISON_SCHEMA_ID,
        SIDECAR_CONTRACT_VERSION,
        {"kind": kind, "produced_by": produced_by, "report": report},
    )
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def read_sidecar(path: Path | str) -> dict[str, Any]:
    """The sidecar's envelope and body at the current contract, current or pre-contract.

    A sidecar written before the family was registered is the bare body, with nowhere to carry an
    identity or say what produced it. The family declares `report` as the field such a file became,
    so the body is wrapped there and the envelope's own defaults record the producer as unstated --
    an archived comparison stays a readable record rather than an unlabelled blob.
    """
    target = Path(path)
    raw = json.loads(target.read_text(encoding="utf-8"))
    return decode(RETRIEVAL_COMPARISON_SCHEMA_ID, raw, source=str(target))


def sidecar_report(path: Path | str) -> dict[str, Any]:
    """Just the measurement body of a sidecar."""
    report = read_sidecar(path).get("report")
    return report if isinstance(report, dict) else {}
