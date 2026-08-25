"""Load an axiom set from Turtle or from its typed JSON mirror, and write both together.

Turtle is the SOURCE: it is the form a domain reviewer reads and signs, and the form a diff shows
one constraint per block. The JSON beside it is the same set through the typed models, so a
consumer that only wants the constraint list does not re-parse Turtle, and so `make ci` can prove
the two never drifted. Writing them apart is what would let them disagree, so `save_axioms`
writes both from one set or neither.
"""

import json
from pathlib import Path

from llb.core.fsutil import atomic_write_text
from llb.prep.ontology.axioms.constants import CANDIDATE_AXIOMS_JSON, CANDIDATE_AXIOMS_TURTLE
from llb.prep.ontology.axioms.deserialize import load_turtle
from llb.prep.ontology.axioms.models import AxiomSet
from llb.prep.ontology.axioms.serialize import dump_turtle

TURTLE_SUFFIX = ".ttl"
JSON_SUFFIX = ".json"


def json_path_for(turtle_path: Path) -> Path:
    """The typed mirror that lives beside a Turtle axiom file."""
    return Path(turtle_path).with_suffix(JSON_SUFFIX)


def read_header(text: str) -> list[str]:
    """The leading comment block of a Turtle document, with its `#` markers stripped.

    The header is prose about the SET (its version, its sign-off status), which no triple carries,
    so a regenerated file has to be handed the same header to come back byte-identical.
    """
    header: list[str] = []
    for line in text.splitlines():
        if not line.startswith("#"):
            break
        header.append(line[2:] if line.startswith("# ") else line[1:])
    return header


def load_axioms(path: Path | str) -> AxiomSet:
    """Read an axiom set from a `.ttl` (parsed) or a `.json` (validated) file."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"axiom file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix == JSON_SUFFIX:
        return AxiomSet.model_validate(json.loads(text))
    return load_turtle(text)


def save_axioms(
    axiom_set: AxiomSet, turtle_path: Path | str, header: list[str]
) -> tuple[Path, Path]:
    """Write the Turtle document and its JSON mirror; returns both paths."""
    turtle_path = Path(turtle_path)
    turtle_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(turtle_path, dump_turtle(axiom_set, header))
    json_target = json_path_for(turtle_path)
    payload = json.dumps(axiom_set.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
    atomic_write_text(json_target, payload)
    return turtle_path, json_target


def candidate_paths(samples_root: Path | str) -> tuple[Path, Path]:
    """The committed candidate set's two paths under a `samples/ontology` directory."""
    root = Path(samples_root)
    return root / CANDIDATE_AXIOMS_TURTLE, root / CANDIDATE_AXIOMS_JSON
