"""A re-pinned aggregate must still derive each cited field from its own recorded evidence."""

from copy import deepcopy
import json
from pathlib import Path
from typing import cast

import pytest

from llb.bench.memory.crossover_restatement.design import (
    load_restatement_design,
    published_crossovers,
)
from llb.bench.memory.crossover_restatement.reading import (
    FORM_FOLD_STEP,
    FORM_INTERPOLATED,
    FORM_PORTABLE_RATIO,
)
from llb.bench.published_value.fixture import (
    committed_aggregate_path,
    write_provenance_fixture,
)
from llb.bench.published_value.provenance import PublishedValueResolver, provenance_pair

ROOT = Path(__file__).resolve().parents[4]
DESIGN = ROOT / "samples/benchmarks/agentic_compact_crossover_restatement_design.json"


def _published(form: str) -> dict[str, object]:
    return next(
        row
        for row in published_crossovers(load_restatement_design(DESIGN))
        if row["form"] == form and row["depth"] == 6
    )


def _payload(row: dict[str, object]) -> tuple[str, str, dict[str, object]]:
    artifact, field = provenance_pair(row["provenance"], where="test")
    payload = json.loads(committed_aggregate_path(ROOT, artifact).read_text(encoding="utf-8"))
    return artifact, field, cast(dict[str, object], payload)


def _change_surface(payload: dict[str, object]) -> None:
    row = next(
        row for row in cast(list[dict[str, object]], payload["depth_surface"]) if row["depth"] == 6
    )
    row["crossover_max_prompt_chars"] = float(cast(float, row["crossover_max_prompt_chars"])) + 1.0


def _change_fold_boundary(payload: dict[str, object]) -> None:
    row = next(
        row for row in cast(list[dict[str, object]], payload["depth_ladders"]) if row["depth"] == 6
    )
    boundary = cast(dict[str, object], row["boundary"])
    boundary["guard_boundary_chars"] = int(cast(int, boundary["guard_boundary_chars"])) + 1


def _change_cap_peak(payload: dict[str, object]) -> None:
    peaks = cast(dict[str, int], payload["cap_peak_prompt_chars"])
    peaks["6"] += 1


@pytest.mark.parametrize(
    ("form", "mutate"),
    [
        (FORM_INTERPOLATED, _change_surface),
        (FORM_FOLD_STEP, _change_fold_boundary),
        (FORM_PORTABLE_RATIO, _change_cap_peak),
    ],
)
def test_a_hand_written_field_re_pinned_with_its_aggregate_is_refused(tmp_path, form, mutate):
    """A fresh digest authenticates the edited bytes; their own cells must still produce the edit."""
    row = _published(form)
    artifact, _field, payload = _payload(row)
    mutate(payload)
    write_provenance_fixture(tmp_path, {artifact: json.dumps(payload).encode("utf-8")})

    with pytest.raises(ValueError, match="committed aggregate is internally inconsistent"):
        PublishedValueResolver(root=tmp_path).resolve(row["provenance"], where="test")


def test_a_field_with_no_source_cells_is_refused_instead_of_taken_from_the_aggregate(tmp_path):
    row = _published(FORM_INTERPOLATED)
    artifact, _field, payload = _payload(row)
    payload["cells"] = [
        cell for cell in cast(list[dict[str, object]], payload["cells"]) if cell["depth"] != 6
    ]
    write_provenance_fixture(tmp_path, {artifact: json.dumps(payload).encode("utf-8")})

    with pytest.raises(ValueError, match="records no depth 6 cells"):
        PublishedValueResolver(root=tmp_path).resolve(row["provenance"], where="test")


def test_an_unregistered_numeric_field_is_not_resolvable_by_default(tmp_path):
    row = deepcopy(_published(FORM_INTERPOLATED))
    artifact, _field, payload = _payload(row)
    payload["invented_published_value"] = 17
    write_provenance_fixture(tmp_path, {artifact: json.dumps(payload).encode("utf-8")})
    row["provenance"] = {"artifact": artifact, "field": "invented_published_value"}

    with pytest.raises(ValueError, match="has no registered aggregate-internal derivation"):
        PublishedValueResolver(root=tmp_path).resolve(row["provenance"], where="test")
