"""The published family tables are generated, so the only thing worth testing is the disagreement.

The shipped-tree assertion is the point: README and the reference page must still say what the
roster manifest says, and nothing else in the toolchain notices when they part. The synthetic cases
pin each way the register itself can be wrong -- two current generations, a model on a generation
nobody declared, a license that disagrees with its generation -- because each of those reads as
"fine" to someone skimming the YAML.
"""

from pathlib import Path

import pytest

from llb.backends.roster import build_register, load_register, register_findings
from llb.core.contracts.models import FamilySpec, ModelSpec
from llb.core.paths import PROJECT_ROOT
from llb.quality.roster_docs import DOC_BLOCKS, ROSTER_MANIFEST, render_block, sync_findings


def _family(**overrides: object) -> FamilySpec:
    family: dict[str, object] = {
        "id": "qwen",
        "label": "Qwen (Alibaba)",
        "role": "multilingual-baseline",
        "focus": "Multilingual baseline",
        "generations": [
            {
                "id": "3.8",
                "status": "current",
                "label": "Qwen3.8",
                "license": "Apache-2.0",
                "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
                "weights_url": "https://huggingface.co/Qwen/Qwen3.8-27B",
            }
        ],
    }
    family.update(overrides)
    return family  # type: ignore[return-value]


def _model(name: str, generation: str, license_name: str = "Apache-2.0") -> ModelSpec:
    return {  # type: ignore[return-value]
        "name": name,
        "backend": "ollama",
        "source": f"{name}:tag",
        "family": "qwen",
        "generation": generation,
        "license": license_name,
    }


def test_committed_roster_register_is_consistent() -> None:
    register = load_register(PROJECT_ROOT / ROSTER_MANIFEST)

    assert register_findings(register) == []
    assert {family.id for family in register.families} >= {"qwen", "gemma", "mamaylm"}


def test_qwen_carries_the_current_generation_and_the_one_it_replaced() -> None:
    register = load_register(PROJECT_ROOT / ROSTER_MANIFEST)

    qwen = register.family("qwen")
    assert qwen is not None
    current = qwen.current
    assert current is not None and current.id == "3.8"
    assert "qwen3.8-27b" in current.model_names
    assert "3.6" in {generation.id for generation in qwen.previous}


def test_published_blocks_match_the_committed_manifest() -> None:
    register = load_register(PROJECT_ROOT / ROSTER_MANIFEST)

    assert sync_findings(register, root=PROJECT_ROOT) == []


@pytest.mark.parametrize("block", [block.name for block in DOC_BLOCKS])
def test_every_block_names_the_current_qwen_generation(block: str) -> None:
    register = load_register(PROJECT_ROOT / ROSTER_MANIFEST)

    rendered = render_block(block, register)

    assert "Qwen3.8" in rendered


def test_a_stale_block_is_reported_then_rewritten(tmp_path: Path) -> None:
    for doc in {block.doc for block in DOC_BLOCKS}:
        target = tmp_path / doc
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((PROJECT_ROOT / doc).read_text(encoding="utf-8"), encoding="utf-8")
    stale = tmp_path / DOC_BLOCKS[0].doc
    stale.write_text(
        stale.read_text(encoding="utf-8").replace("Qwen3.8", "Qwen3.6"), encoding="utf-8"
    )
    register = load_register(PROJECT_ROOT / ROSTER_MANIFEST)

    reported = sync_findings(register, root=tmp_path)
    rewritten = sync_findings(register, root=tmp_path, write=True)

    assert len(reported) == 1 and "stale" in reported[0]
    assert rewritten == []
    assert sync_findings(register, root=tmp_path) == []


def test_two_current_generations_are_a_finding() -> None:
    generations = [
        {"id": "3.8", "status": "current", "license": "Apache-2.0", "license_url": "https://x"},
        {"id": "3.6", "status": "current", "license": "Apache-2.0", "license_url": "https://x"},
    ]
    register = build_register(
        [_family(generations=generations)], [_model("a", "3.8"), _model("b", "3.6")]
    )

    findings = register_findings(register)

    assert any("2 generation(s) marked `current`" in finding for finding in findings)


def test_a_model_on_an_undeclared_generation_is_a_finding() -> None:
    register = build_register([_family()], [_model("a", "3.8"), _model("b", "3.9")])

    findings = register_findings(register)

    assert any("generation `3.9` is not declared" in finding for finding in findings)


def test_a_generation_nobody_carries_is_a_finding() -> None:
    register = build_register([_family()], [])

    findings = register_findings(register)

    assert any("no model carries it" in finding for finding in findings)


def test_a_model_license_that_disagrees_with_its_generation_is_a_finding() -> None:
    register = build_register([_family()], [_model("a", "3.8", license_name="Gemma")])

    findings = register_findings(register)

    assert any("disagrees with" in finding for finding in findings)
