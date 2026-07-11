import json

import pytest

from llb.goldset.chains import ChainItem, ChainStep, dump_chains, load_chains, validate_chains
from llb.goldset.promote_chains import promote_chain_bundle
from llb.goldset.schema import SourceSpan

_DOC_ID = "source/doc.md"
_TEXT = "Alpha links Beta. Unreviewed filler. Beta links Gamma."


def _chain(chain_id: str, *, verified: bool = True) -> ChainItem:
    first = "Alpha links Beta"
    second = "Beta links Gamma"
    first_start = _TEXT.index(first)
    second_start = _TEXT.index(second)
    return ChainItem(
        chain_id=chain_id,
        verified=verified,
        steps=[
            ChainStep(
                order=1,
                question="How are Alpha and Beta linked?",
                reference_answer=first,
                source_doc_id=_DOC_ID,
                source_spans=[
                    SourceSpan(
                        doc_id=_DOC_ID,
                        char_start=first_start,
                        char_end=first_start + len(first),
                        text=first,
                    )
                ],
            ),
            ChainStep(
                order=2,
                question="How are Beta and Gamma linked?",
                reference_answer=second,
                source_doc_id=_DOC_ID,
                source_spans=[
                    SourceSpan(
                        doc_id=_DOC_ID,
                        char_start=second_start,
                        char_end=second_start + len(second),
                        text=second,
                    )
                ],
                dependency_note="The first step establishes the Alpha to Beta link.",
            ),
        ],
    )


def _accepted_bundle(tmp_path, chains: list[ChainItem]):
    accepted = tmp_path / "bundle" / "accepted"
    dump_chains(chains, accepted / "chains.jsonl")
    doc = accepted / "corpus" / _DOC_ID
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(_TEXT, encoding="utf-8")
    return tmp_path / "bundle"


def test_promote_chain_bundle_compacts_and_remaps_corpus(tmp_path):
    bundle = _accepted_bundle(tmp_path, [_chain("c1"), _chain("c2")])
    output = tmp_path / "fixture"

    manifest = promote_chain_bundle(bundle, output, min_chains=2)

    promoted = load_chains(output / "chains.jsonl")
    compact_text = (output / "corpus" / _DOC_ID).read_text(encoding="utf-8")
    assert len(promoted) == 2
    assert all(chain.verified for chain in promoted)
    assert "Unreviewed filler" not in compact_text
    assert not validate_chains(promoted, output / "corpus")["errors"]
    assert manifest["chains"] == 2
    on_disk = json.loads((output / "fixture_manifest.json").read_text(encoding="utf-8"))
    assert on_disk["compact_corpus"] is True
    assert on_disk["documents"][0]["reviewed_spans"] == 2


def test_promote_chain_bundle_rejects_too_few_chains(tmp_path):
    bundle = _accepted_bundle(tmp_path, [_chain("c1")])

    with pytest.raises(ValueError, match="below required minimum"):
        promote_chain_bundle(bundle, tmp_path / "fixture", min_chains=2)


def test_promote_chain_bundle_rejects_unverified_rows(tmp_path):
    bundle = _accepted_bundle(tmp_path, [_chain("c1", verified=False)])

    with pytest.raises(ValueError, match="unverified chains"):
        promote_chain_bundle(bundle, tmp_path / "fixture", min_chains=1)


def test_promote_chain_bundle_refuses_existing_destination(tmp_path):
    bundle = _accepted_bundle(tmp_path, [_chain("c1")])
    output = tmp_path / "fixture"
    output.mkdir()

    with pytest.raises(ValueError, match="destination already exists"):
        promote_chain_bundle(bundle, output, min_chains=1)
