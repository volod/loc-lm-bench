"""Shared fixtures/factories for the verification tests (split across test_goldset_verify*.py).

Not collected by pytest (module name does not start with `test_`). The gold-item / chain / bundle
/ worksheet-row builders live here so the sampling+acceptance and the session test modules share
one definition.
"""

import json

from llb.goldset.chains import CHAINS_FILENAME, ChainItem, ChainStep, dump_chains
from llb.goldset.schema import GoldItem, SourceSpan, dump_goldset
from llb.goldset.verify import WORKSHEET_COLS, write_worksheet_rows

DOC = "squad/doc1.txt"
TEXT = "Леся Українка народилася 1871 року в Новограді-Волинському. Вона була поетесою."
CHAIN_DOC = "chains/doc.txt"
CHAIN_TEXT = "Alpha керує Beta. Beta належить Gamma. Gamma має офіс у Києві."


def _item(item_id, *, answer="1871", provenance="frontier-drafted", split="calibration", doc=DOC):
    start = TEXT.find(answer)
    return GoldItem(
        id=item_id,
        question=f"Коли подія {item_id}?",
        reference_answer=answer,
        source_doc_id=doc,
        source_spans=[
            SourceSpan(doc_id=doc, char_start=start, char_end=start + len(answer), text=answer)
        ],
        provenance=provenance,
        split=split,
    )


def _bundle(tmp_path, items, *, synthetic=False):
    """Write a minimal draft bundle (goldset.jsonl + corpus/) under tmp_path."""
    dump_goldset(items, tmp_path / "goldset.jsonl")
    doc = tmp_path / "corpus" / DOC
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(TEXT + "\n", encoding="utf-8")
    if synthetic:
        (tmp_path / "provenance.json").write_text(
            json.dumps({"synthetic": True, "kind": "synthetic-planted"}), encoding="utf-8"
        )
    return tmp_path


def _chain(chain_id="c1", *, verified=False):
    first = "Alpha керує Beta"
    second = "Beta належить Gamma"
    s1 = CHAIN_TEXT.index(first)
    s2 = CHAIN_TEXT.index(second)
    return ChainItem(
        chain_id=chain_id,
        steps=[
            ChainStep(
                order=1,
                question="Що встановлено про Alpha і Beta?",
                reference_answer=first,
                source_doc_id=CHAIN_DOC,
                source_spans=[
                    SourceSpan(
                        doc_id=CHAIN_DOC,
                        char_start=s1,
                        char_end=s1 + len(first),
                        text=first,
                    )
                ],
            ),
            ChainStep(
                order=2,
                question="Що встановлено про Beta і Gamma?",
                reference_answer=second,
                source_doc_id=CHAIN_DOC,
                source_spans=[
                    SourceSpan(
                        doc_id=CHAIN_DOC,
                        char_start=s2,
                        char_end=s2 + len(second),
                        text=second,
                    )
                ],
                dependency_note="Крок 1 встановлює зв'язок Alpha і Beta.",
            ),
        ],
        verified=verified,
    )


def _chain_bundle(tmp_path, chains):
    dump_chains(chains, tmp_path / CHAINS_FILENAME)
    doc = tmp_path / "corpus" / CHAIN_DOC
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(CHAIN_TEXT, encoding="utf-8")
    return tmp_path


def _ws_row(item_id, decision="", stratum="s", **over):
    row = {col: "" for col in WORKSHEET_COLS}
    row.update({"item_id": item_id, "stratum": stratum, "decision": decision})
    row.update(over)
    return row


def _ws(tmp_path, rows):
    path = tmp_path / "verify.csv"
    write_worksheet_rows(path, rows, WORKSHEET_COLS)
    return path


def _ticking_clock(step=30.0):
    state = {"now": 0.0}

    def clock():
        state["now"] += step
        return state["now"]

    return clock
