"""Per-question yield audit for `--repeat-blocks drop`: which questions did the strip re-home?

`drop` lifts POOLED recall by removing repeated boilerplate from the index, but it is not
loss-free at the QUESTION level: an item whose gold evidence sat on a dropped copy is re-homed
onto the byte-identical survivor, which sits in a DIFFERENT section, and an item whose evidence
straddled a rewrite is dropped from the scored set entirely
(`llb.prep.pdf.repeat_corpus.GoldsetRemap`). Pooled recall cannot say whether a re-homed question
is still answerable.

This audit asks it directly, per item, on the retrieval seam alone (no model): retrieve on the
BASELINE (keep) corpus against the item's original spans, retrieve on the STRIPPED (drop) corpus
against its remapped spans, and compare the two top-k hits. A re-homed item that retrieval still
covers on the stripped corpus is `held`; one it no longer covers is `lost`; a dropped-from-set
item that the baseline could answer is a hard loss. The verdict names the questions `drop`
silently moves so an operator adopts or holds it with that list in view, not just the pooled delta.
"""

from typing import Protocol

from typing_extensions import TypedDict

from llb.core.contracts.rag import ChunkRecord, SourceSpanRecord
from llb.goldset.schema import GoldItem
from llb.rag.retrieval import recall_at_k

# How the strip touched an item, from the goldset remap.
CHANGE_UNMOVED = "unmoved"  # no span sat in a rewritten block
CHANGE_REHOMED = "rehomed"  # a span's evidence moved onto the survivor of a dropped copy
CHANGE_DROPPED = "dropped"  # a span straddled a rewrite; the item left the scored set

# Per-item retrieval verdict.
VERDICT_HELD = "held"  # baseline hit AND stripped hit -- yield preserved
VERDICT_LOST = "lost"  # baseline hit, stripped miss -- drop cost this question
VERDICT_RECOVERED = "recovered"  # baseline miss, stripped hit -- drop helped
VERDICT_STILL_MISSING = "still_missing"  # miss both sides -- unrelated to drop
VERDICT_DROPPED = "dropped_from_set"  # removed entirely; hard loss iff the baseline could answer it


class Retriever(Protocol):
    """The RAG-store seam (shared with `compare_retrieval`): question -> top-k chunks."""

    def retrieve(self, question: str, k: int) -> list[ChunkRecord]: ...


class ItemYield(TypedDict):
    """One question's fate under the strip."""

    id: str
    change: str
    baseline_hit: bool
    stripped_hit: bool | None  # None when the item was dropped from the scored set
    verdict: str


class YieldReport(TypedDict):
    """Per-question yield of a `drop` strip beside its pooled recall on the same items."""

    k: int
    n: int
    kept: int  # items still in the scored set after the strip (n minus dropped)
    dropped_from_set: int  # items whose evidence straddled a rewrite and left the set
    # Pooled recall@k over the KEPT items only, so the two are one comparable denominator; the
    # dropped items are reported separately as a hard loss, not folded into a recall miss.
    baseline_recall: float
    stripped_recall: float
    verdicts: dict[str, int]  # verdict -> count over ALL items
    moved: list[
        ItemYield
    ]  # every item whose fate the strip changed (re-homed, dropped, or flipped)
    lost: list[str]  # ids the strip cost: a flip to miss, or a dropped item the baseline could hit
    adopt: bool  # True when the strip cost no question retrieval could previously answer


def audit_repeat_yield(
    baseline_items: list[GoldItem],
    stripped_items: list[GoldItem],
    baseline_store: Retriever,
    stripped_store: Retriever,
    dropped_ids: set[str],
    rehomed_ids: set[str],
    k: int,
) -> YieldReport:
    """Compare per-item retrieval on the keep vs drop corpora and name the re-homed questions.

    `baseline_items` is the original gold set; `stripped_items` is its remap (dropped items
    absent, re-homed items pointing at the survivor). Both stores are queried once per item.
    Pooled recall is measured over the KEPT items on one denominator; a dropped item is a hard
    loss reported on its own, never diluted into the recall figure.
    """
    stripped_by_id = {item.id: item for item in stripped_items}
    per_item: list[ItemYield] = []
    baseline_kept_hits = 0
    stripped_kept_hits = 0
    for item in baseline_items:
        baseline_hit = _hit(baseline_store, item, k)
        change = _change(item.id, dropped_ids, rehomed_ids)
        if change == CHANGE_DROPPED:
            per_item.append(_yield(item.id, change, baseline_hit, None, VERDICT_DROPPED))
            continue
        stripped_hit = _hit(stripped_store, stripped_by_id[item.id], k)
        baseline_kept_hits += baseline_hit
        stripped_kept_hits += stripped_hit
        per_item.append(
            _yield(
                item.id, change, baseline_hit, stripped_hit, _verdict(baseline_hit, stripped_hit)
            )
        )
    return _report(per_item, baseline_kept_hits, stripped_kept_hits, k)


def format_yield_report(report: YieldReport) -> str:
    """ASCII summary lines for the CLI (AGENTS.md: ASCII-only)."""
    verdicts = report["verdicts"]
    lines = [
        f"[audit-repeat-yield] n={report['n']} k={report['k']} "
        f"(kept {report['kept']}, dropped {report['dropped_from_set']})",
        f"  pooled recall@{report['k']} over kept items: keep {report['baseline_recall']:.3f} "
        f"-> drop {report['stripped_recall']:.3f} "
        f"({report['stripped_recall'] - report['baseline_recall']:+.3f})",
        "  verdicts: "
        + ", ".join(f"{verdict} {count}" for verdict, count in sorted(verdicts.items()) if count),
    ]
    for entry in report["moved"]:
        stripped = "n/a" if entry["stripped_hit"] is None else _hit_word(entry["stripped_hit"])
        lines.append(
            f"    {entry['id']}: {entry['change']} -- keep {_hit_word(entry['baseline_hit'])} "
            f"-> drop {stripped} [{entry['verdict']}]"
        )
    verdict = "ADOPT" if report["adopt"] else "HOLD"
    detail = (
        "the strip cost no question retrieval could answer"
        if report["adopt"]
        else f"the strip cost {len(report['lost'])} answerable question(s): {', '.join(report['lost'])}"
    )
    lines.append(f"  verdict: {verdict} -- {detail}")
    return "\n".join(lines)


def _hit(store: Retriever, item: GoldItem, k: int) -> bool:
    spans: list[SourceSpanRecord] = [
        {
            "doc_id": span.doc_id,
            "char_start": span.char_start,
            "char_end": span.char_end,
            "text": span.text,
        }
        for span in item.source_spans
    ]
    return recall_at_k(store.retrieve(item.question, k), spans, k) == 1.0


def _change(item_id: str, dropped_ids: set[str], rehomed_ids: set[str]) -> str:
    if item_id in dropped_ids:
        return CHANGE_DROPPED
    if item_id in rehomed_ids:
        return CHANGE_REHOMED
    return CHANGE_UNMOVED


def _verdict(baseline_hit: bool, stripped_hit: bool) -> str:
    if baseline_hit and stripped_hit:
        return VERDICT_HELD
    if baseline_hit and not stripped_hit:
        return VERDICT_LOST
    if not baseline_hit and stripped_hit:
        return VERDICT_RECOVERED
    return VERDICT_STILL_MISSING


def _yield(
    item_id: str, change: str, baseline_hit: bool, stripped_hit: bool | None, verdict: str
) -> ItemYield:
    return {
        "id": item_id,
        "change": change,
        "baseline_hit": baseline_hit,
        "stripped_hit": stripped_hit,
        "verdict": verdict,
    }


def _report(
    per_item: list[ItemYield], baseline_kept_hits: int, stripped_kept_hits: int, k: int
) -> YieldReport:
    n = len(per_item)
    dropped_from_set = sum(1 for entry in per_item if entry["verdict"] == VERDICT_DROPPED)
    kept = n - dropped_from_set
    verdicts: dict[str, int] = {}
    for entry in per_item:
        verdicts[entry["verdict"]] = verdicts.get(entry["verdict"], 0) + 1
    # a lost question: retrieval flipped to a miss (whether the item was re-homed or an unmoved
    # item the corpus-wide edit shifted), or a dropped item the baseline could answer.
    lost = [
        entry["id"]
        for entry in per_item
        if entry["verdict"] == VERDICT_LOST
        or (entry["verdict"] == VERDICT_DROPPED and entry["baseline_hit"])
    ]
    # every item whose fate the strip changed: re-homed/dropped by construction, plus any item
    # whose retrieval flipped (a ranking side-effect of the corpus-wide edit).
    moved = [
        entry
        for entry in per_item
        if entry["change"] != CHANGE_UNMOVED
        or entry["verdict"] in (VERDICT_LOST, VERDICT_RECOVERED)
    ]
    return {
        "k": k,
        "n": n,
        "kept": kept,
        "dropped_from_set": dropped_from_set,
        "baseline_recall": baseline_kept_hits / kept if kept else 0.0,
        "stripped_recall": stripped_kept_hits / kept if kept else 0.0,
        "verdicts": verdicts,
        "moved": moved,
        "lost": lost,
        "adopt": not lost,
    }


def _hit_word(hit: bool) -> str:
    return "hit" if hit else "miss"
