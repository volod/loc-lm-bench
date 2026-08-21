"""Render the per-hop probe as ASCII Markdown (AGENTS.md: no Unicode, no box-drawing).

Four blocks, in the order the diagnosis is read: the coverage curve per budget (does a bigger k
carry both hops?), the budget histogram (how big would k have to be?), the counted diagnoses with
the explanation they support, and the per-item hop ledger (with n in the tens, the items ARE the
evidence).
"""

from llb.rag.fusion_evidence.stats import format_interval
from llb.rag.multihop_probe.models import (
    BUDGET_BUCKET_BEYOND,
    DIAGNOSES,
    MultiHopProbeReport,
    SliceProbe,
)

_UNREACHED = "-"


def _rank(rank: int | None) -> str:
    return _UNREACHED if rank is None else str(rank)


def _curve_table(slice_probe: SliceProbe, confidence: float, probe_depth: int) -> list[str]:
    lines = [
        f"Coverage curve (n={slice_probe['n']} items, {slice_probe['n_hops']} labeled spans, "
        f"{confidence:.0%} bootstrap CI on all-spans@k):",
        "",
        "| k | all-spans@k | span coverage | recall@k | hop hit rate | hop hit rate, span query |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for point in slice_probe["curve"]:
        lines.append(
            f"| {point['k']} | {format_interval(point['all_spans_at_k'])} "
            f"| {point['span_coverage']:.3f} | {point['recall_at_k']:.3f} "
            f"| {point['hop_hit_rate']:.3f} | {point['span_query_hop_hit_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "`hop hit rate` is the share of LABELED SPANS the item's own question ranks within k; "
            f"the last column re-asks each span with its own text as the query (probe depth "
            f"{probe_depth}), which is the most favorable query that span can be given.",
            "",
        ]
    )
    return lines


def _histogram_table(slice_probe: SliceProbe, probe_depth: int) -> list[str]:
    histogram = slice_probe["diagnosis"]["budget_histogram"]
    lines = [
        "Smallest cutoff that carries EVERY labeled span of an item, under the item's question:",
        "",
        "| smallest sufficient k | items |",
        "| --- | ---: |",
    ]
    for bucket, count in histogram.items():
        label = "never reached" if bucket == BUDGET_BUCKET_BEYOND else f"<= {bucket}"
        lines.append(f"| {label} | {count} |")
    lines.extend(
        [
            "",
            f"These are ranks in the ONE deep pass (depth {probe_depth}). A lane that re-fuses its "
            "candidate pool per requested depth (a hybrid dense/lexical store does) can rank the "
            "same chunk differently at a shallow k, so this histogram and the curve above may "
            "disagree by an item; the curve is what the operator gets, the histogram is why.",
            "",
        ]
    )
    return lines


def _diagnosis_block(slice_probe: SliceProbe) -> list[str]:
    diagnosis = slice_probe["diagnosis"]
    operating = slice_probe["curve"][0]["k"]
    lines = [
        "| diagnosis | items |",
        "| --- | ---: |",
        *(f"| {name} | {diagnosis['counts'][name]} |" for name in DIAGNOSES),
        "",
        f"`covered` is the measured `all-spans@{operating}` outcome -- the items the retrieval AT "
        "the operating budget carried every labeled span for -- so this row always agrees with the "
        "curve above. The other three read the deep-pass ranks, because they answer what would fix "
        "the miss.",
        "",
        f"**Explanation supported: {diagnosis['explanation']}** -- {diagnosis['reason']}.",
        "",
    ]
    return lines


def _slice_section(
    name: str, slice_probe: SliceProbe, confidence: float, probe_depth: int
) -> list[str]:
    return [
        f"### Slice: {name}",
        "",
        *(
            ["No item falls in this slice, so nothing is measured here.", ""]
            if slice_probe["n"] == 0
            else [
                *_curve_table(slice_probe, confidence, probe_depth),
                *_histogram_table(slice_probe, probe_depth),
                *_diagnosis_block(slice_probe),
            ]
        ),
    ]


def _context_row(label: str, slice_probe: SliceProbe) -> str:
    """One compact line per context slice: the curve as point estimates plus its explanation."""
    curve = " | ".join(f"{point['all_spans_at_k']['mean']:.3f}" for point in slice_probe["curve"])
    counts = slice_probe["diagnosis"]["counts"]
    ledger = "/".join(str(counts[name]) for name in DIAGNOSES)
    return (
        f"| {label} | {slice_probe['n']} | {curve} | {ledger} "
        f"| {slice_probe['diagnosis']['explanation']} |"
    )


def _context_table(report: MultiHopProbeReport) -> list[str]:
    """Every item set the focus slice is read against, one row each (the focus slice is above)."""
    focus = report["focus_slice"]
    others = sorted(name for name in report["slices"] if name != focus)
    budgets = " | ".join(f"all-spans@{k}" for k in report["budgets"])
    return [
        "### Context slices",
        "",
        f"`ledger` counts items as {'/'.join(DIAGNOSES)}.",
        "",
        f"| item set | n | {budgets} | ledger | explanation |",
        "| --- | ---: | " + " | ".join(["---:"] * len(report["budgets"])) + " | :-: | --- |",
        _context_row("every item", report["overall"]),
        *(_context_row(name, report["slices"][name]) for name in others),
        "",
    ]


def _item_ledger(report: MultiHopProbeReport) -> list[str]:
    if not report["items"]:
        return []
    lines = [
        f"### Per-item hop ledger ({report['focus_slice']})",
        "",
        "`question rank` / `span-query rank` are 1-based ranks of the first retrieved chunk "
        f"covering that labeled span, searched to depth {report['probe_depth']}; "
        f"`{_UNREACHED}` means the span was never reached.",
        "",
        "| item | spans | question ranks | span-query ranks | smallest sufficient k | diagnosis |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]
    for item in report["items"]:
        question_ranks = " / ".join(_rank(hop["question_rank"]) for hop in item["hops"])
        span_ranks = " / ".join(_rank(hop["span_query_rank"]) for hop in item["hops"])
        smallest = item["min_budget"]
        label = "never" if smallest == BUDGET_BUCKET_BEYOND else str(smallest)
        lines.append(
            f"| {item['item_id']} | {item['n_spans']} | {question_ranks} | {span_ranks} "
            f"| {label} | {item['diagnosis']} |"
        )
    lines.append("")
    return lines


def format_probe_report(report: MultiHopProbeReport) -> str:
    """The Markdown artifact written beside `probe.json`."""
    focus = report["focus_slice"]
    lines = [
        "# Multi-hop per-hop retrievability probe",
        "",
        f"Lane `{report['lane']}`, {report['n_items']} scored items, budgets "
        f"{', '.join(str(k) for k in report['budgets'])}, probe depth {report['probe_depth']}, "
        f"seed {report['seed']}, {report['resamples']} resamples.",
        "",
        "The lane separates the two explanations a stuck `all-spans@k` can have: the missing hop "
        "is retrievable by the question and sits below the cut (BUDGET), or the question never "
        "reaches it at any depth while its own text does (QUERY). They lead to opposite fixes. A hop "
        "no query form reaches at the operating budget is neither, and is counted separately.",
        "",
    ]
    lines.extend(
        _slice_section(focus, report["slices"][focus], report["confidence"], report["probe_depth"])
    )
    lines.extend(_context_table(report))
    lines.extend(_item_ledger(report))
    return "\n".join(lines).rstrip() + "\n"
