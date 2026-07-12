"""Stratified sampling + worksheet construction for the human verification gate.

The sampling half of `verify.py`: stratification, deterministic draws, corpus windows, the
read-only reviewer signals (cross-check verdict, retrieval rank, page citation), the confidence
ordering of the review queue, and building/merging the verification worksheet from a draft
bundle. Everything here is pure I/O over the bundle -- no model, endpoint, or GPU -- so it stays
fully unit-tested. Shared constants, bundle-layout helpers, and the atomic worksheet I/O live in
`verify.py`, which re-exports these names so `llb.goldset.verify.<name>` keeps working.
"""

import json
import random
from collections.abc import Sequence
from pathlib import Path

from llb.core.fsutil import atomic_write_text
from llb.goldset.chains import ChainItem, chain_stratum_key, load_chains
from llb.goldset.schema import GoldItem, load_goldset
from llb.goldset.verify_base import (
    CONTEXT_CHARS,
    CORPUS_DIRNAME,
    CROSS_CHECK_SUFFIX,
    KIND_AUTO,
    KIND_CHAINS,
    KIND_GOLDSET,
    RETRIEVAL_RANK_SOURCES,
    SAMPLE_MANIFEST,
    _worksheet_sample_kind,
    bundle_is_synthetic,
    find_chains,
    find_goldset,
    load_worksheet,
    resolve_sample_kind,
    write_worksheet_rows,
)
from llb.rag.page_metadata import intersect_pages, load_page_citations

# --- strata -------------------------------------------------------------------------------


def stratum_key(item: GoldItem) -> str:
    """The stratum an item belongs to: provenance x split x source doc.

    These are the axes actually present on a canonical `GoldItem` (the draft pipeline's free-form
    `kind`/`difficulty` tags are dropped on dump; synthetic is a bundle-level constant). Sampling
    per stratum keeps an error concentrated in one cell -- e.g. a single source doc -- from hiding
    behind a clean overall rate."""
    return f"{item.provenance}|{item.split}|{item.source_doc_id}"


def stratify(items: Sequence[GoldItem]) -> dict[str, list[GoldItem]]:
    """Group items by `stratum_key`, preserving input order within each stratum."""
    strata: dict[str, list[GoldItem]] = {}
    for item in items:
        strata.setdefault(stratum_key(item), []).append(item)
    return strata


def stratum_quotas(strata_sizes: dict[str, int], n: int) -> dict[str, int]:
    """Exact per-stratum allocation summing to `min(n, population)` (deterministic).

    Floor of one per non-empty stratum first (largest strata first when `n` cannot cover them
    all), then a largest-remainder top-up distributes the remaining budget proportionally,
    each stratum capped at its own size -- so proportional rounding can neither overshoot nor
    undershoot the requested sample size (verify-sample-exact-allocation).
    """
    total = sum(strata_sizes.values())
    budget = min(n, total)
    quotas = {key: 0 for key in strata_sizes}
    allocated = 0
    for key in sorted(strata_sizes, key=lambda k: (-strata_sizes[k], k)):
        if allocated >= budget:
            break
        if strata_sizes[key] > 0:
            quotas[key] = 1
            allocated += 1
    while allocated < budget:
        open_keys = [key for key in strata_sizes if quotas[key] < strata_sizes[key]]
        winner = max(
            sorted(open_keys),
            key=lambda k: (n * strata_sizes[k] / total) - quotas[k],
        )
        quotas[winner] += 1
        allocated += 1
    return quotas


def draw_stratified_sample(items: Sequence[GoldItem], n: int, *, seed: int = 13) -> list[GoldItem]:
    """Draw exactly `min(n, population)` items spread across strata (deterministic given `seed`).

    Allocates the budget with `stratum_quotas` (floor of one per non-empty stratum, exact
    largest-remainder proportional top-up), shuffles within each stratum by `seed`, then
    returns the union in canonical (input) order. If `n` >= len(items), returns all items.
    """
    total = len(items)
    if n >= total:
        return list(items)
    strata = stratify(items)
    rng = random.Random(seed)
    picked: set[int] = set()
    index = {id(it): i for i, it in enumerate(items)}
    quotas = stratum_quotas({key: len(group) for key, group in strata.items()}, n)
    for key, group in sorted(strata.items()):
        order = list(group)
        rng.shuffle(order)
        for it in order[: quotas[key]]:
            picked.add(index[id(it)])
    return [items[i] for i in sorted(picked)]


def draw_chain_sample(chains: Sequence[ChainItem], n: int, *, seed: int = 13) -> list[ChainItem]:
    """Draw exactly `min(n, population)` chain items spread across chain strata."""
    total = len(chains)
    if n >= total:
        return list(chains)
    strata: dict[str, list[ChainItem]] = {}
    for chain in chains:
        strata.setdefault(chain_stratum_key(chain), []).append(chain)
    rng = random.Random(seed)
    picked: set[int] = set()
    index = {id(chain): i for i, chain in enumerate(chains)}
    quotas = stratum_quotas({key: len(group) for key, group in strata.items()}, n)
    for key, group in sorted(strata.items()):
        order = list(group)
        rng.shuffle(order)
        for chain in order[: quotas[key]]:
            picked.add(index[id(chain)])
    return [chains[i] for i in sorted(picked)]


# --- cross-check sidecar ------------------------------------------------------------------


def load_cross_check(bundle: Path) -> dict[str, dict[str, object]]:
    """Index any `*.cross_check.json` verdicts in the bundle by item id (empty if none).

    The verdict is read-only context for the human (the second frontier already ran these
    checks); it is shown only with `--show-crosscheck` so it never anchors the verification.
    """
    verdicts: dict[str, dict[str, object]] = {}
    for path in sorted(Path(bundle).glob(f"*{CROSS_CHECK_SUFFIX}")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for v in payload.get("verdicts", []):
            item_id = v.get("item_id")
            if item_id:
                verdicts[str(item_id)] = v
    return verdicts


# --- corpus windows -----------------------------------------------------------------------


def corpus_window(text: str, char_start: int, char_end: int, ctx: int = CONTEXT_CHARS) -> str:
    """Render the cited span inside its surrounding corpus text, the span delimited by >>><<<.

    The window is what the human reads to confirm grounding without leaving the tool; it is
    captured into the worksheet at sample time so the CSV stays a self-contained artifact.
    """
    lo = max(0, char_start - ctx)
    hi = min(len(text), char_end + ctx)
    before = text[lo:char_start]
    span = text[char_start:char_end]
    after = text[char_end:hi]
    prefix = "..." if lo > 0 else ""
    suffix = "..." if hi < len(text) else ""
    return f"{prefix}{before}>>>{span}<<<{after}{suffix}"


def _corpus_text(corpus_root: Path, doc_id: str, cache: dict[str, str | None]) -> str | None:
    if doc_id not in cache:
        path = corpus_root / doc_id
        cache[doc_id] = path.read_text(encoding="utf-8") if path.is_file() else None
    return cache[doc_id]


# --- retrieval-rank + page-citation context (read-only reviewer signals) --------------------


def load_retrieval_ranks(bundle: Path) -> dict[str, int]:
    """Index per-item `retrieval_rank` from any bundle sidecar that records it (empty if none).

    Reads `RETRIEVAL_RANK_SOURCES` (rows keyed by `id`); a null / missing rank means the item's
    gold span was NOT retrieved within top-k and is simply absent from the result.
    """
    ranks: dict[str, int] = {}
    for name in RETRIEVAL_RANK_SOURCES:
        path = Path(bundle) / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            item_id = row.get("id")
            rank = row.get("retrieval_rank")
            if item_id and isinstance(rank, int) and rank > 0:
                ranks[str(item_id)] = rank
    return ranks


def page_citation_for_span(
    corpus_root: Path,
    doc_id: str,
    char_start: int,
    char_end: int,
    cache: dict[str, tuple[str | None, list[dict[str, object]]] | None],
) -> str:
    """Render a `<source.pdf> p.N[-M]` citation for a span, "" when no PDF sidecar covers it.

    Reuses the PDF lane's `*.citations.json` sidecar (the same page join `build-index` uses), so
    the reviewer can check the original PDF page without hunting for offsets.
    """
    if doc_id not in cache:
        cache[doc_id] = load_page_citations(Path(corpus_root), doc_id)
    cite = cache[doc_id]
    if cite is None:
        return ""
    source, spans = cite
    pages = intersect_pages(char_start, char_end, spans)
    if not pages:
        return ""
    label = f"p.{pages[0]}" if pages[0] == pages[-1] else f"p.{pages[0]}-{pages[-1]}"
    name = Path(source).name if source else ""
    return f"{name} {label}".strip()


# --- confidence ordering (review queue) -----------------------------------------------------


def row_confidence(row: dict[str, str]) -> float:
    """A draft item's prior plausibility from the read-only signals on its worksheet row.

    Each cross-check verdict flag contributes +1 (true) / -1 (false); a top-k retrieval rank
    contributes 1/rank. Higher = more likely fine. Purely heuristic -- it ORDERS the review
    queue and never decides anything.
    """
    score = 0.0
    for col in ("cc_grounded", "cc_non_circular", "cc_supported", "cc_answerable"):
        value = (row.get(col) or "").strip().lower()
        if value == "true":
            score += 1.0
        elif value == "false":
            score -= 1.0
    rank = (row.get("retrieval_rank") or "").strip()
    if rank.isdigit() and int(rank) > 0:
        score += 1.0 / int(rank)
    return score


def confidence_order(rows: Sequence[dict[str, str]]) -> list[int]:
    """Row indices ordered LEAST-confident first (ties keep worksheet order).

    Suspicious items meet the reviewer's fresh attention first; the tail becomes quick
    confirmations -- that is the throughput win. The worksheet itself is never reordered.
    """
    return sorted(range(len(rows)), key=lambda i: (row_confidence(rows[i]), i))


# --- worksheet row building ----------------------------------------------------------------


def _row_for(
    item: GoldItem,
    corpus_root: Path,
    cache: dict[str, str | None],
    verdict: dict[str, object],
    *,
    synthetic: bool,
    retrieval_rank: int | None = None,
    page_cache: dict[str, tuple[str | None, list[dict[str, object]]] | None] | None = None,
) -> dict[str, str]:
    span = item.source_spans[0]
    text = _corpus_text(corpus_root, span.doc_id, cache)
    context = (
        corpus_window(text, span.char_start, span.char_end) if text is not None else "(missing doc)"
    )
    page = (
        page_citation_for_span(corpus_root, span.doc_id, span.char_start, span.char_end, page_cache)
        if page_cache is not None
        else ""
    )

    def _flag(key: str) -> str:
        value = verdict.get(key)
        return "" if value is None else ("true" if value else "false")

    return {
        "item_kind": KIND_GOLDSET,
        "item_id": item.id,
        "provenance": item.provenance,
        "split": item.split,
        "source_doc_id": item.source_doc_id,
        "synthetic": "true" if synthetic else "false",
        "stratum": stratum_key(item),
        "question": item.question,
        "reference_answer": item.reference_answer,
        "span_doc_id": span.doc_id,
        "span_text": span.text,
        "context": context,
        "retrieval_rank": "" if retrieval_rank is None else str(retrieval_rank),
        "page_citation": page,
        "chain_steps": "",
        "cc_grounded": _flag("grounded"),
        "cc_non_circular": _flag("non_circular"),
        "cc_supported": _flag("supported"),
        "cc_answerable": _flag("answerable"),
        "cc_note": str(verdict.get("note", "")),
    }


def _chain_step_contexts(
    chain: ChainItem,
    corpus_root: Path,
    cache: dict[str, str | None],
    page_cache: dict[str, tuple[str | None, list[dict[str, object]]] | None],
) -> list[dict[str, str]]:
    steps: list[dict[str, str]] = []
    for step in chain.steps:
        span = step.source_spans[0]
        text = _corpus_text(corpus_root, span.doc_id, cache)
        context = (
            corpus_window(text, span.char_start, span.char_end)
            if text is not None
            else "(missing doc)"
        )
        page = page_citation_for_span(
            corpus_root, span.doc_id, span.char_start, span.char_end, page_cache
        )
        steps.append(
            {
                "order": str(step.order),
                "question": step.question,
                "reference_answer": step.reference_answer,
                "dependency_note": step.dependency_note,
                "span_doc_id": span.doc_id,
                "span_text": span.text,
                "context": context,
                "page_citation": page,
            }
        )
    return steps


def _row_for_chain(
    chain: ChainItem,
    corpus_root: Path,
    cache: dict[str, str | None],
    page_cache: dict[str, tuple[str | None, list[dict[str, object]]] | None],
) -> dict[str, str]:
    steps = _chain_step_contexts(chain, corpus_root, cache, page_cache)
    final = steps[-1] if steps else {}
    first_doc = chain.steps[0].source_doc_id if chain.steps else ""
    return {
        "item_kind": KIND_CHAINS,
        "item_id": chain.chain_id,
        "provenance": chain.provenance,
        "split": chain.split,
        "source_doc_id": first_doc,
        "synthetic": "false",
        "stratum": chain_stratum_key(chain),
        "question": " -> ".join(step.question for step in chain.steps),
        "reference_answer": chain.steps[-1].reference_answer if chain.steps else "",
        "span_doc_id": final.get("span_doc_id", ""),
        "span_text": final.get("span_text", ""),
        "context": final.get("context", ""),
        "retrieval_rank": "",
        "page_citation": final.get("page_citation", ""),
        "chain_steps": json.dumps(steps, ensure_ascii=False),
    }


def _sample_rows(
    bundle: Path, sample: Sequence[GoldItem], *, synthetic: bool
) -> list[dict[str, str]]:
    """Build worksheet rows for `sample`, joining cross-check, retrieval rank, and page cites."""
    verdicts = load_cross_check(bundle)
    ranks = load_retrieval_ranks(bundle)
    corpus_root = bundle / CORPUS_DIRNAME
    cache: dict[str, str | None] = {}
    page_cache: dict[str, tuple[str | None, list[dict[str, object]]] | None] = {}
    return [
        _row_for(
            it,
            corpus_root,
            cache,
            verdicts.get(it.id, {}),
            synthetic=synthetic,
            retrieval_rank=ranks.get(it.id),
            page_cache=page_cache,
        )
        for it in sample
    ]


def _sample_chain_rows(bundle: Path, sample: Sequence[ChainItem]) -> list[dict[str, str]]:
    """Build worksheet rows for chain samples, with every step rendered into chain_steps JSON."""
    corpus_root = bundle / CORPUS_DIRNAME
    cache: dict[str, str | None] = {}
    page_cache: dict[str, tuple[str | None, list[dict[str, object]]] | None] = {}
    return [_row_for_chain(chain, corpus_root, cache, page_cache) for chain in sample]


def _write_sample_manifest(
    out_path: Path,
    bundle: Path,
    manifest_update: dict[str, object],
    *,
    merge_existing: bool = False,
) -> None:
    """Write the sibling `sample_manifest.json` (merge into the existing one on enlargement)."""
    manifest_path = Path(out_path).with_name(SAMPLE_MANIFEST)
    manifest: dict[str, object] = {}
    if merge_existing and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
    manifest.update({"bundle": str(bundle), "worksheet": str(out_path)})
    manifest.update(manifest_update)
    atomic_write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2))


def build_sample_worksheet(
    bundle: Path, out_path: Path, *, n: int, seed: int = 13, kind: str = KIND_AUTO
) -> tuple[int, dict[str, int]]:
    """Draw a stratified sample from a draft bundle and write the verification worksheet.

    Returns `(sample_size, strata_sizes)` and writes a `sample_manifest.json` beside the
    worksheet documenting the bundle, seed, requested/actual size, and per-stratum counts (the
    "document the size + strata" half of the procedure). Human columns are left blank.
    """
    bundle = Path(bundle)
    resolved_kind = resolve_sample_kind(bundle, kind)
    synthetic = bundle_is_synthetic(bundle) if resolved_kind == KIND_GOLDSET else False
    if resolved_kind == KIND_CHAINS:
        chains = load_chains(find_chains(bundle))
        chain_sample = draw_chain_sample(chains, n, seed=seed)
        rows = _sample_chain_rows(bundle, chain_sample)
        population = len(chains)
        sample_size = len(chain_sample)
        strata_sizes: dict[str, int] = {}
        for chain in chain_sample:
            key = chain_stratum_key(chain)
            strata_sizes[key] = strata_sizes.get(key, 0) + 1
    else:
        items = load_goldset(find_goldset(bundle))
        gold_sample = draw_stratified_sample(items, n, seed=seed)
        rows = _sample_rows(bundle, gold_sample, synthetic=synthetic)
        population = len(items)
        sample_size = len(gold_sample)
        strata_sizes = {}
        for it in gold_sample:
            strata_sizes[stratum_key(it)] = strata_sizes.get(stratum_key(it), 0) + 1
    write_worksheet_rows(out_path, rows)

    _write_sample_manifest(
        out_path,
        bundle,
        {
            "kind": resolved_kind,
            "synthetic": synthetic,
            "seed": seed,
            "requested": n,
            "sample_size": sample_size,
            "population": population,
            "strata": strata_sizes,
        },
    )
    return sample_size, strata_sizes


def merge_sample_worksheet(
    bundle: Path, out_path: Path, *, n: int, seed: int = 13, kind: str = KIND_AUTO
) -> tuple[int, int]:
    """Enlarge an existing worksheet ADDITIVELY to ~`n` rows; returns `(added, total)`.

    Draws a fresh stratified sample of size `n` over the whole bundle and appends rows ONLY for
    item ids the worksheet does not already hold: existing rows -- including every human decision
    -- are rewritten byte-for-byte untouched, and a decided row is never re-shown or re-drawn.
    Falls back to a fresh `build_sample_worksheet` when the worksheet does not exist yet.
    """
    out_path = Path(out_path)
    if not out_path.is_file():
        size, _ = build_sample_worksheet(bundle, out_path, n=n, seed=seed, kind=kind)
        return size, size
    bundle = Path(bundle)
    manifest_kind = _worksheet_sample_kind(out_path)
    resolved_kind = resolve_sample_kind(bundle, manifest_kind if kind == KIND_AUTO else kind)
    existing_rows, fieldnames = load_worksheet(out_path)
    existing_ids = {(row.get("item_id") or "").strip() for row in existing_rows}
    synthetic = bundle_is_synthetic(bundle) if resolved_kind == KIND_GOLDSET else False
    new_rows, population = _new_sample_rows(
        bundle, resolved_kind, n, seed, existing_ids, synthetic=synthetic
    )
    if not new_rows:
        return 0, len(existing_rows)
    all_rows = [*existing_rows, *new_rows]
    write_worksheet_rows(out_path, all_rows, fieldnames)
    _write_sample_manifest(
        out_path,
        bundle,
        {
            "kind": resolved_kind,
            "synthetic": synthetic,
            "seed": seed,
            "requested": n,
            "sample_size": len(all_rows),
            "population": population,
            "merged_added": len(new_rows),
        },
        merge_existing=True,
    )
    return len(new_rows), len(all_rows)


def _new_sample_rows(
    bundle: Path,
    resolved_kind: str,
    n: int,
    seed: int,
    existing_ids: set[str],
    *,
    synthetic: bool,
) -> tuple[list[dict[str, str]], int]:
    """`(new worksheet rows, population size)` for ids the worksheet does not already hold."""
    if resolved_kind == KIND_CHAINS:
        chains = load_chains(find_chains(bundle))
        chain_sample = draw_chain_sample(chains, n, seed=seed)
        new_chains = [chain for chain in chain_sample if chain.chain_id not in existing_ids]
        rows = _sample_chain_rows(bundle, new_chains) if new_chains else []
        return rows, len(chains)
    items = load_goldset(find_goldset(bundle))
    gold_sample = draw_stratified_sample(items, n, seed=seed)
    new_items = [it for it in gold_sample if it.id not in existing_ids]
    rows = _sample_rows(bundle, new_items, synthetic=synthetic) if new_items else []
    return rows, len(items)
