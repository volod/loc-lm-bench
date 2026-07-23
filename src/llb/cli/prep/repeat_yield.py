"""`audit-repeat-yield`: per-question yield audit for `--repeat-blocks drop`.

Builds the drop-stripped corpus, indexes both the keep and drop corpora with the pinned embedder,
and asks -- per item -- whether retrieval still reaches the evidence of every question the strip
re-homed onto a survivor, so an operator adopts or holds `drop` with the moved-question list in
view. See `llb.prep.pdf.repeat_yield`.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import typer

from llb.cli.app import app
from llb.cli.helpers import load_config

if TYPE_CHECKING:
    from llb.core.config import RunConfig

YIELD_REPORT_NAME = "repeat_yield.json"


@app.command("audit-repeat-yield")
def audit_repeat_yield_cmd(
    corpus: Path = typer.Option(..., help="baseline (keep) converted corpus root"),
    goldset: Path = typer.Option(..., help="gold set scored against the corpus"),
    config: Optional[Path] = typer.Option(None, help="YAML run config for chunking/embedder"),
    out: Optional[Path] = typer.Option(
        None, help="working dir for the drop-stripped corpus, stores, and report"
    ),
    k: int = typer.Option(10, help="recall@k cutoff"),
    split: Optional[str] = typer.Option(None, help="restrict to one gold split"),
    min_repeats: Optional[int] = typer.Option(
        None, help="occurrences inside one document before a block counts as repeated (default 3)"
    ),
    strategy: Optional[str] = typer.Option(None, help="chunking strategy (default from config)"),
    chunk_size: Optional[int] = typer.Option(None, help="chunk size (default from config)"),
    chunk_overlap: Optional[int] = typer.Option(None, help="chunk overlap (default from config)"),
    embedding_model: Optional[str] = typer.Option(None, help="embedder id (default from config)"),
    recover_straddle: bool = typer.Option(
        False,
        "--recover-straddle",
        help="split a gold span that crosses a removed block boundary and re-anchor both sides "
        "instead of dropping the item, then audit the recovered yield",
    ),
) -> None:
    """Measure which questions `--repeat-blocks drop` re-homes, beside its pooled recall gain.

    Runs the `drop` strip into `--out`, indexes the keep and drop corpora identically, and reports
    a per-item held/lost/recovered verdict plus an adopt-or-hold decision naming any question the
    strip cost that retrieval could previously answer.
    """
    from llb.goldset.schema import load_goldset
    from llb.prep.pdf.repeat_corpus import REPEAT_REPORT_NAME, strip_corpus_repeats
    from llb.prep.pdf.repeat_yield import Retriever, audit_repeat_yield, format_yield_report
    from llb.prep.pdf.repeats import DEFAULT_MIN_REPEATS, REPEAT_DROP
    from llb.rag.store import RagStore

    cfg = load_config(
        config,
        corpus_root=corpus,
        goldset_path=goldset,
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
    )
    work = out or (cfg.data_dir / "retrieval-noise-floor" / "repeat-yield")
    drop_corpus = work / "drop-corpus"
    drop_goldset = work / "drop-goldset.jsonl"
    strip = strip_corpus_repeats(
        corpus,
        drop_corpus,
        mode=REPEAT_DROP,
        min_repeats=min_repeats or DEFAULT_MIN_REPEATS,
        goldset=goldset,
        goldset_out=drop_goldset,
        recover_straddle=recover_straddle,
    )
    (work / REPEAT_REPORT_NAME).write_text(
        json.dumps(strip, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    remap = strip["goldset"]
    if remap is None:  # unreachable: we passed a goldset, so the remap is always present
        typer.echo("[error] goldset remap missing from strip report", err=True)
        raise typer.Exit(code=2)

    baseline_items = [it for it in load_goldset(goldset) if split is None or it.split == split]
    stripped_items = load_goldset(drop_goldset)
    if split is not None:
        keep_ids = {it.id for it in baseline_items}
        stripped_items = [it for it in stripped_items if it.id in keep_ids]

    baseline_store: Retriever = _build_store(RagStore, cfg, corpus)
    stripped_store: Retriever = _build_store(RagStore, cfg, drop_corpus)
    report = audit_repeat_yield(
        baseline_items,
        stripped_items,
        baseline_store,
        stripped_store,
        dropped_ids=set(remap["dropped"]),
        rehomed_ids=set(remap["rehomed"]),
        k=k,
    )
    typer.echo(format_yield_report(report))
    report_path = work / YIELD_REPORT_NAME
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(f"[audit-repeat-yield] wrote report -> {report_path}")


def _build_store(rag_store: Any, cfg: "RunConfig", corpus_root: Path) -> Any:
    """One flat FAISS store over `corpus_root` under the config's chunking + pinned embedder."""
    return rag_store.build(
        corpus_root,
        cfg.strategy,
        cfg.chunk_size,
        cfg.chunk_overlap,
        cfg.embedding_model,
        mode="flat",
    )
