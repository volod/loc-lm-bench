"""The processing report: a human-readable skeleton later stages fill in.

It is rendered from the page manifest and the config, never accumulated separately, so it can
never disagree with the ledger. Sections no stage feeds yet say so instead of being omitted.
"""

from pathlib import Path

from llb.core.contracts.data_prep.book import BookOcrRunConfig, BookPageManifest
from llb.core.fsutil import atomic_write_text
from llb.prep.book.constants import (
    METADATA_DIRNAME,
    PAGE_STATUSES,
    PROCESSING_REPORT_FILENAME,
    STATUS_FAILED,
)
from llb.prep.book.ledger import REUSED

_NOT_YET = "Not measured: no stage of this run produces it yet."


def _stage_table(manifest: BookPageManifest) -> list[str]:
    columns = [*PAGE_STATUSES, REUSED]
    lines = ["| Stage | " + " | ".join(columns) + " |", "| --- |" + " --- |" * len(columns)]
    for name in manifest.stage_names:
        counts = manifest.summary.get(name, {})
        lines.append(f"| `{name}` | " + " | ".join(str(counts.get(c, 0)) for c in columns) + " |")
    return lines


def _failures(manifest: BookPageManifest) -> list[str]:
    rows = [
        f"- {page.page_id} (PDF page {page.pdf_page}) `{state.stage}` after {state.attempts} "
        f"attempt(s): {state.error}"
        for page in manifest.pages
        for state in page.stages
        if state.status == STATUS_FAILED
    ]
    return rows or ["None."]


def render_report(config: BookOcrRunConfig, manifest: BookPageManifest) -> str:
    source = config.source
    stages = ", ".join(
        f"`{spec.name}`" + (f" ({spec.model})" if spec.model else "") for spec in config.stages
    )
    lines = [
        f"# Book recovery run {config.run_id}",
        "",
        "## Source",
        "",
        f"- File: `{Path(source.original_path).name}` ({source.size_bytes} bytes, "
        f"{source.page_count} PDF pages)",
        f"- SHA-256: `{source.sha256}`",
        f"- Held as: {source.policy}",
        "",
        "## Selection",
        "",
        f"- Pages: `{config.pages_spec}` -> {manifest.n_pages} page(s) in batches of "
        f"{config.batch_pages}",
        f"- Stages: {stages}",
        f"- Attempts per stage: at most {config.max_attempts}",
        f"- Reused from: {config.reuse_from or 'nothing'}",
        "",
        "## Stage status",
        "",
        *_stage_table(manifest),
        "",
        "## Failed pages",
        "",
        *_failures(manifest),
        "",
        "## Recognition quality",
        "",
        _NOT_YET,
        "",
        "## Review queue",
        "",
        _NOT_YET,
        "",
    ]
    return "\n".join(lines)


def write_report(run_dir: Path, config: BookOcrRunConfig, manifest: BookPageManifest) -> Path:
    path = run_dir / METADATA_DIRNAME / PROCESSING_REPORT_FILENAME
    atomic_write_text(path, render_report(config, manifest))
    return path
