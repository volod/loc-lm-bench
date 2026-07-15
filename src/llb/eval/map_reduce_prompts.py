"""Focused map reduce prompts implementation."""

from llb.core.contracts.common import ChatMessage

from llb.prompts.registry import render_chat, render_text

NO_INFO_MARKER = "(немає інформації)"

MAP_SYSTEM_PROMPT = render_text("eval.map_reduce.map_system", {"no_info_marker": NO_INFO_MARKER})

REDUCE_SYSTEM_PROMPT = render_text(
    "eval.map_reduce.reduce_system", {"no_info_marker": NO_INFO_MARKER}
)


def split_document(document: str, max_chars: int, overlap: int) -> list[str]:
    """Split `document` into overlapping char windows (offset-free; map-reduce scores the
    synthesized answer, not spans). Pure and dependency-free, mirroring the always-available
    `fixed` chunker. `overlap` is clamped below `max_chars` so the window always advances."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    text = document.strip()
    if not text:
        return []
    overlap = max(0, min(overlap, max_chars - 1))
    step = max_chars - overlap
    segments: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        segments.append(text[start : start + max_chars])
        start += step
    return segments


def build_map_messages(question: str, segment: str) -> list[ChatMessage]:
    return render_chat(
        "eval.map_reduce.map_chat",
        {"no_info_marker": NO_INFO_MARKER, "question": question, "segment": segment},
    )


def build_reduce_messages(question: str, partials: list[str]) -> list[ChatMessage]:
    joined = "\n".join(f"[{i}] {p}" for i, p in enumerate(partials, 1))
    return render_chat(
        "eval.map_reduce.reduce_chat",
        {"no_info_marker": NO_INFO_MARKER, "question": question, "partials": joined},
    )


def is_no_info(partial: str) -> bool:
    """True when a map partial declined to answer from its segment."""
    return NO_INFO_MARKER in partial or not partial.strip()


def map_text_prompt(question: str, segment: str) -> str:
    """A single-string MAP prompt (the category suite categories drive map-reduce via a `complete: str->str`)."""
    return render_text(
        "eval.map_reduce.map_text",
        {"system_prompt": MAP_SYSTEM_PROMPT, "question": question, "segment": segment},
    )


def reduce_text_prompt(question: str, partials: list[str]) -> str:
    """A single-string REDUCE prompt over the surviving map partials."""
    joined = "\n".join(f"[{i}] {p}" for i, p in enumerate(partials, 1))
    return render_text(
        "eval.map_reduce.reduce_text",
        {"system_prompt": REDUCE_SYSTEM_PROMPT, "question": question, "partials": joined},
    )
