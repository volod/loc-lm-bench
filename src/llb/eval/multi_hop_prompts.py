"""Focused multi hop prompts implementation."""

from llb.core.contracts.common import ChatMessage
from llb.core.contracts.rag import ChunkRecord

from llb.eval.common import format_context

from llb.prompts.registry import render_chat, render_text

CONTINUE = "continue"

STOP = "stop"

DONE_MARKER = "ГОТОВО"  # enough gathered -> synthesize the answer

NEXT_MARKER = "ДАЛІ:"  # need more -> the text after the colon is the next sub-query

CONTROLLER_SYSTEM_PROMPT = render_text(
    "eval.multi_hop.controller_system",
    {"done_marker": DONE_MARKER, "next_marker": NEXT_MARKER},
)

ANSWER_SYSTEM_PROMPT = render_text("eval.multi_hop.answer_system")

DEFAULT_MAX_HOPS = 3


def parse_controller(text: str) -> tuple[str, str]:
    """Map a controller reply to (decision, next_subquery).

    `DONE_MARKER` -> (STOP, ""); `NEXT_MARKER <q>` -> (CONTINUE, q). Anything else (including an
    empty reply) is treated as STOP so a malformed controller turn ends the loop safely rather
    than spinning; `max_hops` is the hard bound regardless.
    """
    stripped = (text or "").strip()
    if not stripped:
        return STOP, ""
    upper = stripped.upper()
    idx = upper.find(NEXT_MARKER)
    if idx >= 0:
        remainder = stripped[idx + len(NEXT_MARKER) :].strip().splitlines()
        subquery = remainder[0].strip() if remainder else ""
        if subquery:
            return CONTINUE, subquery
    return STOP, ""


def _chunk_key(chunk: ChunkRecord) -> str:
    """Stable identity for de-duplicating chunks gathered across hops."""
    chunk_id = chunk.get("chunk_id")
    if chunk_id:
        return str(chunk_id)
    return f"{chunk.get('doc_id', '?')}:{chunk.get('char_start')}:{chunk.get('char_end')}"


def build_controller_messages(question: str, gathered: list[ChunkRecord]) -> list[ChatMessage]:
    facts = format_context(gathered) if gathered else "(поки що нічого не знайдено)"
    return render_chat(
        "eval.multi_hop.controller_chat",
        {
            "done_marker": DONE_MARKER,
            "next_marker": NEXT_MARKER,
            "question": question,
            "facts": facts,
        },
    )


def build_answer_messages(question: str, gathered: list[ChunkRecord]) -> list[ChatMessage]:
    facts = format_context(gathered)
    return render_chat(
        "eval.multi_hop.answer_chat",
        {"question": question, "facts": facts},
    )
