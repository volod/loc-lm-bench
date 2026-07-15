"""Load tuning-split items and apply the deterministic teacher-answer quality gate."""

from pathlib import Path
from typing import cast

from llb.core.contracts.common import ChatMessage
from llb.core.contracts.rag import ChunkRecord
from llb.eval import common as eval_common
from llb.eval.graph import build_messages
from llb.finetune.distill.model import GatedTeacherRecord, TeacherResponse
from llb.goldset.schema import GoldItem, load_goldset
from llb.scoring.correctness import answer_correctness


def _load_items(goldset_path: Path, *, split: str, limit: int | None) -> list[GoldItem]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")
    items = [item for item in load_goldset(goldset_path) if item.verified and item.split == split]
    items.sort(key=lambda item: item.id)
    if limit is not None:
        items = items[:limit]
    if not items:
        raise SystemExit(f"[distill] no verified {split!r} items in {goldset_path}")
    return items


def _gate_responses(
    items: list[GoldItem], responses: list[TeacherResponse], *, gate: float
) -> list[GatedTeacherRecord]:
    by_id = {response.item_id: response for response in responses}
    missing = [item.id for item in items if item.id not in by_id]
    if missing:
        raise ValueError(f"teacher did not return answers for item ids: {', '.join(missing)}")
    records: list[GatedTeacherRecord] = []
    for item in items:
        response = by_id[item.id]
        corr = answer_correctness(response.answer, item.reference_answer)
        gate_score = float(corr["score"])
        accepted = response.status == eval_common.OK and gate_score >= gate
        records.append(
            GatedTeacherRecord(
                item=item,
                answer=response.answer,
                status=response.status,
                gate_score=gate_score,
                token_f1=float(corr["token_f1"]),
                exact=float(corr["exact"]),
                contains=float(corr["contains"]),
                accepted=accepted,
                context=response.context,
                retrieved=response.retrieved,
                messages=response.messages,
            )
        )
    return records


def _messages_for_record(record: GatedTeacherRecord) -> list[ChatMessage]:
    if record.messages:
        return [cast(ChatMessage, dict(message)) for message in record.messages]
    context = record.context or _context_from_record(record)
    return build_messages(record.item.question, context)


def _context_from_record(record: GatedTeacherRecord) -> str:
    chunks: list[ChunkRecord] = list(record.retrieved)
    if not chunks:
        chunks = [
            {
                "doc_id": span.doc_id,
                "char_start": span.char_start,
                "char_end": span.char_end,
                "text": span.text,
            }
            for span in record.item.source_spans
        ]
    return eval_common.format_context(chunks)
