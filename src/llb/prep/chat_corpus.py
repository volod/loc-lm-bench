"""category expansion chat-period -- chat-log-shaped synthetic planter + REAL chat-corpus ingestion.

chat-period analysis is text-analysis over CHAT-LOG documents, so it reuses the text analysis planted-label
schema + the `bench.text_analysis` runner unchanged. This module adds the two chat-specific
producers the plan calls for:

  * a chat-log-shaped SYNTHETIC planter -- a generated UA conversation + planted labels (reuses the
    `prepare_text_analysis_corpus` generate -> parse -> ground -> bundle flow with a chat prompt);
  * REAL chat-corpus ingestion -- parse an exported chat log into a chat-shaped document, then DRAFT
    text-analysis labels FROM it with a LOCAL completion (NO egress, per the OQ-egress decision),
    grounded against the rendered doc via the shared `plant_labels`.

Both write a self-contained bundle (`corpus/` + `text_analysis_labels.jsonl` + `provenance.json`)
the runner scores via the REAL path (`--real-corpus`, `synthetic=false`), reported SEPARATELY from
synthetic. `complete` is injectable, so rendering / parsing / grounding are unit-tested with no
network or key; the real path takes a LOCAL `complete` (the CLI builds an OpenAI-compatible one), so
the real chat logs never leave the host.
"""

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from llb.core.contracts import PlantedLabelRecord
from llb.prep.frontier import LLMComplete, parse_json_block
from llb.prep.text_analysis_corpus import DEFAULT_KINDS, _count_by_kind, plant_labels
from llb.prompts import render_text
from llb.scoring import text_analysis as ta

_LOG = logging.getLogger(__name__)

_DEFAULT_SPEAKER = "Учасник"
# Keys a chat export may use for the speaker / the message body.
_SPEAKER_KEYS = ("speaker", "from", "sender", "role", "name", "author", "user")
_TEXT_KEYS = ("text", "content", "message", "body")


def _message_fields(message: dict[str, Any]) -> tuple[str, str]:
    """Extract (speaker, text) from one message across common chat-export shapes."""
    speaker = next((str(message[k]) for k in _SPEAKER_KEYS if message.get(k)), _DEFAULT_SPEAKER)
    raw_text = next((message[k] for k in _TEXT_KEYS if k in message), "")
    if isinstance(raw_text, list):  # Telegram splits text into runs; join their text
        raw_text = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in raw_text
        )
    return speaker.strip(), str(raw_text).strip()


def render_chat_log(messages: Iterable[dict[str, Any]]) -> str:
    """Render a message list into a chat-log document (`Speaker: text` lines, blank-line separated)."""
    lines = []
    for message in messages:
        speaker, text = _message_fields(message)
        if text:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def load_chat_conversations(path: Path | str) -> list[tuple[str, list[dict[str, Any]]]]:
    """Parse an exported chat corpus into `[(chat_id, messages), ...]`.

    Accepts a JSON array of conversations (`[{id?, messages:[...]}]`), a single Telegram-style export
    (`{messages:[...]}` or `{chats:{list:[{messages:[...]}]}}`), or JSONL of messages (one chat).
    """
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        raw: Any = json.loads(text)
    except json.JSONDecodeError:  # JSONL of messages -> a single conversation
        messages = [json.loads(line) for line in text.splitlines() if line.strip()]
        return [("chat-000", messages)]

    if isinstance(raw, dict) and isinstance(raw.get("chats"), dict):  # Telegram full export
        raw = raw["chats"].get("list", [])
    if isinstance(raw, dict):  # a single conversation export
        raw = [raw]
    conversations: list[tuple[str, list[dict[str, Any]]]] = []
    for i, convo in enumerate(raw if isinstance(raw, list) else []):
        parsed = _conversation_from(i, convo)
        if parsed is not None:
            conversations.append(parsed)
    return conversations


def _conversation_from(index: int, convo: Any) -> tuple[str, list[dict[str, Any]]] | None:
    """`(chat_id, messages)` from one exported conversation record, or None if malformed."""
    if not isinstance(convo, dict):
        return None
    messages = convo.get("messages", [])
    if not isinstance(messages, list):
        return None
    chat_id = str(convo.get("id", convo.get("name", f"chat-{index:03d}")))
    return chat_id, [m for m in messages if isinstance(m, dict)]


def chat_doc_prompt(topic: str, n_per_kind: int, kinds: tuple[str, ...]) -> str:
    """Synthetic CHAT-LOG planter prompt: a UA conversation (not an essay) + planted labels."""
    asks = "\n".join(f'  - "{kind}"' for kind in kinds)
    return render_text(
        "prep.chat_corpus.doc",
        {"topic": topic, "n_per_kind": n_per_kind, "asks": asks},
    )


def chat_label_draft_prompt(
    doc_id: str, chat_text: str, n_per_kind: int, kinds: tuple[str, ...]
) -> str:
    """Draft text-analysis labels FROM a real chat-log document (no doc generation, no egress)."""
    asks = "\n".join(f'  - "{kind}"' for kind in kinds)
    return render_text(
        "prep.chat_corpus.label_draft",
        {
            "doc_id": doc_id,
            "chat_text": chat_text,
            "n_per_kind": n_per_kind,
            "asks": asks,
        },
    )


def ingest_chat_corpus(
    conversations: list[tuple[str, list[dict[str, Any]]]],
    *,
    complete: LLMComplete,
    kinds: tuple[str, ...] = DEFAULT_KINDS,
    n_per_kind: int = 2,
    out_dir: Path | str | None = None,
    source: str = "real-chat",
) -> tuple[dict[str, str], list[PlantedLabelRecord]]:
    """Ingest a REAL chat corpus: render each conversation, draft grounded labels with a LOCAL
    `complete` (no egress), and write a `synthetic: false` bundle for the runner's real path."""
    unknown = [kind for kind in kinds if kind not in ta.ALL_KINDS]
    if unknown:
        raise ValueError(f"unknown text-analysis kinds: {unknown}")
    out_dir = Path(out_dir) if out_dir is not None else None
    corpus_dir = out_dir / "corpus" if out_dir is not None else None

    docs: dict[str, str] = {}
    records: list[PlantedLabelRecord] = []
    for i, (chat_id, messages) in enumerate(conversations):
        doc_id = f"chat-{i:03d}"
        document = render_chat_log(messages)
        if not document.strip():
            continue
        docs[doc_id] = document
        raw_labels = _draft_chat_labels(chat_id, document, complete, n_per_kind, kinds)
        records += plant_labels(doc_id, document, raw_labels)
        if corpus_dir is not None:
            corpus_dir.mkdir(parents=True, exist_ok=True)
            (corpus_dir / f"{doc_id}.md").write_text(document, encoding="utf-8")

    if out_dir is not None:
        _write_chat_bundle(out_dir, docs, records, source)
    return docs, records


def _draft_chat_labels(
    chat_id: str,
    document: str,
    complete: LLMComplete,
    n_per_kind: int,
    kinds: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Draft raw label dicts for one chat document; unparseable output keeps the doc unlabeled."""
    raw = complete(chat_label_draft_prompt(chat_id, document, n_per_kind, kinds))
    try:
        parsed = parse_json_block(raw)
    except json.JSONDecodeError:
        _LOG.warning("[ingest-chat] unparseable labels for %s; doc kept, no labels", chat_id)
        return []
    if not isinstance(parsed, list):
        return []
    return [entry for entry in parsed if isinstance(entry, dict)]


def _write_chat_bundle(
    out_dir: Path,
    docs: dict[str, str],
    records: list[PlantedLabelRecord],
    source: str,
) -> None:
    """Persist the labels JSONL + `synthetic: false` provenance beside the staged corpus."""
    (out_dir / "text_analysis_labels.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )
    (out_dir / "provenance.json").write_text(
        json.dumps(
            {
                "kind": "real-chat-corpus",
                "synthetic": False,  # REAL chat logs -> reported separately from synthetic
                "egress": "none",  # drafted with a LOCAL completion (OQ-egress: no egress)
                "source": source,
                "n_docs": len(docs),
                "n_labels": len(records),
                "labels_by_kind": _count_by_kind(records),
                "corpus_root": "corpus",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _LOG.info("[ingest-chat] %d chat docs, %d labels -> %s", len(docs), len(records), out_dir)


def prepare_synthetic_chat_corpus(
    topics: list[str],
    *,
    planter_model: str,
    judge_model: str,
    kinds: tuple[str, ...] = DEFAULT_KINDS,
    n_per_kind: int = 2,
    complete: LLMComplete | None = None,
    out_dir: Path | str | None = None,
) -> tuple[dict[str, str], list[PlantedLabelRecord]]:
    """Generate a SYNTHETIC chat-log corpus + planted labels (reuses the text-analysis flow with the
    chat-shaped prompt; tagged `synthetic: true`, reported separately from the real chat corpus)."""
    from llb.prep.text_analysis_corpus import prepare_text_analysis_corpus

    return prepare_text_analysis_corpus(
        topics,
        planter_model=planter_model,
        judge_model=judge_model,
        kinds=kinds,
        n_per_kind=n_per_kind,
        complete=complete,
        out_dir=out_dir,
        prompt_builder=chat_doc_prompt,
        provenance_kind="synthetic-chat",
    )
