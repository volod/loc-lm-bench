"""One block, one crop, one Ollama vision reply. A transport failure is raised.

The reply is a JSON object and nothing else. Prose is not mined for a correction. Thinking is
off: a thinking model can spend the whole budget before it writes an edit.
"""

import base64
import json
import logging
import urllib.error
import urllib.request
from string import Template
from typing import Any

from llb.prep.book.correct.constants import OLLAMA_NUM_CTX, OLLAMA_NUM_PREDICT, OLLAMA_TIMEOUT_S
from llb.prep.book.correct.proposal import BlockPrompt, Proposal, ProposalResult
from llb.prep.book.survey.ollama_smoke import ollama_host

_LOG = logging.getLogger(__name__)

_PROMPT = """Correct one OCR block. Propose only the substitutions in the class list.
Do not rewrite, summarise, complete, reorder, or translate.
Offsets are character indexes into the block text.
Set visual to true only when the crop shows the replacement glyph.
If nothing is one of those substitutions, return {"edits":[]}.
Return one JSON object and no other text.

Classes: digit, sign, decimal, unit, index, latin-greek, bracket, equation-number.
An edit is {"start":0,"end":1,"replacement":"0","class":"digit","evidence":"crop shows 0","visual":true}.

Block:
${text}

Previous block:
${before}

Next block:
${after}

Previous page edge:
${previous_page}

Next page edge:
${next_page}
"""


class ModelRequestError(RuntimeError):
    """The correction model did not answer. The page is retried, not stored as no edits."""


class OllamaProposer:
    """Vision proposals from a local Ollama tag. `close` unloads the model."""

    def __init__(self, model: str, host: str | None = None) -> None:
        self._model = model
        self._host = host or ollama_host()
        self._used = False

    def propose(self, prompt: BlockPrompt) -> ProposalResult:
        text = self._chat(fill_prompt(prompt), prompt.crop_png, keep_alive=None)
        parsed = parse_reply(text)
        if parsed is None:
            _LOG.info(
                "correct block %s: model reply was not a structured edit list", prompt.block_id
            )
            return ProposalResult(unparsed=True)
        return ProposalResult(tuple(parsed))

    def unload(self) -> None:
        """Drop the model from Ollama when this batch is finished."""
        if not self._used:
            return
        try:
            self._chat(" ", b"", keep_alive=0)
        except ModelRequestError:
            _LOG.info("correct model %s: unload request failed", self._model)

    def _chat(self, content: str, crop_png: bytes, keep_alive: int | None) -> str:
        message: dict[str, Any] = {"role": "user", "content": content}
        if crop_png:
            message["images"] = [base64.b64encode(crop_png).decode("ascii")]
        payload: dict[str, Any] = {
            "model": self._model,
            "stream": False,
            "think": False,
            "messages": [message],
            "options": {
                "temperature": 0,
                "num_predict": OLLAMA_NUM_PREDICT,
                "num_ctx": OLLAMA_NUM_CTX,
            },
        }
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        request = urllib.request.Request(
            f"{self._host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        self._used = True
        try:
            with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT_S) as response:
                data = json.loads(response.read().decode("utf-8", "replace"))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise ModelRequestError(f"ollama request failed: {type(exc).__name__}") from exc
        return _reply_text(data)


def fill_prompt(prompt: BlockPrompt) -> str:
    """Substitute one block into the prompt. Braces in the OCR text stay braces."""
    return Template(_PROMPT).substitute(
        text=prompt.text,
        before=prompt.before,
        after=prompt.after,
        previous_page=prompt.previous_page,
        next_page=prompt.next_page,
    )


def parse_reply(text: str) -> list[Proposal] | None:
    """The edit list, or None when the reply is not a JSON object with an `edits` list."""
    payload = _object(text)
    if payload is None:
        return None
    edits = payload.get("edits")
    if not isinstance(edits, list):
        return None
    proposals: list[Proposal] = []
    for item in edits:
        proposal = _edit(item)
        if proposal is None:
            return None
        proposals.append(proposal)
    return proposals


def _reply_text(data: object) -> str:
    if not isinstance(data, dict):
        return ""
    reply = data.get("message") or {}
    if not isinstance(reply, dict):
        return ""
    content = str(reply.get("content") or "")
    if content.strip():
        return content
    thinking = reply.get("thinking")
    if isinstance(thinking, str):
        return thinking
    return ""


def _object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = [line for line in stripped.splitlines() if not line.strip().startswith("```")]
        stripped = "\n".join(lines).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(value, dict):
        return None
    return value


def _edit(item: object) -> Proposal | None:
    if not isinstance(item, dict):
        return None
    start, end = item.get("start"), item.get("end")
    replacement, confusion_class = item.get("replacement"), item.get("class")
    evidence, visual = item.get("evidence"), item.get("visual")
    if type(start) is not int or type(end) is not int:
        return None
    if not isinstance(replacement, str) or not isinstance(confusion_class, str):
        return None
    if not isinstance(evidence, str) or type(visual) is not bool:
        return None
    return Proposal(start, end, replacement, confusion_class, evidence, visual)
