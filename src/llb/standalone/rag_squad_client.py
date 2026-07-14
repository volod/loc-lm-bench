"""Focused rag squad client implementation."""

from __future__ import annotations
import json
import re
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, cast

RAG_SERVICE_URL = os.environ.get("RAG_SERVICE_URL", "http://localhost:8000/query")

SERVICE_NAME = os.environ.get("RAG_SERVICE_NAME", "rag-service")

RAG_API_KEY = os.environ.get("RAG_API_KEY", "")

REQUEST_TIMEOUT_S = float(os.environ.get("RAG_TIMEOUT_S", "120"))

RETRY_ATTEMPTS = int(os.environ.get("RAG_RETRIES", "3"))

RETRY_BACKOFF_S = float(os.environ.get("RAG_RETRY_BACKOFF_S", "2"))

NOT_FOUND = "НЕВІДОМО"

INSTRUCTIONS = (
    "Відповідай на запитання виключно за знайденим контекстом.\n"
    "1. Використовуй ЛИШЕ знайдений контекст, не додавай зовнішніх знань і нічого не вигадуй.\n"
    "2. Дай найкоротшу точну відповідь — дослівний фрагмент із контексту, без пояснень.\n"
    f"3. Якщо відповіді немає в контексті — не відповідай, поверни рівно одне слово: {NOT_FOUND}.\n"
    "4. Відповідай українською мовою."
)


def build_request(question: str) -> urllib.request.Request:
    """Build the HTTP request sent to the RAG service. ADJUST field names to your API.

    Default assumes a JSON service of the shape:
        request : {"query": <question>, "instructions": <guidance>}
        response: {"answer": <text>}                      (see `parse_answer`)
    Drop "instructions" if your service does not accept per-request guidance.
    """
    body = json.dumps({"query": question, "instructions": INSTRUCTIONS}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if RAG_API_KEY:
        headers["Authorization"] = f"Bearer {RAG_API_KEY}"
    return urllib.request.Request(RAG_SERVICE_URL, data=body, headers=headers, method="POST")


def parse_answer(payload: dict[str, Any]) -> str:
    """Extract the answer text from the service response. ADJUST to your API.

    Common shapes:
        {"answer": "..."}                            -> payload["answer"]
        {"result": {"text": "..."}}                  -> payload["result"]["text"]
        {"choices": [{"message": {"content": ...}}]} -> OpenAI-compatible chat
    """
    return cast(str, payload["answer"])


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def clean_answer(text: str) -> str:
    """Strip reasoning-model <think> blocks and surrounding whitespace."""
    return _THINK_RE.sub("", text).strip()


RETRYABLE = (urllib.error.URLError, TimeoutError, ConnectionError)


def query_rag_service(question: str) -> str:
    """Ask the RAG service one question and return the cleaned answer, retrying transient errors."""
    last_exc: Exception | None = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            req = build_request(question)
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
                payload = cast(dict[str, Any], json.loads(resp.read().decode("utf-8")))
            return clean_answer(parse_answer(payload))
        except RETRYABLE as exc:
            last_exc = exc
            if attempt < RETRY_ATTEMPTS:
                wait = RETRY_BACKOFF_S * attempt
                log(
                    f"  transient error (attempt {attempt}/{RETRY_ATTEMPTS}): {exc} — retrying in {wait:.0f}s"
                )
                time.sleep(wait)
    if last_exc is not None:
        raise last_exc  # exhausted retries
    raise RuntimeError("RAG service query failed without an attempt")


def log(msg: str) -> None:
    """Progress logging to stderr so it never pollutes the JSONL on stdout redirection."""
    print(msg, file=sys.stderr, flush=True)
