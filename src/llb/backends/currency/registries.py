"""The two upstream registries a currency probe reads, and the recorded replay of their answers.

Both adapters answer the same narrow question -- what artifact names does this registry currently
offer under this namespace -- and both degrade the same way: a registry that does not answer
returns its reason, never an exception, because one unreachable registry must not cost the report
every other family's row.

Every fetch is timestamped at the moment the response arrives, so the report states when each
reading was taken rather than when it was printed. A cassette records those responses verbatim and
replays them, which is what makes the probe testable and runnable with no network at all.
"""

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from dataclasses import fields as dataclass_fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

OLLAMA = "ollama"
HUGGINGFACE = "huggingface"
REGISTRIES = (OLLAMA, HUGGINGFACE)

OLLAMA_LIBRARY_URL = "https://ollama.com/library"
HF_MODELS_URL = "https://huggingface.co/api/models"
HF_PAGE_LIMIT = 100

# The Ollama library index is HTML; every model on it is linked as `/library/<name>`.
_OLLAMA_LINK = '"/library/'
_OLLAMA_NAME_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_")

_TIMEOUT_S = 20.0
_USER_AGENT = "loc-lm-bench-currency-probe"


@dataclass(frozen=True)
class Response:
    """One registry response: what was asked, when it arrived, and either a body or a reason."""

    url: str
    read_at: str
    body: str | None = None
    error: str | None = None


Fetcher = Callable[[str], Response]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def live_fetch(url: str) -> Response:
    """GET one URL, returning the failure as a reason instead of raising it."""
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
            body = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return Response(url=url, read_at=_now(), error=f"HTTP {exc.code} {exc.reason}")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return Response(url=url, read_at=_now(), error=f"{type(exc).__name__}: {exc}")
    return Response(url=url, read_at=_now(), body=body)


def memoized(fetch: Fetcher) -> Fetcher:
    """One read per URL per report -- every family shares the single Ollama library index."""
    cache: dict[str, Response] = {}

    def cached(url: str) -> Response:
        if url not in cache:
            cache[url] = fetch(url)
        return cache[url]

    return cached


# --- recorded responses ------------------------------------------------------------------


class Cassette:
    """Registry responses recorded verbatim, so a report can be reproduced without a network."""

    def __init__(self, responses: dict[str, Response] | None = None):
        self.responses: dict[str, Response] = dict(responses or {})

    @classmethod
    def load(cls, path: Path | str) -> "Cassette":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        recorded = raw.get("responses", []) if isinstance(raw, dict) else []
        fields = {field.name for field in dataclass_fields(Response)}
        responses = [
            Response(**{k: v for k, v in item.items() if k in fields}) for item in recorded
        ]
        return cls({response.url: response for response in responses})

    def save(self, path: Path | str) -> None:
        payload = {
            "recorded_at": _now(),
            "responses": [asdict(response) for response in self.responses.values()],
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def fetch(self, url: str) -> Response:
        """Replay a recorded response; an unrecorded URL is a missing reading, not a live read."""
        recorded = self.responses.get(url)
        if recorded is None:
            return Response(url=url, read_at=_now(), error="not in the recorded responses")
        return recorded

    def recording(self, fetch: Fetcher = live_fetch) -> Fetcher:
        """A fetcher that answers live and keeps every response for `save`."""

        def record(url: str) -> Response:
            response = fetch(url)
            self.responses[url] = response
            return response

        return record


# --- adapters ----------------------------------------------------------------------------


def ollama_library_url() -> str:
    """The Ollama library index -- one page listing every model the library offers."""
    return OLLAMA_LIBRARY_URL


def hf_models_url(author: str, prefix: str = "") -> str:
    """The Hugging Face model API query for one author, newest first.

    One page is enough for a CURRENCY question: the sort is by last modification, and a generation
    that was just published is by definition among the most recently touched repos of its author.
    A page that truncates therefore drops old generations, never the newest one.
    """
    query = {
        "author": author,
        "limit": str(HF_PAGE_LIMIT),
        "sort": "lastModified",
        "direction": "-1",
    }
    if prefix:
        query["search"] = prefix
    return f"{HF_MODELS_URL}?{urllib.parse.urlencode(query)}"


def parse_ollama_library(body: str) -> tuple[str, ...]:
    """Every model name the library index links, in page order without repeats."""
    names: list[str] = []
    seen: set[str] = set()
    for chunk in body.split(_OLLAMA_LINK)[1:]:
        name = chunk.split('"', 1)[0]
        if not name or set(name) - _OLLAMA_NAME_CHARS or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return tuple(names)


def parse_hf_models(body: str) -> tuple[str, ...]:
    """The repo-name half of every model id the API returned (`Qwen/Qwen3.8-27B` -> `Qwen3.8-27B`)."""
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError(f"not JSON -- {exc}") from None
    if not isinstance(payload, list):
        raise ValueError("expected a JSON list of models")
    names: list[str] = []
    for entry in payload:
        model_id = entry.get("modelId") or entry.get("id") if isinstance(entry, dict) else None
        if model_id:
            names.append(str(model_id).split("/")[-1])
    return tuple(names)
