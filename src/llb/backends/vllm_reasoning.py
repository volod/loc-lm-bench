"""vLLM reasoning-output controls: the request fields, and the verdict that says they are safe.

vLLM's OpenAI-compatible endpoint carries no `think` flag the way Ollama's native API does.
What it exposes instead is `chat_template_kwargs` -- a mapping handed to the model's Jinja chat
template, where a Qwen-style reasoning template reads `enable_thinking` -- plus two request-level
fields of vLLM's own schema, `include_reasoning` and `reasoning_effort`. None of the three is in
the OpenAI schema, so they travel in `extra_body`.

Whether a given vLLM ACCEPTS the two request-level fields depends on its version: a server that
models its request body strictly rejects an unknown field with a 400, and a launcher that breaks
every vLLM run is worse than one that sends no flag. So the fields are gated twice -- an explicit
opt-in by the caller, and then a PROBE that sends them once against the live server and records
which of them this vLLM took. The verdict is cached per vLLM version under `$DATA_DIR`, so the
probe costs one 1-token generation on the first launch of a given version and nothing after that.

The probe is injectable (a send callable -> error token or None), so the ladder and the
persistence are unit-testable without vLLM, CUDA, or a server.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TypedDict, cast

from llb.core.paths import resolve_data_dir

_LOG = logging.getLogger(__name__)

# How much of the reasoning-control body a server accepts, most complete first. `full` is the
# template kwarg plus vLLM's own request fields; `template_only` is the template kwarg alone (the
# portable half -- an unknown Jinja variable is ignored by the template rather than rejected);
# `none` means send nothing, which is the shipped behavior.
FIELDS_FULL = "full"
FIELDS_TEMPLATE_ONLY = "template_only"
FIELDS_NONE = "none"

# The ladder the probe walks, in order.
PROBE_LADDER = (FIELDS_FULL, FIELDS_TEMPLATE_ONLY)

# vLLM's `reasoning_effort` value that asks for no deliberation at all.
REASONING_EFFORT_NONE = "none"

# The probe is one 1-token generation against a server that has already reported ready, so
# this bounds a hung request rather than a cold start.
PROBE_TIMEOUT_S = 60.0


class ThinkingVerdict(TypedDict):
    """Which reasoning-control fields one vLLM version accepted, and how that was established."""

    fields: str  # full | template_only | none
    vllm_version: str | None
    detail: str
    checked_at: str  # ISO-8601 UTC, for provenance


def reasoning_extra_body(think: bool | None, *, fields: str = FIELDS_FULL) -> dict[str, object]:
    """The `extra_body` that asks vLLM to suppress (or allow) the model's reasoning block.

    `think=None` or `fields="none"` yields an EMPTY body -- the caller then sends no `extra_body`
    at all, which is what keeps a non-reasoning run byte-identical to the shipped request shape.
    """
    if think is None or fields == FIELDS_NONE:
        return {}
    body: dict[str, object] = {"chat_template_kwargs": {"enable_thinking": think}}
    if fields == FIELDS_FULL and think is False:
        body["include_reasoning"] = False
        body["reasoning_effort"] = REASONING_EFFORT_NONE
    return body


# extra_body (empty == send none) -> normalized error token, or None when the call succeeded.
ProbeSend = Callable[[dict[str, object]], str | None]


def _verdict(fields: str, detail: str, vllm_version: str | None) -> ThinkingVerdict:
    return {
        "fields": fields,
        "vllm_version": vllm_version,
        "detail": detail,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def probe_thinking_fields(send: ProbeSend, *, vllm_version: str | None) -> ThinkingVerdict | None:
    """Walk the ladder against a live server and return the most complete body it accepted.

    Returns None when the probe is INCONCLUSIVE: a control call carrying no extras failed too, so
    the server -- not the fields -- is what refused, and recording "unsupported" from that would
    pin a wrong verdict for the whole version. An inconclusive probe is not persisted.
    """
    for fields in PROBE_LADDER:
        error = send(reasoning_extra_body(False, fields=fields))
        if error is None:
            return _verdict(
                fields, f"vLLM accepted the {fields} reasoning-control body", vllm_version
            )
        _LOG.debug("[vllm] reasoning-control probe rejected %s body: %s", fields, error)
    control = send({})
    if control is not None:
        _LOG.warning(
            "[vllm] reasoning-control probe inconclusive: a request with no extras also failed (%s)",
            control,
        )
        return None
    return _verdict(
        FIELDS_NONE,
        "vLLM rejected every reasoning-control body while a plain request succeeded",
        vllm_version,
    )


def resolve_extra_body(
    client: object,
    model: str,
    *,
    data_dir: Path | None = None,
    installed: str | None = None,
) -> tuple[dict[str, object], str]:
    """The `extra_body` a suppressed-thinking run should send to `client`, and its level.

    Probes the live server through the shared chat seam -- one 1-token generation per ladder
    rung, at most three -- and caches the verdict per vLLM version. An empty body means send no
    `extra_body` at all.
    """
    from llb.backends.openai_client import chat_once
    from llb.core.contracts.common import ChatMessage

    probe: list[ChatMessage] = [{"role": "user", "content": "hi"}]

    def send(extra_body: dict[str, object]) -> str | None:
        return chat_once(
            client,
            model,
            probe,
            max_tokens=1,
            temperature=0.0,
            timeout=PROBE_TIMEOUT_S,
            extra_body=extra_body or None,
        ).error

    verdict = resolve_thinking_fields(send, data_dir=data_dir, installed=installed)
    return reasoning_extra_body(False, fields=verdict["fields"]), verdict["fields"]


def vllm_version() -> str | None:
    """The installed vLLM version, or None when vLLM is not importable (tests, base install)."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("vllm")
    except PackageNotFoundError:
        return None


def verdict_path(data_dir: Path | None = None) -> Path:
    base = data_dir if data_dir is not None else resolve_data_dir()
    return base / "llb" / "preflight" / "vllm_reasoning.json"


def save_verdict(verdict: ThinkingVerdict, data_dir: Path | None = None) -> Path:
    path = verdict_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    return path


def load_verdict(data_dir: Path | None = None) -> ThinkingVerdict | None:
    """The persisted probe verdict, or None when none has been recorded (best-effort)."""
    try:
        data = json.loads(verdict_path(data_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(data, dict) and data.get("fields") in (
        FIELDS_FULL,
        FIELDS_TEMPLATE_ONLY,
        FIELDS_NONE,
    ):
        return cast(ThinkingVerdict, data)
    return None


def verdict_is_current(verdict: ThinkingVerdict | None, installed: str | None) -> bool:
    """True when a verdict exists AND was recorded against the vLLM now installed.

    A vLLM upgrade is exactly what can add or remove a request field, so a version change
    invalidates the verdict and the probe re-runs on the next launch.
    """
    if verdict is None:
        return False
    recorded = verdict.get("vllm_version")
    return recorded is None or installed is None or recorded == installed


def resolve_thinking_fields(
    send: ProbeSend, *, data_dir: Path | None = None, installed: str | None = None
) -> ThinkingVerdict:
    """The cached verdict for this vLLM, probing once when there is none (never raises)."""
    version = installed if installed is not None else vllm_version()
    cached = load_verdict(data_dir)
    if verdict_is_current(cached, version) and cached is not None:
        return cached
    probed = probe_thinking_fields(send, vllm_version=version)
    if probed is None:
        return _verdict(
            FIELDS_NONE, "reasoning-control probe inconclusive; sending no extras", version
        )
    save_verdict(probed, data_dir)
    _LOG.info("[vllm] reasoning-control probe: %s (%s)", probed["fields"], probed["detail"])
    return probed
