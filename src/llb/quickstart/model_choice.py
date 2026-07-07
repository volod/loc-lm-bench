"""Helpers for quickstart model-selection prompts."""

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _fmt_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_vram(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.0f} MiB"
    except (TypeError, ValueError):
        return "n/a"


def _candidates(data: dict[str, Any]) -> list[dict[str, Any]]:
    values = data.get("candidates")
    return (
        [entry for entry in values if isinstance(entry, dict)] if isinstance(values, list) else []
    )


def _selection_item(data: dict[str, Any], key: str) -> dict[str, Any] | None:
    selection = data.get("selection")
    if not isinstance(selection, dict):
        return None
    item = selection.get(key)
    return item if isinstance(item, dict) else None


def print_table(path: Path) -> None:
    data = _load(path)
    host = _dict(data.get("host"))
    tier = host.get("tier_gb", "?")
    total = host.get("total_mb", "?")
    gpu = host.get("gpu_name") or "planning budget"
    print(f"[models] benchmark host: tier={tier} GiB total_mb={total} gpu={gpu}")
    print("[models] ranked local candidates:")
    for index, item in enumerate(_candidates(data), start=1):
        print(
            "[models] "
            f"{index}. {item.get('model')} | backend={item.get('backend')} "
            f"objective={_fmt_float(item.get('objective'))} "
            f"tok_s={_fmt_float(item.get('tokens_per_s'), 1)} "
            f"peak_vram={_fmt_vram(item.get('peak_vram_mb'))} "
            f"top_k={item.get('top_k', 'n/a')} "
            f"recall={_fmt_float(item.get('recall_at_k'))} "
            f"n={item.get('n_cases', 'n/a')}"
        )
    recommended = _selection_item(data, "recommended_for_host")
    best = _selection_item(data, "best_quality")
    fastest = _selection_item(data, "fastest")
    if recommended:
        print(f"[models] recommended_for_host={recommended.get('model')}")
    if best:
        print(f"[models] best_quality={best.get('model')}")
    if fastest:
        print(f"[models] fastest={fastest.get('model')}")


def print_selection(path: Path, key: str) -> None:
    item = _selection_item(_load(path), key)
    if item is None or not item.get("model"):
        raise SystemExit(f"selection key not found: {key}")
    print(item["model"])


# The PDF quickstart drafter can use local Ollama or a locally launched vLLM server. The shell
# wrapper passes the subset actually available on the host; this default is for direct CLI use.
DRAFTER_BACKENDS = ("ollama", "vllm")


def _select_drafter(data: dict[str, Any], backends: list[str] | None = None) -> dict[str, Any]:
    """Return the best benchmark candidate the local draft endpoint can actually serve.

    Prefers `recommended_for_host` when its backend qualifies; otherwise falls back to the
    highest-ranked candidate with an allowed backend. Exits nonzero when none qualifies.
    """
    allowed = set(backends or DRAFTER_BACKENDS)
    recommended = _selection_item(data, "recommended_for_host")
    if recommended and recommended.get("backend") in allowed and recommended.get("model"):
        return recommended
    for item in _candidates(data):
        if item.get("backend") in allowed and item.get("model"):
            return item
    raise SystemExit(f"no ranked candidate with backend in {sorted(allowed)}")


def print_drafter(path: Path, backends: list[str] | None = None) -> None:
    print(_select_drafter(_load(path), backends)["model"])


def print_drafter_backend(path: Path, backends: list[str] | None = None) -> None:
    print(_select_drafter(_load(path), backends)["backend"])


def print_candidate(path: Path, index: int) -> None:
    candidates = _candidates(_load(path))
    if index < 1 or index > len(candidates):
        raise SystemExit(f"candidate index out of range: {index}")
    model = candidates[index - 1].get("model")
    if not model:
        raise SystemExit(f"candidate has no model: {index}")
    print(model)


def print_candidate_backend(path: Path, index: int) -> None:
    candidates = _candidates(_load(path))
    if index < 1 or index > len(candidates):
        raise SystemExit(f"candidate index out of range: {index}")
    backend = candidates[index - 1].get("backend")
    if not backend:
        raise SystemExit(f"candidate has no backend: {index}")
    print(backend)


def print_speed(path: Path, model: str) -> None:
    for item in _candidates(_load(path)):
        if item.get("model") == model:
            value: Any = item.get("tokens_per_s")
            try:
                print(f"{float(value):.3f}")
            except (TypeError, ValueError):
                print("0")
            return
    print("0")


def print_count(path: Path) -> None:
    print(len(_candidates(_load(path))))


HOST_GEMMA4_FIELDS = (
    "target",
    "model",
    "backend",
    "tier-gb",
    "gpu-memory-utilization",
    "max-model-len",
    "cpu-offload-gb",
    "kv-offloading-size-gb",
)


def print_host_gemma4(
    field: str, gpu_gb: int | None = None, min_context_tokens: int | None = None
) -> None:
    from llb.inference.generate import select_host_gemma4_target

    row = select_host_gemma4_target(gpu_gb=gpu_gb, min_context_tokens=min_context_tokens)
    key = field.replace("-", "_")
    value = row.get(key)
    if isinstance(value, float):
        print(f"{value:g}")
        return
    print("" if value is None else value)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    table = sub.add_parser("table")
    table.add_argument("json", type=Path)

    selection = sub.add_parser("selection")
    selection.add_argument("json", type=Path)
    selection.add_argument("key")

    drafter = sub.add_parser("drafter")
    drafter.add_argument("json", type=Path)
    drafter.add_argument("backends", nargs="*", default=list(DRAFTER_BACKENDS))

    drafter_backend = sub.add_parser("drafter-backend")
    drafter_backend.add_argument("json", type=Path)
    drafter_backend.add_argument("backends", nargs="*", default=list(DRAFTER_BACKENDS))

    candidate = sub.add_parser("candidate")
    candidate.add_argument("json", type=Path)
    candidate.add_argument("index", type=int)

    candidate_backend = sub.add_parser("candidate-backend")
    candidate_backend.add_argument("json", type=Path)
    candidate_backend.add_argument("index", type=int)

    speed = sub.add_parser("speed")
    speed.add_argument("json", type=Path)
    speed.add_argument("model")

    count = sub.add_parser("count")
    count.add_argument("json", type=Path)

    host_gemma4 = sub.add_parser("host-gemma4")
    host_gemma4.add_argument("field", choices=HOST_GEMMA4_FIELDS)
    host_gemma4.add_argument("--gpu-gb", type=int, default=None)
    host_gemma4.add_argument("--min-context-tokens", type=int, default=None)

    args = parser.parse_args(argv)
    if args.command == "table":
        print_table(args.json)
    elif args.command == "selection":
        print_selection(args.json, args.key)
    elif args.command == "drafter":
        print_drafter(args.json, args.backends or None)
    elif args.command == "drafter-backend":
        print_drafter_backend(args.json, args.backends or None)
    elif args.command == "candidate":
        print_candidate(args.json, args.index)
    elif args.command == "candidate-backend":
        print_candidate_backend(args.json, args.index)
    elif args.command == "speed":
        print_speed(args.json, args.model)
    elif args.command == "count":
        print_count(args.json)
    elif args.command == "host-gemma4":
        print_host_gemma4(
            args.field,
            gpu_gb=args.gpu_gb,
            min_context_tokens=args.min_context_tokens,
        )


if __name__ == "__main__":
    main()
