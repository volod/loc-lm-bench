"""Human-readable and machine-readable statistics for comparison.json artifacts."""

import json
from pathlib import Path
from typing import cast

from llb.goldset.verify_acceptance import acceptance_report
from llb.goldset.verify_base import load_worksheet


def load_comparison(path: Path | str) -> dict[str, object]:
    return cast(dict[str, object], json.loads(Path(path).read_text(encoding="utf-8")))


def comparison_statistics(report: dict[str, object]) -> dict[str, object]:
    lanes = cast(dict[str, dict[str, object]], report["lanes"])
    order = cast(list[str], report.get("lane_order") or list(lanes))
    lane_stats: dict[str, dict[str, object]] = {}
    for name in order:
        lane = lanes[name]
        verification = cast(dict[str, object], lane["verify_sample"])
        worksheet = Path(str(verification["worksheet"]))
        review = acceptance_report(load_worksheet(worksheet)[0]) if worksheet.is_file() else {}
        bundle = Path(str(lane["bundle"]))
        provenance = cast(
            dict[str, object], json.loads((bundle / "provenance.json").read_text(encoding="utf-8"))
        )
        endpoint = cast(dict[str, object], provenance["endpoint"])
        stages = cast(dict[str, object], provenance["stages"])
        drafting = cast(dict[str, object], cast(dict[str, object], endpoint["stages"])["drafting"])
        lane_stats[name] = {
            "model": drafting["model"],
            "seeds": lane["seeds"],
            "draft_attempts": stages["draft_attempts"],
            "draft_parsed": stages["draft_parsed"],
            "parse_rate": lane["parse_rate"],
            "kept": lane["kept"],
            "kept_yield": lane["kept_yield"],
            "calibration_passed": cast(dict[str, object], lane["gates"]).get("passed", False),
            "calls": drafting["calls"],
            "prompt_tokens": drafting["total_prompt_tokens"],
            "completion_tokens": drafting["total_completion_tokens"],
            "latency_s": drafting["total_latency_s"],
            "human_decided": review.get("decided", verification["decided"]),
            "human_accept_rate": (
                float(cast(int, review["accepted"])) / float(cast(int, review["decided"]))
                if review.get("decided")
                else verification["accept_rate"]
            ),
        }
    deltas: dict[str, float] = {}
    if len(order) == 2:
        first, second = (lane_stats[name] for name in order)
        for metric in ("parse_rate", "kept_yield"):
            deltas[f"{order[1]}_minus_{order[0]}_{metric}"] = round(
                float(cast(float | int, second[metric])) - float(cast(float | int, first[metric])),
                6,
            )
    reviewed = [name for name in lane_stats if lane_stats[name]["human_accept_rate"] is not None]
    reviewed_accept_rate = sorted(
        reviewed,
        key=lambda name: (-float(cast(float, lane_stats[name]["human_accept_rate"])), name),
    )
    return {
        "kind": report["kind"],
        "corpus_root": report["corpus_root"],
        "shared_seeds": len(cast(list[str], report["shared_seed_fingerprints"])),
        "execution": report.get("execution", {}),
        "lanes": lane_stats,
        "deltas": deltas,
        "rankings": {
            **cast(dict[str, object], report["rankings"]),
            "reviewed_accept_rate": reviewed_accept_rate,
        },
        "finalization": report.get("finalization"),
    }


def format_comparison_statistics(stats: dict[str, object]) -> str:
    lines = [
        f"comparison: {stats['kind']}",
        f"corpus: {stats['corpus_root']}",
        f"shared seeds: {stats['shared_seeds']}",
    ]
    execution = cast(dict[str, object], stats["execution"])
    if execution:
        lines.append(
            f"execution: {execution.get('mode')} order={execution.get('model_order')} "
            f"unload-between={execution.get('unload_between_lanes', False)}"
        )
    lines.append(
        "lane      model                              parsed   kept   parse    yield   "
        "gate  calls  latency  human"
    )
    for name, lane in cast(dict[str, dict[str, object]], stats["lanes"]).items():
        accepted = lane["human_accept_rate"]
        human = "pending" if accepted is None else f"{float(cast(float, accepted)):.1%}"
        lines.append(
            f"{name:<9} {str(lane['model']):<34} "
            f"{lane['draft_parsed']}/{lane['draft_attempts']:<5} "
            f"{lane['kept']}/{lane['seeds']:<5} "
            f"{float(cast(float, lane['parse_rate'])):>6.1%} "
            f"{float(cast(float, lane['kept_yield'])):>7.1%} "
            f"{str(lane['calibration_passed']):<5} "
            f"{lane['calls']!s:<6} {float(cast(float | int, lane['latency_s'])):>7.1f}s "
            f"{human}"
        )
    for name, value in cast(dict[str, float], stats["deltas"]).items():
        lines.append(f"delta {name}: {value:+.1%}")
    lines.append(f"rankings: {stats['rankings']}")
    return "\n".join(lines)
