"""Re-score fixed-item RAG bundles under explicit verbosity policies."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from llb.core.fsutil import atomic_write_text
from llb.scoring.verbosity import POLICY_NAME, mean, pearson, policy_description, ranking_score

POLICY_COLUMNS = {
    "token_f1": "token_f1",
    "recall_only": "token_recall",
    "found_rate": "contains",
    POLICY_NAME: "ranking_score",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: expected non-empty JSONL objects")
    return rows


def _rank(values: Mapping[str, float]) -> dict[str, float]:
    """One-based average ranks, descending, with deterministic tie handling."""
    ordered = sorted(values, key=lambda label: (-values[label], label))
    ranks: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index
        while end + 1 < len(ordered) and values[ordered[end + 1]] == values[ordered[index]]:
            end += 1
        average_rank = (index + end) / 2.0 + 1.0
        for position in range(index, end + 1):
            ranks[ordered[position]] = average_rank
        index = end + 1
    return ranks


def _bundle_summary(run_dir: Path) -> tuple[str, list[str], dict[str, Any]]:
    manifest = _read_json(run_dir / "manifest.json")
    rows = _read_rows(run_dir / "scores.jsonl")
    required = {
        "item_id",
        "objective_score",
        "token_f1",
        "token_precision",
        "token_recall",
        "ranking_score",
        "contains",
        "completion_tokens",
    }
    missing = sorted({key for row in rows for key in required - set(row)})
    if missing:
        raise ValueError(
            f"{run_dir}: bundle predates verbosity decomposition; missing columns: {missing}"
        )
    if any(float(row["objective_score"]) != float(row["token_f1"]) for row in rows):
        raise ValueError(
            f"{run_dir}: objective_score no longer reproduces token_f1 bit-identically"
        )
    if any(
        float(row["ranking_score"])
        != ranking_score(float(row["token_precision"]), float(row["token_recall"]))
        for row in rows
    ):
        raise ValueError(f"{run_dir}: ranking_score does not reproduce the declared policy")
    manifest_metrics = manifest.get("metrics") or {}
    objective = mean([float(row["objective_score"]) for row in rows])
    if (
        manifest_metrics.get("objective_score") is not None
        and float(manifest_metrics["objective_score"]) != objective
    ):
        raise ValueError(f"{run_dir}: manifest objective does not reproduce its case rows")
    config = manifest.get("config") or {}
    model = str(config.get("model") or manifest.get("run_name") or run_dir.name)
    item_ids = [str(row["item_id"]) for row in rows]
    generated = [
        row
        for row in rows
        if row.get("status", "ok") == "ok" and float(row["completion_tokens"]) > 0
    ]
    lengths = [float(row["completion_tokens"]) for row in generated]
    metrics: dict[str, Any] = {
        column: mean([float(row[column]) for row in rows])
        for column in (
            "token_f1",
            "token_precision",
            "token_recall",
            "ranking_score",
            "contains",
        )
    }
    metrics["mean_completion_tokens"] = mean(lengths)
    metrics["length_correlations"] = {
        policy: pearson(lengths, [float(row[column]) for row in generated])
        for policy, column in POLICY_COLUMNS.items()
    }
    metrics["run_dir"] = str(run_dir)
    return model, item_ids, metrics


def analyze(run_dirs: list[Path]) -> dict[str, Any]:
    """Compare models over an identical item set under each candidate policy."""
    if len(run_dirs) < 2:
        raise ValueError("verbosity study requires at least two run bundles")
    models: dict[str, dict[str, Any]] = {}
    expected_ids: list[str] | None = None
    for run_dir in run_dirs:
        model, item_ids, metrics = _bundle_summary(Path(run_dir))
        if model in models:
            raise ValueError(f"verbosity study requires one bundle per model; duplicate: {model}")
        if expected_ids is None:
            expected_ids = item_ids
        elif item_ids != expected_ids:
            raise ValueError("verbosity study bundles must carry the same ordered item ids")
        models[model] = metrics
    ranks = {
        policy: _rank({model: float(metrics[column]) for model, metrics in models.items()})
        for policy, column in POLICY_COLUMNS.items()
    }
    for model, metrics in models.items():
        metrics["ranks"] = {policy: rank[model] for policy, rank in ranks.items()}
    f1_order = sorted(models, key=lambda model: (ranks["token_f1"][model], model))
    chosen_order = sorted(models, key=lambda model: (ranks[POLICY_NAME][model], model))
    changes = [
        {
            "model": model,
            "token_f1_rank": ranks["token_f1"][model],
            "chosen_rank": ranks[POLICY_NAME][model],
        }
        for model in models
        if ranks["token_f1"][model] != ranks[POLICY_NAME][model]
    ]
    return {
        "policy": {
            "chosen": POLICY_NAME,
            "description": policy_description(),
            "reason": (
                "Reference-fact coverage is the primary RAG outcome; the shipped short-answer "
                "instruction makes answer format material but secondary."
            ),
        },
        "n": len(expected_ids or []),
        "item_ids": expected_ids or [],
        "models": models,
        "orders": {"token_f1": f1_order, POLICY_NAME: chosen_order},
        "rank_changes": changes,
        "roster_length_correlations": {
            policy: pearson(
                [float(models[model]["mean_completion_tokens"]) for model in models],
                [float(models[model][column]) for model in models],
            )
            for policy, column in POLICY_COLUMNS.items()
        },
    }


def render(report: Mapping[str, Any]) -> str:
    """Render the compact maintained Markdown evidence artifact."""
    policy = report["policy"]
    lines = [
        "# Headline objective verbosity sensitivity",
        "",
        f"- fixed item set: {report['n']} items",
        f"- chosen policy: `{policy['chosen']}` -- {policy['description']}",
        f"- reason: {policy['reason']}",
        "",
        "| model | precision | recall | F1 | found | policy | mean tokens | r(len,F1) "
        "| F1 rank | recall rank | found rank | chosen rank |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model, metrics in sorted(
        report["models"].items(), key=lambda item: item[1]["ranks"][POLICY_NAME]
    ):
        corr = metrics["length_correlations"]["token_f1"]
        corr_text = "-" if corr is None else f"{corr:.3f}"
        lines.append(
            f"| `{model}` | {metrics['token_precision']:.3f} | "
            f"{metrics['token_recall']:.3f} | {metrics['token_f1']:.3f} | "
            f"{metrics['contains']:.3f} | {metrics['ranking_score']:.3f} | "
            f"{metrics['mean_completion_tokens']:.1f} | "
            f"{corr_text} | {metrics['ranks']['token_f1']:g} | "
            f"{metrics['ranks']['recall_only']:g} | "
            f"{metrics['ranks']['found_rate']:g} | "
            f"{metrics['ranks'][POLICY_NAME]:g} |"
        )
    roster = report["roster_length_correlations"]
    lines += ["", "## Roster sensitivity", ""]
    for name in POLICY_COLUMNS:
        value = roster[name]
        lines.append(f"- length vs `{name}`: {'-' if value is None else f'{value:.3f}'}")
    lines += ["", "## Rank changes", ""]
    if report["rank_changes"]:
        for change in report["rank_changes"]:
            lines.append(
                f"- `{change['model']}`: F1 rank {change['token_f1_rank']:g} -> "
                f"chosen-policy rank {change['chosen_rank']:g}"
            )
    else:
        lines.append("- No roster rank changes.")
    return "\n".join(lines) + "\n"


def write(report: Mapping[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=False)
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"
    atomic_write_text(json_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    atomic_write_text(md_path, render(report))
    return {"json": str(json_path), "report": str(md_path)}
