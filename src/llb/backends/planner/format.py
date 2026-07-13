"""ASCII rendering for model feasibility plans."""

from llb.core.contracts import ModelPlanRow

HEADERS = [
    "model",
    "backend",
    "params",
    "quant",
    "wt_GB",
    "ctx_gpu",
    "ctx_max",
    "gpu/total",
    "verdict",
]


def _gb(mib: float | None) -> str:
    return "-" if mib is None else f"{mib / 1024:.1f}"


def _format_row(row: ModelPlanRow) -> list[str]:
    n_layers = row["n_layers"]
    split = "-" if not n_layers else f"{row['gpu_layers']}/{n_layers}"
    return [
        row["name"],
        row["backend"],
        "-" if row["params_b"] is None else f"{row['params_b']}B",
        row["quant"] or "-",
        _gb(row["weights_mib"]),
        str(row["ctx_gpu"]) if row["ctx_gpu"] else "-",
        str(row["ctx_max"]) if row["ctx_max"] else "-",
        split,
        row["verdict"],
    ]


def format_plan(rows: list[ModelPlanRow], vram_mib: int, ram_mib: int) -> str:
    """Render a model plan as an ASCII table."""
    table = [_format_row(row) for row in rows]
    widths = [
        max(len(header), *(len(row[index]) for row in table)) if table else len(header)
        for index, header in enumerate(HEADERS)
    ]
    out = [
        f"host budget: VRAM {vram_mib} MiB + RAM {ram_mib} MiB "
        f"(usable after reserves; weights + KV must fit the combined budget)",
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(HEADERS)),
        "  ".join("-" * widths[index] for index in range(len(HEADERS))),
    ]
    out.extend(
        "  ".join(row[index].ljust(widths[index]) for index in range(len(HEADERS))) for row in table
    )
    return "\n".join(out)
