"""Small value coercers shared by the campaign entry (de)serialization and report rendering."""

from pathlib import Path

from llb.core.contracts.models import ModelSpec


def _parse_models(models: list[str]) -> list[str]:
    out: list[str] = []
    for value in models:
        for item in value.split(","):
            model = item.strip()
            if model and model not in out:
                out.append(model)
    return out


def _model_key(spec: ModelSpec) -> str:
    return str(spec.get("name") or spec.get("source"))


def _path_or_none(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def _path_from(value: object) -> Path | None:
    return Path(str(value)) if value else None


def _str_or_none(value: object) -> str | None:
    return None if value is None else str(value)


def _float_or_none(value: object) -> float | None:
    try:
        return None if value is None else float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _ci_from_value(value: object) -> tuple[float, float] | None:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return None
    return (float(value[0]), float(value[1]))


def _fmt(value: object) -> str:
    try:
        return f"{float(value):.4f}"  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/a"
