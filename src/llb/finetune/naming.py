"""Filesystem naming for the fine-tuning artifact trees.

Campaign entries (`$DATA_DIR/finetune-campaign/<ts>/<model>/`) and hyperparameter studies
(`$DATA_DIR/finetune-hparams/<model>/<ts>/`) must name the same model the same way, so an operator
can pair a search with the campaign that consumed it. This module is the single source of truth for
that slug and lives below both so neither has to import the other.
"""

FALLBACK_SLUG = "model"
SLUG_SAFE_EXTRA = "._-"


def model_slug(model: str) -> str:
    """Filesystem-safe directory name for a model id (`Qwen/Qwen2.5-0.5B-Instruct` -> `Qwen-...`)."""
    cleaned = "".join(ch if ch.isalnum() or ch in SLUG_SAFE_EXTRA else "-" for ch in model)
    return cleaned.strip("-") or FALLBACK_SLUG
