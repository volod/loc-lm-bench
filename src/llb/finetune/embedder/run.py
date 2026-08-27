"""One embedder fine-tune run: export the pairs, train, and leave an adoptable model directory.

The run directory is the deliverable, not the weights alone. `pairs/` records what the encoder saw
and `model/` is what `compare-embeddings` is handed as a candidate, so the uplift claim and the
data behind it are one artifact an operator can hand to someone else.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from llb.core.contracts.common import JsonObject
from llb.finetune.embedder.negatives import DEFAULT_NEGATIVES
from llb.finetune.embedder.pairs import DEFAULT_MAX_POSITIVES, export_contrastive_pairs
from llb.finetune.embedder.trainer import TRAINER_AUTO, train_embedder

if TYPE_CHECKING:
    from llb.core.config import RunConfig

_LOG = logging.getLogger(__name__)

METHOD_DIR = "finetune-embedder"
PAIRS_SUBDIR = "pairs"
MODEL_SUBDIR = "model"


@dataclass(frozen=True)
class EmbedderFinetuneResult:
    """Where the run wrote, and what the two manifests say about it."""

    out_dir: Path
    pairs_dir: Path
    model_dir: Path
    pairs_manifest: JsonObject
    tuned_manifest: JsonObject


def run_embedder_finetune(
    cfg: "RunConfig",
    *,
    base_model: str,
    out_dir: Path | str | None = None,
    seed: int = 13,
    trainer: str = TRAINER_AUTO,
    max_positives: int = DEFAULT_MAX_POSITIVES,
    negatives: int = DEFAULT_NEGATIVES,
    hyperparameters: JsonObject | None = None,
) -> EmbedderFinetuneResult:
    """Export tuning-split pairs from `cfg`'s gold set + corpus, then train `base_model` on them.

    The chunking is the RUN CONFIG's, not a training-only choice: the positives have to be the
    chunks a query will actually meet in the store, or an uplift measured here would not be an
    uplift there.
    """
    run_dir = Path(out_dir) if out_dir is not None else _default_run_dir(cfg, base_model)
    pairs_dir = run_dir / PAIRS_SUBDIR
    model_dir = run_dir / MODEL_SUBDIR
    _LOG.info("[finetune-embedder] exporting pairs -> %s", pairs_dir)
    pairs = export_contrastive_pairs(
        goldset_path=cfg.goldset_path,
        corpus_root=cfg.corpus_root,
        out_dir=pairs_dir,
        strategy=cfg.strategy,
        size=cfg.chunk_size,
        overlap=cfg.chunk_overlap,
        max_positives=max_positives,
        negatives=negatives,
        lexical_lemmas=cfg.lexical_lemmas,
        seed=seed,
    )
    _LOG.info(
        "[finetune-embedder] training %s on %d pairs from %d items",
        base_model,
        pairs["n_pairs"],
        len(pairs["item_ids"]),
    )
    tuned = train_embedder(
        pairs_dir=pairs_dir,
        base_model=base_model,
        out_dir=model_dir,
        seed=seed,
        trainer=trainer,
        hyperparameters=hyperparameters,
        goldset_path=cfg.goldset_path,
    )
    return EmbedderFinetuneResult(
        out_dir=run_dir,
        pairs_dir=pairs_dir,
        model_dir=model_dir,
        pairs_manifest=pairs,
        tuned_manifest=tuned,
    )


def _default_run_dir(cfg: "RunConfig", base_model: str) -> Path:
    """`$DATA_DIR/finetune-embedder/<model-slug>/<timestamp>/` -- one directory per training."""
    from llb.bench.common import new_run_timestamp
    from llb.rag.embedding_bakeoff.models import slugify_model

    _run_id, run_ts = new_run_timestamp()
    return cfg.data_dir / METHOD_DIR / slugify_model(base_model) / run_ts
