"""Contrastive embedder training seam: the real sentence-transformers path, and the CI fake.

The objective is multiple-negatives ranking (InfoNCE): within a batch, a question must score its
own gold chunk above its own hard negatives AND above every other question's passages. That is the
objective the retrieval-tuned encoders on the roster were themselves trained with, so a fine-tune
under it moves the encoder along the axis its weights already encode rather than against it.

Two things the seam must not get wrong:

  - **The convention travels with the weights.** A tuned E5 is still queried with `query: ` and
    indexed with `passage: ` (`llb.rag.encoders.families`), so the training rows are prefixed the
    same way. Training bare text and querying prefixed text would tune the encoder for inputs it
    will never see again.
  - **The split guard runs BEFORE any weight moves.** A leaked calibration or final id cannot be
    un-trained once the run starts, so the refusal happens here, at the entry point, and reads the
    gold set rather than trusting the manifest that names it.

Heavy imports (`torch`, `sentence_transformers`, `datasets`) are deferred to the real path, so this
module imports in the base install and the `fake` trainer runs in CI with no GPU and no download.
The real path needs `[rag]` for the encoder and `[finetune]` for `datasets` + `accelerate`.
"""

import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING
from llb.core.contracts.common import JsonObject
from llb.finetune.dataset_io import _read_jsonl
from llb.finetune.embedder.manifest import (
    load_pairs_manifest,
    tuned_manifest,
    write_tuned_manifest,
)
from llb.finetune.embedder.pairs import PAIRS_FILENAME
from llb.finetune.guard import assert_tuning_only
from llb.rag.encoders.families import (
    apply_passage_convention,
    apply_query_convention,
    embedding_family,
)

if TYPE_CHECKING:  # the heavy stack is imported only on the real training path
    from sentence_transformers import SentenceTransformer
    from torch.nn import Module

_LOG = logging.getLogger(__name__)

COMMAND = "finetune-embedder"

TRAINER_AUTO = "auto"
TRAINER_SENTENCE_TRANSFORMERS = "sentence-transformers"
TRAINER_FAKE = "fake"
KNOWN_TRAINERS = (TRAINER_AUTO, TRAINER_SENTENCE_TRANSFORMERS, TRAINER_FAKE)

# Marker the fake trainer writes instead of weights, so a tuned directory produced in CI is
# recognizable as one that was never trained.
FAKE_MARKER = "embedder.fake"

# The two multiple-negatives losses. They optimize the SAME objective; the cached one (GradCache)
# reaches it in `mini_batch_size` chunks with the embeddings recomputed, so peak VRAM stops scaling
# with the batch. That is what lets a 12 GiB card train at a batch the objective actually needs.
LOSS_MNRL = "multiple-negatives-ranking"
LOSS_CACHED_MNRL = "cached-multiple-negatives-ranking"
KNOWN_LOSSES = (LOSS_CACHED_MNRL, LOSS_MNRL)

# Conservative defaults for a base-sized encoder on a single consumer CUDA card. `batch_size` is
# the load-bearing one: in-batch negatives come FROM the batch, so a smaller batch is a weaker
# objective, not merely a slower run. `mini_batch_size` is the opposite -- purely a memory knob,
# because GradCache's gradients are the uncached ones.
DEFAULT_HYPERPARAMETERS: JsonObject = {
    "loss": LOSS_CACHED_MNRL,
    "learning_rate": 2e-5,
    "num_train_epochs": 3.0,
    "per_device_train_batch_size": 16,
    "mini_batch_size": 4,
    "warmup_ratio": 0.1,
    "max_length": 512,
}


def resolved_hyperparameters(overrides: JsonObject | None) -> JsonObject:
    """The training configuration: the defaults above, with the operator's overrides applied.

    An unknown loss is refused HERE rather than in the real path, so the `fake` trainer rejects the
    same configurations the CUDA host would and a typo costs a second instead of a model load.
    """
    params = dict(DEFAULT_HYPERPARAMETERS)
    params.update(overrides or {})
    if params["loss"] not in KNOWN_LOSSES:
        raise SystemExit(
            f"[{COMMAND}] unknown loss {params['loss']!r}; expected one of "
            + " | ".join(KNOWN_LOSSES)
        )
    return params


def train_embedder(
    *,
    pairs_dir: Path | str,
    base_model: str,
    out_dir: Path | str,
    seed: int = 13,
    trainer: str = TRAINER_AUTO,
    hyperparameters: JsonObject | None = None,
    goldset_path: Path | str | None = None,
) -> JsonObject:
    """Train (or fake-train) a corpus-adapted encoder and write its `embedder_manifest.json`."""
    if trainer not in KNOWN_TRAINERS:
        raise SystemExit(
            f"[{COMMAND}] unknown --trainer {trainer!r}; expected one of "
            + " | ".join(KNOWN_TRAINERS)
        )
    pairs = load_pairs_manifest(pairs_dir)
    assert_tuning_only(pairs, goldset_path=goldset_path, command=COMMAND)
    params = resolved_hyperparameters(hyperparameters)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if trainer == TRAINER_FAKE:
        loss_curve = _fake_train(pairs_dir, base_model, out, params)
        concrete = TRAINER_FAKE
    else:
        loss_curve = _sentence_transformers_train(pairs_dir, base_model, out, seed, params)
        concrete = TRAINER_SENTENCE_TRANSFORMERS
    manifest = tuned_manifest(
        base_model=base_model,
        convention_family=embedding_family(base_model),
        pairs=pairs,
        pairs_dir=pairs_dir,
        seed=seed,
        hyperparameters=params,
        trainer=concrete,
        loss_curve=loss_curve,
    )
    write_tuned_manifest(out, manifest)
    return manifest


def training_rows(pairs_dir: Path | str, base_model: str) -> dict[str, list[str]]:
    """Convention-prefixed columns for the trainer: `anchor`, `positive`, `negative_1..n`.

    Rectangular by construction -- the export guarantees one fixed negative count for every row
    (`llb.finetune.embedder.pairs`), and a ragged set is refused here rather than truncated,
    because dropping a column silently weakens the objective on every row that had it.
    """
    records = _read_jsonl(Path(pairs_dir) / PAIRS_FILENAME)
    if not records:
        raise ValueError(f"no exported pairs in {pairs_dir}")
    widths = {len(record.get("negatives") or []) for record in records}
    if len(widths) != 1:
        raise ValueError(
            f"exported pairs carry {sorted(widths)} negatives per row; the trainer needs one "
            "fixed width. Re-export the pairs."
        )
    n_negatives = widths.pop()
    columns: dict[str, list[str]] = {
        "anchor": [
            apply_query_convention(base_model, [str(record["question"])])[0] for record in records
        ],
        "positive": [
            apply_passage_convention(base_model, [str(record["positive"])])[0] for record in records
        ],
    }
    for position in range(n_negatives):
        columns[f"negative_{position + 1}"] = [
            apply_passage_convention(base_model, [str(record["negatives"][position])])[0]
            for record in records
        ]
    return columns


def _fake_train(
    pairs_dir: Path | str, base_model: str, out: Path, params: JsonObject
) -> list[float]:
    """CI trainer: validate the row set, write a marker, and report a deterministic loss curve."""
    rows = training_rows(pairs_dir, base_model)
    (out / FAKE_MARKER).write_text(
        f"base_model={base_model}\nrows={len(rows['anchor'])}\nloss={params['loss']}\n",
        encoding="utf-8",
    )
    return [1.0, 0.5]


def _sentence_transformers_train(
    pairs_dir: Path | str, base_model: str, out: Path, seed: int, params: JsonObject
) -> list[float]:
    """Real path: a multiple-negatives objective over the exported rows, saved as an ST model dir.

    The trainer's own `output_dir` is a TEMPORARY directory, not a subdirectory of `out`: with
    `save_strategy="no"` nothing durable is written there, and the tuned model directory has to
    hold the model and its manifest only -- anything else in it is noise an operator has to
    recognize as noise.
    """
    _require_training_stack()
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from transformers.utils.logging import disable_progress_bar

    # Persisted CLI logs stay line-oriented ASCII, never tqdm control output (as in `Embedder`).
    disable_progress_bar()
    dataset = Dataset.from_dict(training_rows(pairs_dir, base_model))
    model = SentenceTransformer(base_model)
    model.max_seq_length = int(params["max_length"])
    scratch = TemporaryDirectory(prefix="llb-finetune-embedder-")
    arguments = SentenceTransformerTrainingArguments(
        output_dir=scratch.name,
        num_train_epochs=float(params["num_train_epochs"]),
        per_device_train_batch_size=int(params["per_device_train_batch_size"]),
        learning_rate=float(params["learning_rate"]),
        warmup_ratio=float(params["warmup_ratio"]),
        bf16=_bf16_available(),
        seed=seed,
        data_seed=seed,
        save_strategy="no",
        logging_steps=1,
        report_to=[],
        disable_tqdm=True,
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=arguments, train_dataset=dataset, loss=_build_loss(model, params)
    )
    try:
        trainer.train()
        model.save_pretrained(str(out))
    finally:
        scratch.cleanup()
    return [float(entry["loss"]) for entry in trainer.state.log_history if "loss" in entry]


def _build_loss(model: "SentenceTransformer", params: JsonObject) -> "Module":
    """The configured multiple-negatives loss (the name was validated before the model loaded)."""
    from sentence_transformers.sentence_transformer.losses import (
        CachedMultipleNegativesRankingLoss,
        MultipleNegativesRankingLoss,
    )

    if params["loss"] == LOSS_MNRL:
        return MultipleNegativesRankingLoss(model)
    return CachedMultipleNegativesRankingLoss(model, mini_batch_size=int(params["mini_batch_size"]))


def _bf16_available() -> bool:
    """Whether this host can train in bfloat16 (a CUDA card that supports it)."""
    import torch

    return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())


def _require_training_stack() -> None:
    """Check the training stack once, with the install action instead of a late traceback.

    `sentence_transformers` is the encoder half ([rag]); `datasets` and `accelerate` are the
    training half ([finetune]) -- `SentenceTransformerTrainer` is a `transformers.Trainer`, and it
    raises for a missing `accelerate` only after the arguments are built, which is late enough to
    look like a bug in this lane rather than a missing package.
    """
    try:
        import accelerate  # noqa: F401
        import datasets  # noqa: F401
        import sentence_transformers  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"[{COMMAND}] contrastive training needs the encoder stack and the training stack: "
            "make install-extras EXTRAS=rag,finetune"
        ) from exc
