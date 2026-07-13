"""Vocabulary of the LoRA hyperparameter search: the artifact names, budget defaults, base-score
strata, the LoRA/batch search space, the footprint constants, the injectable callable aliases, and
the three record dataclasses.

A leaf module -- the dev-slice carve, search space, objective, and search orchestration build on it.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from llb.core.contracts import JsonObject
from llb.finetune.trainer import DEFAULT_TARGET_MODULES

HPARAMS_METHOD = "finetune-hparams"
HPARAMS_MANIFEST = "hparams_manifest.json"
TRIAL_JOURNAL = "trials.jsonl"
STUDY_DB = "study.db"
TRIALS_DIRNAME = "trials"

DEFAULT_SEED = 13
DEFAULT_MAX_TRIALS = 8
DEFAULT_DEV_FRACTION = 0.25
# Below two items a "held-out dev slice" is not held out from anything: one of train/dev is empty.
MIN_SLICE_ITEMS = 2

# Base-score strata for `--stratify-by-base-score`. An item is ANSWERABLE when the base model
# already scores above zero on it; a dev slice with no answerable item makes the trial objective
# a near-constant that ranks every configuration the same (the first CUDA search on this repo hit
# exactly that with a uniform 3-item slice). Buckets are drawn from in ANSWERABLE-FIRST priority
# order so the floor-of-one lands on discriminating items before zeros.
BUCKET_HIGH = "high"  # objective_score >= HIGH_SCORE_BOUNDARY
BUCKET_LOW = "low"  # 0 < objective_score < HIGH_SCORE_BOUNDARY
BUCKET_ZERO = "zero"  # objective_score == 0.0
BUCKET_UNSCORED = "unscored"  # item absent from the base run's scores.jsonl
HIGH_SCORE_BOUNDARY = 0.5
BUCKET_PRIORITY = (BUCKET_HIGH, BUCKET_LOW, BUCKET_ZERO, BUCKET_UNSCORED)
SCORES_FILENAME = "scores.jsonl"
SECONDS_PER_HOUR = 3600.0
DIGEST_TAG_CHARS = 12

# The dev sub-slice is never materialized as a dataset: only the train sub-slice is trained on, and
# the objective scores dev items straight from the goldset.
TRAIN_ROLE = "train"

STATE_COMPLETE = "complete"
STATE_PRUNED = "pruned"
STATE_FAILED = "failed"

# The LoRA search space. `lora_alpha` is sampled as a MULTIPLE of the rank rather than
# independently: the effective update scale is alpha/r, so an independent alpha would spend most of
# the budget on rank/alpha pairs that differ only in a scale the optimizer can already reach.
LORA_R_CHOICES = [4, 8, 16, 32, 64]
LORA_ALPHA_MULTIPLIERS = [1, 2, 4]
LORA_DROPOUT_RANGE = (0.0, 0.2)
LORA_DROPOUT_STEP = 0.05
LEARNING_RATE_RANGE = (1e-5, 5e-4)
EPOCHS_RANGE = (1, 4)
# Named module sets rather than a free subset: PEFT target modules are architecture-specific, and a
# sampled arbitrary subset would mostly produce configurations that fail to attach.
TARGET_MODULE_PRESETS: dict[str, list[str]] = {
    "qv": ["q_proj", "v_proj"],
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "attn_mlp": list(DEFAULT_TARGET_MODULES),
}

# Effective-batch axis (finetune-hparams-effective-batch-axis). Effective batch size interacts
# strongly with the learning rate, so a best learning rate is only best AT its batch geometry.
# The two knobs are sampled as ONE named geometry (per_device x grad_accum) rather than
# independently -- independent draws would mostly differ only in a VRAM/wall-clock trade at the
# same effective batch, wasting budget on gradient-equivalent points. Geometries stay
# single-per-device-heavy because the constrained 16 GB host is the design target.
BATCH_GEOMETRY_CHOICES: dict[str, tuple[int, int]] = {
    "1x4": (1, 4),  # the trainer's conservative default
    "1x8": (1, 8),
    "2x4": (2, 4),
    "2x8": (2, 8),
}
MAX_LENGTH_CHOICES = [512, 1024, 2048]

# Pre-run VRAM feasibility (finetune-hparams-infeasible-point-prune). A LoRA pair holds two
# matrices of `hidden x rank` per targeted module per layer, and training multiplies each
# parameter by weight + gradient (bf16) plus fp32 Adam moments and master copy. The estimate is
# deliberately coarse -- it exists to prune KNOWN-infeasible points before a fine-tune is paid
# for, not to replace the measured-OOM prune that stays in place after it.
LORA_MATRICES_PER_MODULE = 2
ADAPTER_TRAIN_BYTES_PER_PARAM = 16.0  # 2 weight + 2 grad + 8 Adam m/v + 4 fp32 master

# (adapter_dir, hyperparameters) -> dev-slice objective. Injectable so CI needs no backend.
ObjectiveFn = Callable[[Path, JsonObject], float]
# (dataset_dir, model, adapter_dir, seed, hyperparameters) -> adapter manifest.
TrialTrainerFn = Callable[[Path, str, Path, int, JsonObject], JsonObject]
Clock = Callable[[], float]
# hyperparameters -> estimated training footprint in MiB, or None when the arch is unknown.
EstimateFn = Callable[[JsonObject], float | None]


@dataclass(frozen=True)
class DevSlice:
    """A seeded, disjoint split of the tuning-split item ids into train and held-out dev.

    `strata` is set only by the stratified carve: per-bucket population/dev counts and the
    base-score distribution the slice was drawn against, recorded into `hparams_manifest.json`.
    """

    train_ids: tuple[str, ...]
    dev_ids: tuple[str, ...]
    seed: int
    dev_fraction: float
    strata: JsonObject | None = None

    def as_dict(self) -> JsonObject:
        payload: JsonObject = {
            "seed": self.seed,
            "dev_fraction": self.dev_fraction,
            "n_train": len(self.train_ids),
            "n_dev": len(self.dev_ids),
            "train_ids": list(self.train_ids),
            "dev_ids": list(self.dev_ids),
        }
        if self.strata is not None:
            payload["strata"] = self.strata
        return payload


@dataclass(frozen=True)
class TrialRecord:
    number: int
    state: str
    objective: float | None
    hyperparameters: JsonObject
    duration_s: float = 0.0
    estimated_adapter_mib: float | None = None

    def as_dict(self) -> JsonObject:
        return {
            "number": self.number,
            "state": self.state,
            "objective": self.objective,
            "hyperparameters": self.hyperparameters,
            "duration_s": round(self.duration_s, 3),
            "estimated_adapter_mib": self.estimated_adapter_mib,
        }


@dataclass
class HparamSearchResult:
    out_dir: Path
    manifest_path: Path
    dev_slice: DevSlice
    trials: list[TrialRecord] = field(default_factory=list)
    best_trial: int | None = None
    best_objective: float | None = None
    best_hyperparameters: JsonObject | None = None
    budget_exhausted: bool = False

    @property
    def n_complete(self) -> int:
        return sum(1 for trial in self.trials if trial.state == STATE_COMPLETE)
