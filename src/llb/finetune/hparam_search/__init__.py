"""Budgeted per-model LoRA hyperparameter search that never leaves the tuning split.

The split discipline of `optimize/tuner.py` is extended one level down: that tuner searches RAG and
serving knobs on the tuning split while `final` stays held out. Here the search space is the LoRA
configuration itself, and the held-out set is carved from *inside* the tuning split -- a seeded dev
sub-slice that no trial ever trains on. Calibration and final never enter a trial at all, and a
guard refuses a dataset that so much as names one of their item ids.

The Optuna conventions are the tuner's: a seeded `TPESampler`, a persistent SQLite study so a killed
search resumes, and pruned rather than crashed trials on a measured OOM. The trainer and the
objective are injectable, so CI runs a complete study over the fake trainer with a synthetic
objective and no CUDA.

The implementation is split into `model` (vocabulary + dataclasses), `dev_slice` (held-out carve),
`space` (LoRA search space + guards + footprint estimate), `objective` (per-trial fine-tune +
score), `search` (study orchestration), and `manifest_io` (read the recorded best back). The
public API is re-exported here so callers keep importing `llb.finetune.hparam_search`.
"""

from llb.finetune.hparam_search.dev_slice import (
    base_score_bucket,
    carve_dev_slice,
    carve_stratified_dev_slice,
    load_base_scores,
)
from llb.finetune.hparam_search.manifest_io import (
    latest_hparams_manifest,
    load_hparams_manifest,
    trainer_defaults,
)
from llb.finetune.hparam_search.model import (
    BATCH_GEOMETRY_CHOICES,
    DEFAULT_DEV_FRACTION,
    DEFAULT_MAX_TRIALS,
    DEFAULT_SEED,
    HPARAMS_MANIFEST,
    HPARAMS_METHOD,
    MAX_LENGTH_CHOICES,
    STATE_COMPLETE,
    STATE_FAILED,
    STATE_PRUNED,
    Clock,
    DevSlice,
    EstimateFn,
    HparamSearchResult,
    ObjectiveFn,
    TrialRecord,
    TrialTrainerFn,
)
from llb.finetune.hparam_search.space import (
    adapter_param_estimate,
    assert_tuning_only,
    estimated_adapter_train_mib,
    suggest_lora_hyperparameters,
)
from llb.finetune.hparam_search.search import search_hyperparameters

__all__ = [
    "BATCH_GEOMETRY_CHOICES",
    "DEFAULT_DEV_FRACTION",
    "DEFAULT_MAX_TRIALS",
    "DEFAULT_SEED",
    "HPARAMS_MANIFEST",
    "HPARAMS_METHOD",
    "MAX_LENGTH_CHOICES",
    "STATE_COMPLETE",
    "STATE_FAILED",
    "STATE_PRUNED",
    "Clock",
    "DevSlice",
    "EstimateFn",
    "HparamSearchResult",
    "ObjectiveFn",
    "TrialRecord",
    "TrialTrainerFn",
    "adapter_param_estimate",
    "assert_tuning_only",
    "base_score_bucket",
    "carve_dev_slice",
    "carve_stratified_dev_slice",
    "estimated_adapter_train_mib",
    "latest_hparams_manifest",
    "load_base_scores",
    "load_hparams_manifest",
    "search_hyperparameters",
    "suggest_lora_hyperparameters",
    "trainer_defaults",
]
