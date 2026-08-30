"""Shared constants for joint model + RAG-config search."""

JOINT_SEARCH_METHOD = "joint-search"

# Successive-halving: keep roughly 1/eta of candidates each screen round.
DEFAULT_ETA = 2

# How many survivors advance to the per-finalist multi-objective tune.
DEFAULT_MIN_FINALISTS = 2

# Cheap screen case cap on the tuning split (first round; later rounds multiply by eta).
DEFAULT_SCREEN_LIMIT = 8

DEFAULT_OBJECTIVES = "quality,latency"

# vLLM context cap every candidate cell is served at when neither a flag nor a `--config` file
# names one. A search must always name one: a roster model's native window is 128k+, whose KV
# cache no 16 GiB card can hold.
DEFAULT_SEARCH_MAX_MODEL_LEN = 8192

# Artifact file names under ``$DATA_DIR/joint-search/<run>/``.
MANIFEST_FILE = "manifest.json"
LEDGER_FILE = "ledger.json"
SCOREBOARD_JSON = "scoreboard.json"
SCOREBOARD_MD = "scoreboard.md"

# Resume markers: cheap-screen cells and finished finalist tunes.
SCREEN_DIR = "screen"
FINALIST_RESULT_FILE = "result.json"
# Per-pick final-split markers under ``finalists/<slug>/picks/<goal>.json``.
PICKS_DIR = "picks"
MARKER_STATUS_DONE = "done"

# Optuna SQLite root (same as ``llb.optimize.multi_objective_runtime.OPTUNA_METHOD``).
OPTUNA_METHOD = "optuna"
