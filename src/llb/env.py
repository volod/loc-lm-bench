"""Canonical environment variable names for loc-lm-bench.

Every module reads process configuration through these constants. User-facing
names, defaults, and descriptions live in ``.env.example`` at the project root.
"""

# Paths
DATA_DIR = "DATA_DIR"

# Logging
LLB_LOG = "LLB_LOG"

# RAG embedder device override (e.g. cpu / cuda / cuda:1). Unset -> sentence-transformers
# auto-selects (CUDA when available). Force `cpu` to keep the GPU free for a co-resident local
# judge/candidate (the embedder is tiny, so CPU encoding is cheap).
LLB_EMBED_DEVICE = "LLB_EMBED_DEVICE"

# Hugging Face downloads (gated models and datasets)
HF_TOKEN = "HF_TOKEN"

# Candidate inference endpoints (RunConfig + backends)
OLLAMA_HOST = "OLLAMA_HOST"
VLLM_HOST = "VLLM_HOST"
LLAMACPP_HOST = "LLAMACPP_HOST"
VLLM_USE_FLASHINFER_SAMPLER = "VLLM_USE_FLASHINFER_SAMPLER"
# Comma-separated flashinfer versions the preflight auto-pins when the bundled one fails (vLLM serving preflight).
FLASHINFER_CANDIDATES = "LLB_FLASHINFER_CANDIDATES"

# Local LLM judge. JUDGE_MODEL is the litellm route for the judge model; unset -> no judge
# runs (objective correctness ranks alone). Mirrors the Makefile JUDGE_MODEL knob and the
# --judge-model CLI flag. DEEPEVAL_* configure the OpenAI-compatible judge endpoint (no cloud
# egress).
JUDGE_MODEL = "JUDGE_MODEL"
DEEPEVAL_JUDGE_BASE_URL = "DEEPEVAL_JUDGE_BASE_URL"
DEEPEVAL_JUDGE_API_KEY = "DEEPEVAL_JUDGE_API_KEY"
DEEPEVAL_TELEMETRY_OPT_OUT = "DEEPEVAL_TELEMETRY_OPT_OUT"
# Relocate DeepEval's `.deepeval` keystore + results out of the project root, under $DATA_DIR/cache.
# DeepEval reads these env names directly (constants.HIDDEN_DIR / settings.DEEPEVAL_RESULTS_FOLDER).
DEEPEVAL_CACHE_FOLDER = "DEEPEVAL_CACHE_FOLDER"
DEEPEVAL_RESULTS_FOLDER = "DEEPEVAL_RESULTS_FOLDER"

# Frontier LLM prep (litellm reads standard provider key names)
OPENAI_API_KEY = "OPENAI_API_KEY"
ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"
GEMINI_API_KEY = "GEMINI_API_KEY"

# MLflow experiment UI (`make mlflow`)
MLFLOW_HOST = "MLFLOW_HOST"
MLFLOW_PORT = "MLFLOW_PORT"

# vLLM install/build (`make build-vllm`; optional overrides)
VLLM_SPEC = "VLLM_SPEC"
VLLM_SOURCE_DIR = "VLLM_SOURCE_DIR"
VLLM_BUILD_REQUIREMENTS = "VLLM_BUILD_REQUIREMENTS"
REBUILD_VLLM_WHEEL = "REBUILD_VLLM_WHEEL"
MAX_JOBS = "MAX_JOBS"

# Keys that must appear as active assignments in .env.example (not comment-only).
DOCUMENTED_ENV_VARS = (
    DATA_DIR,
    HF_TOKEN,
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    GEMINI_API_KEY,
    OLLAMA_HOST,
    VLLM_HOST,
    LLAMACPP_HOST,
    DEEPEVAL_JUDGE_BASE_URL,
    DEEPEVAL_JUDGE_API_KEY,
    DEEPEVAL_TELEMETRY_OPT_OUT,
    MLFLOW_HOST,
    MLFLOW_PORT,
)
