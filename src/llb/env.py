"""Canonical environment variable names for loc-lm-bench.

Every module reads process configuration through these constants. User-facing
names, defaults, and descriptions live in ``.env.example`` at the project root.
"""

# Paths
DATA_DIR = "DATA_DIR"

# Logging
LLB_LOG = "LLB_LOG"

# Hugging Face downloads (gated models and datasets)
HF_TOKEN = "HF_TOKEN"

# Candidate inference endpoints (RunConfig + backends)
OLLAMA_HOST = "OLLAMA_HOST"
VLLM_HOST = "VLLM_HOST"
VLLM_USE_FLASHINFER_SAMPLER = "VLLM_USE_FLASHINFER_SAMPLER"

# Judge endpoint fallbacks when judge_model uses a legacy prefix
OLLAMA_API_BASE = "OLLAMA_API_BASE"
HOSTED_VLLM_API_BASE = "HOSTED_VLLM_API_BASE"

# Local DeepEval judge (OpenAI-compatible endpoint; no cloud egress)
DEEPEVAL_JUDGE_BASE_URL = "DEEPEVAL_JUDGE_BASE_URL"
DEEPEVAL_JUDGE_API_KEY = "DEEPEVAL_JUDGE_API_KEY"
DEEPEVAL_TELEMETRY_OPT_OUT = "DEEPEVAL_TELEMETRY_OPT_OUT"

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
    VLLM_USE_FLASHINFER_SAMPLER,
    DEEPEVAL_JUDGE_BASE_URL,
    DEEPEVAL_JUDGE_API_KEY,
    DEEPEVAL_TELEMETRY_OPT_OUT,
    MLFLOW_HOST,
    MLFLOW_PORT,
)
