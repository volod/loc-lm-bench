"""Backend launcher construction for adapter serving plans."""

from llb.backends.base import BackendLauncher
from llb.core.config import RunConfig
from llb.finetune.serving.model import (
    ADAPTER_LORA_NAME,
    BACKEND_OLLAMA,
    BACKEND_VLLM,
    ServePlan,
)


def default_launcher(plan: ServePlan, config: RunConfig) -> BackendLauncher:
    """Build a backend launcher from a direct-LoRA or merged-artifact plan."""
    if plan.backend == BACKEND_VLLM:
        from llb.backends.vllm import VllmLauncher
        from llb.finetune.adapter_manifest import adapter_lora_rank

        return VllmLauncher(
            plan.served_model,
            host=config.vllm_host,
            port=config.vllm_port,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            cpu_offload_gb=config.cpu_offload_gb,
            kv_offloading_size_gb=config.kv_offloading_size_gb,
            dtype=config.dtype,
            quantization=config.quantization,
            adapter_path=plan.adapter_path,
            adapter_name=ADAPTER_LORA_NAME,
            max_lora_rank=adapter_lora_rank(plan.adapter_path),
        )
    if plan.backend == BACKEND_OLLAMA:
        from llb.backends.ollama import OllamaLauncher

        return OllamaLauncher(plan.served_model, host=config.ollama_host)
    from llb.backends.llamacpp import LlamaCppLauncher
    from llb.backends.llamacpp_command import resolve_llama_server_binary

    return LlamaCppLauncher(
        plan.served_model,
        host=config.llamacpp_host,
        n_gpu_layers=config.n_gpu_layers,
        ctx_size=config.max_model_len,
        binary=resolve_llama_server_binary(config.data_dir),
    )


def backend_endpoint(backend: str, config: RunConfig) -> str:
    if backend == BACKEND_VLLM:
        return config.vllm_host
    if backend == BACKEND_OLLAMA:
        return config.ollama_host
    return config.llamacpp_host
