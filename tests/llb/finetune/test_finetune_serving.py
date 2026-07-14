"""Tests for finetune serving."""

import json
from pathlib import Path
import pytest
from llb.backends.vllm_command import build_vllm_command, served_lora_rank
from llb.finetune.adapter_manifest import adapter_lora_rank


def test_vllm_command_enables_lora_module():
    cmd = build_vllm_command("base-model", adapter_path="/tmp/adapter", adapter_name="adapter")
    assert "--enable-lora" in cmd
    assert "--lora-modules" in cmd
    assert "adapter=/tmp/adapter" in cmd
    assert "--max-lora-rank" not in cmd, "an unknown rank leaves vLLM on its own default"


def test_vllm_command_sizes_max_lora_rank_to_the_adapter():
    """vLLM defaults `--max-lora-rank` to 16, so a rank-32 adapter fails `add_lora` without this."""
    cmd = build_vllm_command("base-model", adapter_path="/tmp/adapter", max_lora_rank=32)

    assert cmd[cmd.index("--max-lora-rank") + 1] == "32"


def test_max_lora_rank_rounds_up_to_a_value_vllm_accepts():
    assert served_lora_rank(4) == 8, "vLLM accepts 1, 8, 16, ... -- never 4"
    assert served_lora_rank(16) == 16
    assert served_lora_rank(17) == 32

    with pytest.raises(SystemExit, match="exceeds the largest servable rank"):
        served_lora_rank(1024)


def test_max_lora_rank_is_omitted_when_no_adapter_is_served():
    assert "--max-lora-rank" not in build_vllm_command("base-model", max_lora_rank=64)


def test_adapter_lora_rank_prefers_the_peft_config(tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_manifest.json").write_text(
        json.dumps({"hyperparameters": {"lora_r": 8}}), encoding="utf-8"
    )
    assert adapter_lora_rank(adapter) == 8, "fall back to our manifest when PEFT wrote no config"

    (adapter / "adapter_config.json").write_text(json.dumps({"r": 64}), encoding="utf-8")

    assert adapter_lora_rank(adapter) == 64, "PEFT's config describes the weights actually on disk"
    assert adapter_lora_rank(None) is None
    assert adapter_lora_rank(tmp_path / "missing") is None
