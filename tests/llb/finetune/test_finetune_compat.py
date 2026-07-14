"""Compressed-QAT trainability probes (compressed-qat-adapter-support): pure stages + report."""

import json
from pathlib import Path

from llb.finetune.compat import (
    config_compat_probe,
    linear_class_summary,
    probe_trainability,
    select_target_modules,
    unsupported_linear_classes,
)
from llb.finetune.compat_quantization import (
    VERDICT_NOT_TRAINABLE,
    VERDICT_TRAINABLE,
    VERDICT_UNKNOWN,
    assess_quantization,
    inspect_quantization,
)
from llb.finetune.compat_runtime import COMPAT_REPORT


# --- fake linear module classes: the CLASS NAME is what introspection matches on ---


class Linear:  # plain full-precision linear
    pass


class Linear4bit:  # bitsandbytes
    pass


class CompressedLinear:  # compressed-tensors: no PEFT dispatch
    pass


class LayerNorm:
    pass


def _llama_like_modules(linear_cls=Linear):
    return [
        ("model.embed_tokens", Linear()),
        ("model.layers.0.self_attn.q_proj", linear_cls()),
        ("model.layers.0.self_attn.k_proj", linear_cls()),
        ("model.layers.0.self_attn.v_proj", linear_cls()),
        ("model.layers.0.self_attn.o_proj", linear_cls()),
        ("model.layers.0.mlp.gate_proj", linear_cls()),
        ("model.layers.0.mlp.up_proj", linear_cls()),
        ("model.layers.0.mlp.down_proj", linear_cls()),
        ("model.layers.0.input_layernorm", LayerNorm()),
        ("lm_head", Linear()),
    ]


class FakeModel:
    def __init__(self, modules):
        self._modules = modules

    def named_modules(self):
        return list(self._modules)


# --- stage 1: config introspection ---


def test_inspect_quantization_reads_dicts_and_objects():
    assert inspect_quantization({}).is_native is False
    info = inspect_quantization({"quantization_config": {"quant_method": "compressed-tensors"}})
    assert info.quant_method == "compressed-tensors"

    class Cfg:
        quantization_config = {"quant_method": "awq", "bits": 4}

    info = inspect_quantization(Cfg())
    assert info.quant_method == "awq"
    assert info.details["bits"] == 4


def test_assess_quantization_verdicts():
    trainable, strategy, blocker = assess_quantization(inspect_quantization({}))
    assert (trainable, strategy, blocker) == (VERDICT_TRAINABLE, "peft-lora", None)

    verdict, strategy, _ = assess_quantization(
        inspect_quantization({"quantization_config": {"quant_method": "awq"}})
    )
    assert (verdict, strategy) == (VERDICT_TRAINABLE, "peft-awq")

    verdict, strategy, blocker = assess_quantization(
        inspect_quantization({"quantization_config": {"quant_method": "compressed-tensors"}})
    )
    assert verdict == VERDICT_NOT_TRAINABLE
    assert strategy is None
    assert "compressed-tensors" in blocker and "fallback" in blocker

    verdict, _, blocker = assess_quantization(
        inspect_quantization({"quantization_config": {"quant_method": "exotic-q"}})
    )
    assert verdict == VERDICT_UNKNOWN
    assert "exotic-q" in blocker


# --- stage 2 helpers: module scan + per-architecture target selection ---


def test_linear_class_summary_flags_compressed_linears():
    summary = linear_class_summary(_llama_like_modules(CompressedLinear))
    assert summary["CompressedLinear"] == 7
    assert unsupported_linear_classes(summary) == ["CompressedLinear"]
    assert unsupported_linear_classes(linear_class_summary(_llama_like_modules())) == []


def test_select_target_modules_grounds_in_existing_names():
    targets = select_target_modules(_llama_like_modules())
    assert targets == ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    # non-llama naming still yields an attachable set via the known vocabulary
    falcon = [("h.0.self_attention.query_key_value", Linear()), ("h.0.mlp.dense", Linear())]
    assert select_target_modules(falcon) == ["query_key_value", "dense"]
    # fully exotic names fall back to the most frequent linear suffixes; heads never enter
    exotic = [("blk.0.mix", Linear()), ("blk.1.mix", Linear()), ("lm_head", Linear())]
    assert select_target_modules(exotic) == ["mix"]


# --- the staged probe: skip reason lands BEFORE any weights load ---


def test_probe_skips_compressed_checkpoint_before_loading_weights(tmp_path: Path):
    loads: list[str] = []

    report = probe_trainability(
        "fake/qat-w4a16-ct",
        out_root=tmp_path,
        config_loader=lambda model: {"quantization_config": {"quant_method": "compressed-tensors"}},
        loader=lambda model: loads.append(model) or FakeModel([]),
    )
    assert report["verdict"] == VERDICT_NOT_TRAINABLE
    assert "compressed-tensors" in report["blocker"]
    assert loads == []  # the deterministic skip fires before training/loading starts
    saved = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
    assert saved["verdict"] == VERDICT_NOT_TRAINABLE
    assert Path(report["report_path"]).name == COMPAT_REPORT


def test_probe_trainable_checkpoint_attaches_and_reports(tmp_path: Path):
    attached: list[list[str]] = []

    report = probe_trainability(
        "fake/full-precision",
        out_root=tmp_path,
        config_loader=lambda model: {},
        loader=lambda model: FakeModel(_llama_like_modules()),
        attach_fn=lambda model_obj, targets: attached.append(targets),
    )
    assert report["verdict"] == VERDICT_TRAINABLE
    assert report["injection_strategy"] == "peft-lora"
    assert attached and attached[0][0] == "q_proj"
    assert report["linear_classes"]["Linear"] >= 7


def test_probe_records_attach_failure_as_blocker(tmp_path: Path):
    def boom(_model_obj, _targets):
        raise RuntimeError("CUDA out of memory")

    report = probe_trainability(
        "fake/oom",
        out_root=tmp_path,
        config_loader=lambda model: {},
        loader=lambda model: FakeModel(_llama_like_modules()),
        attach_fn=boom,
    )
    assert report["verdict"] == VERDICT_NOT_TRAINABLE
    assert "CUDA out of memory" in report["blocker"]


def test_probe_flags_compressed_linears_found_at_load_time(tmp_path: Path):
    report = probe_trainability(
        "fake/silent-ct",
        out_root=tmp_path,
        config_loader=lambda model: {},  # config is silent; the loaded modules tell the truth
        loader=lambda model: FakeModel(_llama_like_modules(CompressedLinear)),
        attach_fn=lambda *_: None,
    )
    assert report["verdict"] == VERDICT_NOT_TRAINABLE
    assert "CompressedLinear" in report["blocker"]


def test_config_compat_probe_never_false_skips(monkeypatch):
    import llb.finetune.compat as compat

    monkeypatch.setattr(
        compat,
        "_default_config_loader",
        lambda model, **kwargs: (_ for _ in ()).throw(OSError("offline")),
    )
    verdict = config_compat_probe("fake/unreachable")
    assert verdict["verdict"] == VERDICT_UNKNOWN
