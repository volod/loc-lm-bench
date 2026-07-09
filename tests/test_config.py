from pathlib import Path

import pytest
from pydantic import ValidationError

from llb.core.config import RUN_EVAL_METHOD, RunConfig
from llb.core.paths import PROJECT_ROOT


def test_defaults_are_cuda_free_ollama():
    cfg = RunConfig()
    assert cfg.backend == "ollama"
    assert cfg.temperature == 0.0  # deterministic by default
    assert cfg.judge_threshold == 0.6
    assert "e5" in cfg.embedding_model


def test_retrieval_mode_defaults_to_flat():
    cfg = RunConfig()
    assert cfg.retrieval_mode == "flat"
    assert cfg.child_chunk_size == 400


def test_vllm_serving_defaults():
    cfg = RunConfig()
    assert cfg.gpu_memory_utilization == 0.85
    assert cfg.vllm_port == 8000
    assert cfg.measure_telemetry is False
    assert RunConfig(backend="vllm", model="org/Model").backend == "vllm"


def test_vllm_endpoint_port_must_match_launcher_port():
    with pytest.raises(ValidationError, match="must match"):
        RunConfig(backend="vllm", vllm_host="http://localhost:8000", vllm_port=8001)


def test_judge_endpoint_must_be_explicit_http_url_without_credentials():
    assert RunConfig(judge_base_url="http://localhost:9000/v1").judge_base_url.endswith("/v1")
    with pytest.raises(ValidationError, match="judge_base_url must be an http"):
        RunConfig(judge_base_url="localhost:9000")
    with pytest.raises(ValidationError, match="must not contain credentials"):
        RunConfig(judge_base_url="http://user:secret@localhost:9000/v1")


def test_index_and_run_dirs_under_data_dir(tmp_path):
    cfg = RunConfig(data_dir=tmp_path, run_name="r1")
    assert cfg.index_dir() == tmp_path / "llb" / "rag"
    assert cfg.run_dir("20260620T120000Z") == (tmp_path / RUN_EVAL_METHOD / "20260620T120000Z")
    assert cfg.corpus_root == tmp_path / "llb" / "corpus"
    assert cfg.goldset_path == tmp_path / "llb" / "goldset" / "goldset_uk.jsonl"


def test_load_yaml_overrides_defaults(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("model: mistral:7b\ntop_k: 8\n", encoding="utf-8")
    cfg = RunConfig.load(p)
    assert cfg.model == "mistral:7b"
    assert cfg.top_k == 8
    assert cfg.backend == "ollama"  # unspecified key keeps the default


def test_fingerprint_is_json_serializable():
    fp = RunConfig().fingerprint()
    assert fp["model"] and isinstance(fp["data_dir"], str)


def test_data_dir_from_environment_is_project_relative(monkeypatch):
    monkeypatch.setenv("DATA_DIR", "var/test-data")
    cfg = RunConfig()
    assert cfg.data_dir == PROJECT_ROOT / "var" / "test-data"
    assert cfg.corpus_root == cfg.data_dir / "llb" / "corpus"


def test_relative_paths_do_not_depend_on_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    cfg = RunConfig(data_dir="runtime", corpus_root="corpus")
    assert cfg.data_dir == PROJECT_ROOT / "runtime"
    assert cfg.corpus_root == PROJECT_ROOT / "corpus"


def test_overrides_are_revalidated_and_rebase_default_paths(tmp_path):
    cfg = RunConfig().with_overrides(data_dir=tmp_path)
    assert cfg.corpus_root == tmp_path / "llb" / "corpus"
    with pytest.raises(ValidationError, match="chunk_overlap"):
        cfg.with_overrides(chunk_size=100, chunk_overlap=100)


def test_unknown_yaml_keys_are_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("topkk: 8\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="topkk"):
        RunConfig.load(path)


def test_invalid_yaml_has_path_context(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("top_k: [\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot load config"):
        RunConfig.load(path)


def test_run_timestamp_must_be_one_path_segment():
    with pytest.raises(ValueError, match="path segment"):
        RunConfig().run_dir(str(Path("nested") / "timestamp"))


def test_query_prep_defaults_to_empty_noop():
    assert RunConfig().query_prep == []
    assert RunConfig().query_glossary_path is None


def test_query_prep_accepts_known_steps():
    cfg = RunConfig().with_overrides(query_prep=["normalize", "typos", "glossary"])
    assert cfg.query_prep == ["normalize", "typos", "glossary"]


def test_query_prep_rejects_unknown_step():
    with pytest.raises(ValidationError, match="unknown query_prep step"):
        RunConfig().with_overrides(query_prep=["normalize", "nope"])


def test_query_prep_rejects_duplicate_step():
    with pytest.raises(ValidationError, match="unique"):
        RunConfig().with_overrides(query_prep=["typos", "typos"])
