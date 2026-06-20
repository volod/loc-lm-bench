from llb.config import RunConfig


def test_defaults_are_cuda_free_ollama():
    cfg = RunConfig()
    assert cfg.backend == "ollama"
    assert cfg.temperature == 0.0  # deterministic by default
    assert cfg.judge_threshold == 0.6
    assert "e5" in cfg.embedding_model


def test_index_and_run_dirs_under_data_dir(tmp_path):
    cfg = RunConfig(data_dir=tmp_path, run_name="r1")
    assert cfg.index_dir() == tmp_path / "llb" / "rag"
    assert cfg.run_dir() == tmp_path / "llb" / "runs" / "r1"


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
