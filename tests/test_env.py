"""`.env.example` stays aligned with the canonical env registry."""

from llb import env
from llb.paths import PROJECT_ROOT


def test_env_example_documents_required_vars():
    text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    for name in env.DOCUMENTED_ENV_VARS:
        assert f"{name}=" in text, f".env.example is missing active assignment for {name}"


def test_env_module_names_are_unique():
    names = [
        value
        for key, value in vars(env).items()
        if key.isupper() and isinstance(value, str) and key != "DOCUMENTED_ENV_VARS"
    ]
    assert len(names) == len(set(names))
