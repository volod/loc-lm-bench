import json

import pytest

from llb.backends import telemetry
from llb.prompts.engine import PromptAugmentation, render_template
from llb.prompts.registry import DEFAULT_TEMPLATE_ROOT
from llb.prompts.registry import render_chat, render_text, render_text_list
from llb.prompts.registry_generation import generate_registry


def test_render_template_substitutes_and_fails_fast():
    assert render_template("Hello {{ user.name }}", {"user": {"name": "Ada"}}) == "Hello Ada"
    with pytest.raises(KeyError):
        render_template("Hello {{ missing }}", {})


def test_generated_registry_matches_checked_in_registry():
    current = json.loads((DEFAULT_TEMPLATE_ROOT / "registry.json").read_text(encoding="utf-8"))
    assert generate_registry() == current


def test_fixed_prompt_list_comes_from_registry():
    prompts = render_text_list("telemetry.throughput")
    assert telemetry.DEFAULT_THROUGHPUT_PROMPTS == prompts
    assert len(prompts) == 3
    assert all(prompt for prompt in prompts)


def test_chat_template_supports_system_and_user_augmentation():
    messages = render_chat(
        "eval.rag.chat",
        {"context": "BASE CONTEXT", "question": "Питання?"},
        PromptAugmentation(system_prefix="SYS PROMPT", user_suffix="USER SUFFIX"),
    )
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith("SYS PROMPT")
    assert render_text("eval.rag.system") in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "BASE CONTEXT" in messages[1]["content"]
    assert messages[1]["content"].endswith("USER SUFFIX")
