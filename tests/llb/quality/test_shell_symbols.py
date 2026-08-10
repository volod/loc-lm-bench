"""A shell helper that vanishes is not a lint finding anywhere else -- it is a run-time failure.

The shipped-tree assertion is the point of the tool. The synthetic cases pin what counts as a CALL
(including a name handed to a runner) against what only LOOKS like one (prose in a comment, a `#`
inside parameter expansion), and the two ways a definition legitimately enters scope: a sourced
file, and the `# llb-requires:` line a shared module uses to name the sibling its functions assume.

The prefix scan is the other half: the call scan can only see a helper that carries the prefix, so
an unprefixed definition is refused rather than left silently outside the check -- with the one
exemption that keeps the rule honest, a function defined inside a make recipe.
"""

import subprocess

from llb.core.paths import PROJECT_ROOT
from llb.quality.shell_symbols import (
    strip_comment,
    unprefixed_definitions,
    unresolved_calls,
    unused_requirements,
)

_COMMON = """#!/usr/bin/env bash
llb_load_env() {
  echo loading
}
"""


def _repo(tmp_path, files: dict[str, str]):
    for name, text in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    return tmp_path


def test_the_shipped_shell_layer_resolves_every_call():
    """The CI-able assertion: every `llb_*` name in the repo has a definition in its own scope."""
    assert unresolved_calls(PROJECT_ROOT) == []


def test_the_shipped_shell_layer_prefixes_every_function():
    """The rule that keeps the assertion above meaningful: no helper sits outside the scan."""
    assert unprefixed_definitions(PROJECT_ROOT) == []


def test_the_shipped_shell_layer_uses_every_declared_sibling():
    """The gate proof: every file-level contract supports a direct call in its declaring fragment."""
    assert unused_requirements(PROJECT_ROOT) == []


def test_a_definition_without_the_prefix_is_named_with_its_line(tmp_path):
    """An unprefixed helper is not a call the scan can see, so it is refused where it is written."""
    root = _repo(
        tmp_path,
        {
            "scripts/quickstart/helpers.sh": (
                '# shellcheck shell=bash\nresolve_path() {\n  echo "$1"\n}\n'
            ),
        },
    )

    assert unprefixed_definitions(root) == [
        "scripts/quickstart/helpers.sh:2: shell function without the llb_ prefix -> resolve_path"
    ]


def test_a_recipe_local_function_is_exempt_from_the_prefix(tmp_path):
    """A function defined inside a `bash -c` recipe never enters the sourced namespace."""
    root = _repo(
        tmp_path,
        {
            "make/eval.mk": (
                "matrix:\n"
                "\t@set -e; \\\n"
                '\twants_backend() { case " $(BACKENDS) " in *" $$1 "*) return 0 ;; esac; }; \\\n'
                "\twants_backend ollama\n"
            ),
        },
    )

    assert unprefixed_definitions(root) == []


def test_a_call_to_a_deleted_helper_is_named_with_its_line(tmp_path):
    """The failure this exists for: the definition goes, the call stays, every other scan is green."""
    root = _repo(
        tmp_path,
        {
            "scripts/shared/common.sh": _COMMON,
            "scripts/run.sh": (
                "#!/usr/bin/env bash\n"
                "# shellcheck source=shared/common.sh\n"
                '. "$SCRIPT_DIR/shared/common.sh"\n'
                "llb_load_env\n"
                "llb_check_shell_scripts\n"
            ),
        },
    )

    assert unresolved_calls(root) == [
        "scripts/run.sh:5: no definition in scope -> llb_check_shell_scripts"
    ]


def test_a_name_handed_to_a_runner_is_a_call(tmp_path):
    """`llb_fail_if_output "$LABEL" llb_some_scan` breaks exactly like a direct call does."""
    root = _repo(
        tmp_path,
        {
            "scripts/shared/common.sh": _COMMON,
            "scripts/run.sh": (
                "#!/usr/bin/env bash\n"
                "# shellcheck source=shared/common.sh\n"
                '. "$SCRIPT_DIR/shared/common.sh"\n'
                'llb_load_env "label" llb_missing_scan\n'
            ),
        },
    )

    assert unresolved_calls(root) == [
        "scripts/run.sh:4: no definition in scope -> llb_missing_scan"
    ]


def test_prose_and_parameter_expansion_are_not_calls(tmp_path):
    """Comments here name helpers constantly, and `#` also appears mid-word in an expansion."""
    root = _repo(
        tmp_path,
        {
            "scripts/run.sh": (
                "#!/usr/bin/env bash\n"
                "# llb_load_env reads .env before anything else runs.\n"
                'name="${BASH_SOURCE#llb_prefix}"  # llb_check_shell_scripts is gone\n'
                'echo "$name"\n'
            ),
        },
    )

    assert unresolved_calls(root) == []


def test_a_shared_module_declares_the_sibling_its_functions_assume(tmp_path):
    """A module sourced BY an entrypoint never sources its sibling, so it says so instead."""
    files = {
        "scripts/shared/common.sh": _COMMON,
        "scripts/shared/gate.sh": ("#!/usr/bin/env bash\nllb_gate() {\n  llb_load_env\n}\n"),
    }
    without = _repo(tmp_path / "without", files)
    assert unresolved_calls(without) == [
        "scripts/shared/gate.sh:3: no definition in scope -> llb_load_env"
    ]

    declared = dict(files)
    declared["scripts/shared/gate.sh"] = (
        "#!/usr/bin/env bash\n# llb-requires: common.sh\nllb_gate() {\n  llb_load_env\n}\n"
    )
    assert unresolved_calls(_repo(tmp_path / "declared", declared)) == []


def test_a_requirement_left_after_its_last_call_is_named_with_its_line(tmp_path):
    """A whole-file declaration is stale when the caller names none of that sibling's functions."""
    root = _repo(
        tmp_path,
        {
            "scripts/shared/common.sh": _COMMON,
            "scripts/shared/gate.sh": (
                "#!/usr/bin/env bash\n# llb-requires: common.sh\nllb_gate() {\n  echo gate\n}\n"
            ),
        },
    )

    assert unused_requirements(root) == [
        "scripts/shared/gate.sh:2: unused llb-requires declaration -> common.sh"
    ]


def test_one_direct_reference_justifies_a_whole_file_requirement(tmp_path):
    """Minimality is file-level: callers need not reference every function in the sibling."""
    root = _repo(
        tmp_path,
        {
            "scripts/shared/common.sh": (_COMMON + "llb_other_helper() {\n  echo other\n}\n"),
            "scripts/shared/gate.sh": (
                "#!/usr/bin/env bash\n# llb-requires: common.sh\nllb_gate() {\n  llb_load_env\n}\n"
            ),
        },
    )

    assert unused_requirements(root) == []


def test_a_make_recipe_is_checked_against_what_it_sources(tmp_path):
    """The call sites outside `scripts/`: a recipe that sources the shared module by literal path."""
    root = _repo(
        tmp_path,
        {
            "scripts/shared/common.sh": _COMMON,
            "make/dev.mk": (
                "demo:\n"
                "\t@bash -c 'source \"$(PROJECT_ROOT)/scripts/shared/common.sh\"; llb_load_env'\n"
                "stale:\n"
                "\t@bash -c 'source \"$(PROJECT_ROOT)/scripts/shared/common.sh\"; llb_ensure_env'\n"
            ),
        },
    )

    assert unresolved_calls(root) == ["make/dev.mk:4: no definition in scope -> llb_ensure_env"]


def test_strip_comment_keeps_quoted_hashes_and_expansions():
    """The three shapes a naive split on `#` would get wrong."""
    assert strip_comment('run "a # b" # tail').strip() == 'run "a # b"'
    assert strip_comment("echo ${name#llb_}") == "echo ${name#llb_}"
    assert strip_comment("# whole line") == ""
