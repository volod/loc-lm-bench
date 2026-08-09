"""A source file the repo cannot see is coverage that exists on exactly one machine.

`.gitignore` carried the packaging rule `build/` unanchored, so it matched a directory of that
name at any depth: `tests/llb/build/test_build_helper.py` -- the only test of `llb_max_jobs`, the
canonical parallelism cap for heavy CUDA builds -- was never committed, while pytest collected it
and `make ci` ran it on the box that happened to hold the file. Nothing failed; the coverage was
simply absent from every fresh clone and from GitHub CI.

The failure mode is silent by construction, so it needs a check that is not. These tests pin the
invariant (no source or test file is hidden from the repo) and the rule shape that restores it
(the packaging rules are anchored to the root, where the artifacts they name are actually
written). See docs/impl/current/host-validation.md#code-quality-checks.
"""

import shutil
import subprocess

import pytest

from llb.core.paths import PROJECT_ROOT

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")

SOURCE_ROOTS = ("src", "tests", "scripts")
SOURCE_SUFFIXES = (".py", ".sh", ".md")

# Build artifacts that SHOULD stay ignored inside a source tree; everything else under
# SOURCE_ROOTS is hand-written and belongs in the repo.
ARTIFACT_PARTS = ("__pycache__",)
ARTIFACT_SUFFIXES = (".pyc", ".pyo")


def _ignored(*paths: str) -> set[str]:
    """The subset of `paths` that `.gitignore` hides, matched by rule alone.

    `--no-index` keeps the answer about the RULES rather than about what happens to be tracked
    today: a tracked file is exempt from ignore rules, which is exactly the exemption that let the
    unanchored rule look harmless for every package that had already been committed.
    """
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        input="\n".join(paths),
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        check=False,
    )
    assert result.returncode in (0, 1), result.stderr
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def test_no_source_or_test_file_is_hidden_from_the_repo():
    """The invariant: a file under a source root is either committable or a build artifact."""
    candidates = [
        path
        for root in SOURCE_ROOTS
        for path in sorted((PROJECT_ROOT / root).rglob("*"))
        if path.is_file()
        and path.suffix in SOURCE_SUFFIXES
        and path.suffix not in ARTIFACT_SUFFIXES
        and not any(part in ARTIFACT_PARTS for part in path.parts)
    ]
    assert candidates, "found no source files to check -- the walk is wrong, not the tree"

    relative = [str(path.relative_to(PROJECT_ROOT)) for path in candidates]
    hidden = _ignored(*relative)

    assert not hidden, (
        "these files are on disk but invisible to git, so they run here and nowhere else: "
        f"{sorted(hidden)} -- narrow the matching .gitignore rule "
        "(`git check-ignore -v <path>` names it)"
    )


def test_the_packaging_rules_are_anchored_to_the_repo_root():
    """The rule shape, pinned against a future unanchored `build/`, `lib/`, or `var/`."""
    nested = (
        "tests/llb/build/test_build_helper.py",
        "src/llb/build/vllm.py",
        "src/llb/lib/loader.py",
        "tests/llb/var/test_state.py",
        "docs/dist/guide.md",
    )

    assert _ignored(*nested) == set()


def test_the_packaging_rules_still_ignore_root_build_output():
    """Anchoring narrows the rules; it must not disarm them where the artifacts land."""
    artifacts = ("build/lib/llb/main.py", "dist/llb-0.1.0.whl", "wheels/llb-0.1.0.whl")

    assert _ignored(*artifacts) == set(artifacts)
