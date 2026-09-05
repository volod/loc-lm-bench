# Quality Gate

Run the repository checks after host-specific validation:

```bash
make ci
make lint-md
scripts/code_quality.sh
```

`scripts/code_quality.sh` always prints the largest repository-visible Python files and non-Python
files, including new non-ignored files before they are staged. Root-file, markdown, shell, and
complexity sections are quiet when clean and
appear only when they have findings, missing optional tools, or failures; a shell-lint or
complexity finding also exits the sweep non-zero (see
[Code quality checks](#code-quality-checks) and
[complexity and shell gates](complexity-and-shell-gates.md)).

`make test` is the full local precommit flow when slow tests are acceptable.

## Code quality checks

`make ci` checks Ruff formatting and lint, mypy, the acceptance-gate inventory, the
[complexity and shell-lint gates](complexity-and-shell-gates.md), toolchain-pin integrity
(`make lint-toolchain`), and the non-slow pytest suite.
`make test` adds the full local test flow and Markdown lint; `make lint-md` also runs
`make lint-doc-links` (`llb.quality.doc_links`), which resolves every relative docs link -- file
plus `#anchor` -- so the three-level
current-implementation tree cannot rot into unfindable pages. `scripts/code_quality.sh` is the
wider sweep: it reports long source files and runs both gates, so maintainers can split code at
functional seams. The ~250-line source-file target is soft; cohesive schemas and regular lookup
families may remain whole.

The source-size refactor separates the most frequently edited long modules at their functional
boundaries:

- agent context policy vocabulary, transcript state, and bounded summarization live in
  `bench/agentic/context_policy.py`, `context.py`, and `context_summary.py`;
- context comparison models/pairing, task-kind analysis, recommendation rendering, and persistence
  live in `bench/context_policy/report.py` plus the `_kind`, `_recommendation`, and `_persist`
  modules; `context_policy/report_persist.py` also writes the per-policy bundles
  (`persist_policy_bundles`), so `agentic_context.py` is the sweep alone;
- the episode loop delegates prompt/compaction assembly, controller transport/repair, and mutable
  tally state to `agentic/episode_prompt.py`, `episode_controller.py`, and `episode_state.py`;
- retrieval comparison contracts, resolved settings, and optional output rows live in
  `rag/comparison/models.py`, `comparison/settings.py`, and `comparison/rows.py`;
- controller-authority run contracts and snapshot-isolation proof live in
  `controller_authority/model.py` and `controller_authority/snapshot.py`;
- embedding adoption reason clauses live in `rag/embedding_bakeoff/reason.py`;
- policy-change replay geometry and design loading live in
  `bench/policy_change/geometry.py`;
- embedding CLI validation/output persistence and persisted agentic comparison commands live in
  `cli/rag/compare_embeddings_output.py` and `cli/bench/categories/agentic_compare.py`.

Callers and tests import each symbol from its owning module; the split adds no compatibility
re-export layer. `scripts/code_quality.sh` now prints production `.py`/`.sh` soft-limit findings
separately from the all-files list, whose longer scenario-ledger tests remain visible without being
mistaken for production modules. No tracked shell file currently exceeds the 250-line soft target.
The context, report, episode, retrieval, authority, and embedding verdict modules named above are
all at or below 250 lines after the split. Focused regression suites, Ruff, and full-source mypy
pass.

The two production modules still above 300 lines are cohesive soft-limit exceptions:
`quality/gpu_guard/surface.py` keeps one exhaustive declaration table beside the interpreter
enumeration that audits it, while `quality/gpu_guard/reach/coverage.py` keeps the documented
stdlib classification partition beside its coverage record and filesystem classifiers. Splitting
either lookup/classifier family would add navigation without creating an independent functional
owner. The remaining production findings are within 47 lines of the soft target and stay visible
for future seam-driven cleanup.

**Every check above only sees files the repo can see, so `.gitignore` is part of the gate.** The
packaging rules are anchored to the repo root (`/build/`, `/dist/`, `/lib/`, `/var/`, `/target/`,
...) because an unanchored directory rule matches that name at ANY depth. Unanchored `build/` hid
`tests/llb/build/test_build_helper.py` -- the only test of `llb_max_jobs`, the canonical
parallelism cap AGENTS.md names for heavy CUDA builds -- from every clone: pytest collected it and
`make ci` ran it on the one box that held the file, while `git ls-files` did not know it existed
and GitHub CI never ran it. Nothing failed, which is the whole problem. The rule now matches only
where setuptools/uv actually write (heavy build trees go under `$DATA_DIR`, covered by `.data/`),
the two `!src/llb/build/` negations that used to patch the rule one package at a time are gone, and
the file is committed. `tests/llb/quality/test_ignored_sources.py` holds the invariant: no `.py`,
`.sh`, or `.md` under `src/`, `tests/`, or `scripts/` may be ignored, and the packaging rules must
still catch root build output. The same blindness reaches the shell gate, which scans
`git ls-files --cached --others --exclude-standard` -- a script under an ignored tree is not linted
either. `git status --ignored --short` is the manual read.

**All four of those build tests run in the non-slow suite**, so `make ci`, `make ci-github`, and
GitHub CI verify the vLLM installer's behavior -- the prebuilt path installs `--only-binary :all:`
through uv's shared cache and writes no project wheelhouse, and the source path exports exactly one
ABI-keyed wheel from a clean checkout -- not only the two `llb_max_jobs` / `common.sh` assertions.
Two of them used to carry `@pytest.mark.slow` because an end-to-end run cost ~5.6s on this host.
The cost was not a resolver call the fake `uv` fails to intercept: it was the flashinfer sampler
preflight that `llb.build.vllm.main` ends in, an in-process probe that imports torch and
JIT-builds + launches a kernel on the real GPU (5.3s measured; the other run was fast only by
accident, because its fake `torch` fixture shadows the real one). GitHub CI would never have paid
it -- torch is absent there, so the probe raises `ImportError` and the verdict is `native` in
milliseconds -- which is exactly why the marker cost coverage where the code is trusted from while
buying nothing. `_seed_sampler_preflight_verdict` in `tests/llb/build/test_build_helper.py` now
pre-records a verdict with `driver: None`, which `verdict_is_current` accepts on every host, so the
installer reuses the cached verdict and the tests stay about wheels and uv calls: 5.6s -> 0.08s and
0.52s -> 0.14s. A unit suite that reaches the GPU is the general hazard here -- an end-to-end run of
an installer entrypoint executes everything that entrypoint does, including probes no shell stub
sits in front of.

That hazard is what the next page is about: the non-slow tier's no-GPU and no-download promises are
enforced by autouse guards rather than trusted, in
[the tier guard](no-gpu-tier-guard.md) and re-measured in
[interpreter reach coverage](interpreter-reach-coverage.md).
