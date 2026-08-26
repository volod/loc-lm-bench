# Runtime Version Floors

A host can be correct in every way this checklist measures and still be unable to serve a roster
entry, because the runtime is too OLD to implement that artifact's architecture. Ollama 0.20
answers the Gemma 4 12B GGUF with `unknown model architecture: 'gemma4'`; nothing in that reaches a
run as a host fact -- the launcher records a generic backend error per case, the resolver reports
"source not found", and the entry quietly leaves the roster with no row saying why. The floor is
part of host setup, so it is pinned, checked, and NAMED.

**Where a floor comes from.** Ollama declares it per artifact: `ollama show <tag>` prints a
`requires` line, and `/api/show` returns the same value beside `general.architecture`. That is the
authority whenever the daemon already holds the tag. It cannot answer for a tag that is not pulled
yet -- exactly the case host setup is in -- or for a raw `hf.co/...gguf` reference, which carries
no `requires` field at all, so the manifest pins it per source record:

```yaml
      ollama:
        source: gemma4:12b
        runtime_arch: gemma4            # the architecture the runtime must implement
        min_runtime_version: "0.30.5"   # read from `ollama show gemma4:12b` -> requires
```

Read the pin off the artifact; never guess one. The two signals are merged with the HIGHER floor
winning, so a pin can only raise what the artifact itself demands, and a version that does not
parse (or a daemon that does not answer) produces no skip at all -- a floor check that guessed
would ground a runnable model.

The floors the roster pins today, each read from the artifact on this host:

| source | serves | floor | why that version |
| --- | --- | --- | --- |
| `gemma4:e4b`, `gemma4:26b` | ollama | 0.20.0 | Ollama's own engine served Gemma 4 from its launch |
| `gemma4:12b` | ollama | 0.30.5 | the 12B landed in 0.30.3; its x86/CUDA/Linux crash was fixed in 0.30.5 |
| `hf.co/google/gemma-4-12B-it-qat-q4_0-gguf:Q4_0` | llamacpp | 0.30.5 | the raw QAT GGUF declares no `requires`; pinned to the curated tag's floor, and 0.20 rejected it here |
| `qwen3.8:27b` | ollama | 0.32.12 | `qwen35` architecture support |

**What a host below a floor sees.** Three places, one message, always naming the runtime, the
architecture, and the version required:

- `make resolve-models` prints the candidate as a named skip beside the resolution table --
  `skip gemma-4-12b-it-w4a16 / ollama: ollama 0.20.6 does not implement architecture 'gemma4' --
  gemma4:12b needs ollama >= 0.30.5 (upgrade the host runtime)` -- which is the only place the hole
  is visible when the entry still resolves through another backend;
- `make prep-models` refuses the pull with the same reason instead of spending a multi-GiB download
  on an artifact this host cannot load (`--force` overrides it, for caching ahead of an upgrade);
- the Ollama launcher checks the floor once at `start()` and raises that message, so a doomed run
  fails at launch rather than once per case. A runtime that refuses mid-run anyway is classified
  `architecture_unsupported` rather than the generic `backend_error`, and it is deliberately NOT
  retryable: retrying an old runtime cannot make it younger.

On this host (2026-08-26, Ollama 0.32.15) every floor is satisfied and `resolve-models` prints no
skips -- the state to expect after a normal setup. The pieces:
`src/llb/backends/runtime_floor.py` (floors, merge, the message, the `/api/version` + `/api/show`
probes), the resolver gate in `src/llb/backends/resolver.py` with its probes in
`resolver_probes.py`, the prep gate in `src/llb/backends/prepare/planning.py`, the launcher gate in
`src/llb/backends/ollama.py`; tests in `tests/llb/backends/test_runtime_floor.py`,
`test_resolver_availability.py`, `test_backend_client.py`, and
`tests/llb/backends/prepare/test_prepare_models.py`.
