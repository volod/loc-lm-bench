# Inference config templates

Source of truth for `llb gen-serving-config`. Tier entries live in
[manifest.yaml](manifest.yaml); shell/YAML bodies are rendered from
[templates/](templates/).

Do not copy serve commands into docs -- generate artifacts under
`.data/llb/serving/gpu-<tier>gb/` instead (see
[docs/inference/config-example.md](../../docs/inference/config-example.md)).

Supported GPU tiers: **12, 16, 24, 32** GiB VRAM.
