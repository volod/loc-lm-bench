"""Multi-model fine-tuning campaign orchestration.

The campaign runner schedules the existing adapter self-improvement ingredients across a roster:
base/tuning/final evals, shared SFT export, per-model preference export, trainer seam, VRAM reclaim,
and a resumable JSONL journal. Heavy collaborators are injectable so CI can exercise the control
plane without launching models or training stacks.

Submodules (import from the specific one you need -- there is no re-export surface): `coerce`
(small value coercers), `model` (filenames/verdicts + fn seams + record dataclasses), `entry`
(per-model skip probes + round loop + finalization), `defaults` (the real backend seam
implementations), `state` (resumable journal I/O), `report` (`_write_report` + `latest_campaign`),
and `run` (the `run_finetune_campaign` orchestrator).
"""
