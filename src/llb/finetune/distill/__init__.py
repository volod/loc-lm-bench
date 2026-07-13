"""Local text-level distillation from a stronger local teacher into a student LoRA adapter.

The lane is deliberately control-plane first: teacher generation, adapter training, and adapter
comparison are injectable, so CI uses fakes while a CUDA host uses the same backend, trainer, guard,
and registry seams as `run-eval` and `self-improve`.

Submodules (import from the specific one you need -- there is no re-export surface): `model`
(filenames + gate defaults + the record dataclasses + fn seams), `gate` (load items + apply the
quality gate), `dataset_io` (write the teacher/reference training sets), `defaults` (the real
backend teacher/trainer/comparison implementations), `artifacts` (manifest + report), and `run`
(the `run_distillation` orchestrator).
"""
