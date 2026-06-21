"""Create a timestamped, editable SQuAD skeleton for a gold set authored from scratch."""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from llb.paths import resolve_data_dir

_LOG = logging.getLogger(__name__)
METHOD = "goldset-skeleton"

SKELETON = {
    "version": "1.0",
    "data": [
        {
            "title": "replace-with-document-title",
            "paragraphs": [
                {
                    "context": "Київ є столицею України.",
                    "qas": [
                        {
                            "id": "replace-with-stable-id-001",
                            "question": "Яке місто є столицею України?",
                            "answers": [{"text": "Київ", "answer_start": 0}],
                        }
                    ],
                }
            ],
        }
    ],
}

INSTRUCTIONS = """Gold-set skeleton

1. Replace the example with source paragraphs you are allowed to evaluate.
2. Write stable, unique question ids and Ukrainian questions.
3. Copy each answer exactly from its context and record its zero-based character offset.
4. Keep one factual claim per question; reject ambiguous or unsupported questions.
5. Import the edited file with `make ingest-squad SQUAD_JSON=<path>`.
6. Review the canonical JSONL, then set verified=true only for accepted items.

Full manual: docs/guides/goldset-from-scratch.md
"""


def create_skeleton(out_root: Path, run_timestamp: str | None = None) -> Path:
    """Write one editable skeleton under `$DATA_DIR/goldset-skeleton/<timestamp>/`."""
    timestamp = run_timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    if Path(timestamp).name != timestamp:
        raise ValueError("run_timestamp must be one path segment")
    out_dir = out_root / METHOD / timestamp
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "squad_goldset.json").write_text(
        json.dumps(SKELETON, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "README.txt").write_text(INSTRUCTIONS, encoding="ascii")
    _LOG.info("[goldset-skeleton] editable template -> %s", out_dir)
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create an editable Ukrainian gold-set skeleton.")
    parser.add_argument("--out-root", type=Path, default=None, help="data root override")
    args = parser.parse_args(argv)
    create_skeleton(resolve_data_dir(args.out_root))
    return 0


if __name__ == "__main__":
    from llb.runtime import run

    sys.exit(run(main))
