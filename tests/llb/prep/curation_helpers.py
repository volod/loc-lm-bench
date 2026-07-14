"""Curation of externally drafted artifacts: merge + repair + filter + dedup (curate-drafts).

A fake deterministic embedder drives the semantic near-dup path -- no sentence-transformers, no
GPU. Fixtures simulate two services (batched fenced exports, overlapping and broken rows).
"""

import json
import zlib

import pytest


DOC = (
    "Розділ 1. Загальні положення про облік матеріальних цінностей.\n"
    "Відповідальною особою призначається начальник служби. "
    "Передача здійснюється протягом п'яти робочих днів. "
    "Акт приймання складається у трьох примірниках. "
    "Відповідальною особою призначається головний бухгалтер."
)


class FakeEmbedder:
    """Hashed bag-of-words embedding (stable across calls); near-dup == high token overlap."""

    DIM = 128

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.DIM
            for w in t.casefold().split():
                v[zlib.crc32(w.encode("utf-8")) % self.DIM] += 1.0
            out.append(v)
        return out


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "doc-a.md").write_text(DOC, encoding="utf-8")
    return root


def _squad_file(tmp_path, name, qas, context=None, title="doc-a.md"):
    payload = {
        "version": "1.0",
        "data": [
            {
                "title": title,
                "paragraphs": [
                    {
                        "context": context if context is not None else DOC,
                        "qas": [
                            {
                                "id": qa["id"],
                                "question": qa["q"],
                                "answers": [{"text": qa["a"], "answer_start": 0}],
                            }
                            for qa in qas
                        ],
                    }
                ],
            }
        ],
    }
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


# --- lenient loading --------------------------------------------------------------------------


# --- squad curation ---------------------------------------------------------------------------


# --- security curation ------------------------------------------------------------------------


def _sec_case(case_id, prompt, **over):
    base = {
        "id": case_id,
        "family": "unsafe_content",
        "prompt": prompt,
        "detector_kind": "refuse",
        "expect_refusal": False,
        "benign": True,
        "lang": "uk",
        "attrs": {"vector": "over-refusal-control", "grounding_hint": "начальник служби"},
    }
    base.update(over)
    return base


# --- chains curation --------------------------------------------------------------------------


def _chain(chain_id, steps):
    return {
        "chain_id": chain_id,
        "lang": "uk",
        "steps": [
            {
                "order": i + 1,
                "question": q,
                "reference_answer": a,
                "source_doc_id": "doc-a.md",
                "quote": quote,
                "dependency_note": "" if i == 0 else "будується на попередньому кроці",
            }
            for i, (q, a, quote) in enumerate(steps)
        ],
    }


# --- inventory curation -----------------------------------------------------------------------


# --- output writing ---------------------------------------------------------------------------


# --- grounded curation (external-draft contract Artifact B) ----------------------------------


def _grounded_file(tmp_path, name, rows):
    path = tmp_path / name
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return path
