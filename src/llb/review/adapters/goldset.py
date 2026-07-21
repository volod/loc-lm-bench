"""Goldset-family adapters, including draft comparison and cutoff translation."""

from dataclasses import dataclass
from pathlib import Path

from llb.goldset.verify_acceptance_report import infer_reject_code
from llb.goldset.verify_base import (
    ACCEPT,
    CHECK_COLS,
    PASS,
    REJECT,
    load_worksheet,
)
from llb.goldset.verify_commands import TRANSLATION_PROFILE
from llb.goldset.verify_card import _is_synthetic_row
from llb.goldset.verify_session.commands import _clear_row, _save, _set_decision
from llb.goldset.verify_session.decision import _edit_still_grounds
from llb.goldset.verify_session.loop import _resolve_corpus_root
from llb.review.core import ReviewAction, ReviewAdapter, ReviewRecord
from llb.review.presentation import fields_section

_CHECK_ACTIONS = tuple(
    action
    for key, field, label in (
        ("g", "chk_grounded", "Grounded"),
        ("a", "chk_answerable", "Answerable"),
        ("r", "chk_reference", "Reference"),
        ("p", "chk_planted", "Planted"),
    )
    for action in (
        ReviewAction(key, f"{label} pass", f"check:{field}:pass", "positive"),
        ReviewAction(key.upper(), f"{label} fail", f"check:{field}:fail", "negative"),
    )
)
_ACTIONS = _CHECK_ACTIONS + (
    ReviewAction("y", "Accept", ACCEPT, "positive"),
    ReviewAction("x", "Reject", REJECT, "negative"),
    ReviewAction("c", "Clear", "clear", "neutral"),
)
_DATA_FIELDS = ("question", "reference_answer", "edited_answer", "chain_steps")
_EVIDENCE_FIELDS = ("context", "span_text", "page_citation", "cc_note")
_META_FIELDS = (
    "item_id",
    "item_kind",
    "split",
    "provenance",
    "source_doc_id",
    "stratum",
    "retrieval_rank",
    "chk_grounded",
    "chk_answerable",
    "chk_reference",
    "chk_planted",
    "reject_code",
    "human_note",
)


@dataclass(slots=True)
class _Worksheet:
    path: Path
    label: str
    rows: list[dict[str, str]]
    fieldnames: list[str]
    corpus_root: Path | None


class GoldsetVerifyAdapter(ReviewAdapter):
    """Adapter over one or more verification CSVs using their merge-on-save writer."""

    kind = "goldset-verify"

    def __init__(self, worksheet: Path | str) -> None:
        path = Path(worksheet)
        self.path = path
        self._worksheets = [self._load(path, "verification")]
        self._refs = self._flatten()

    @classmethod
    def _from_paths(
        cls,
        path: Path,
        worksheets: list[tuple[str, Path]],
    ) -> "GoldsetVerifyAdapter":
        instance = cls.__new__(cls)
        instance.path = path
        instance._worksheets = [instance._load(value, label) for label, value in worksheets]
        instance._refs = instance._flatten()
        return instance

    @staticmethod
    def _load(path: Path, label: str) -> _Worksheet:
        rows, fields = load_worksheet(path)
        return _Worksheet(path, label, rows, fields, _resolve_corpus_root(path))

    def _flatten(self) -> list[tuple[_Worksheet, dict[str, str]]]:
        return [(ledger, row) for ledger in self._worksheets for row in ledger.rows]

    @property
    def actions(self) -> tuple[ReviewAction, ...]:
        return _ACTIONS

    def __len__(self) -> int:
        return len(self._refs)

    def record(self, index: int) -> ReviewRecord:
        ledger, row = self._refs[index]
        item_id = row.get("item_id") or str(index + 1)
        stratum = row.get("stratum") or row.get("split") or "all"
        if len(self._worksheets) > 1:
            stratum = f"{ledger.label}: {stratum}"
        return ReviewRecord(
            key=item_id,
            title=f"{ledger.label}: {item_id}",
            sections=(
                fields_section("Record content", row, _DATA_FIELDS, "data"),
                fields_section("Evidence", row, _EVIDENCE_FIELDS, "evidence"),
                fields_section("Metadata and checks", row, _META_FIELDS, "metadata"),
            ),
            stratum=stratum,
            verdict=(row.get("decision") or "").strip(),
        )

    def apply(self, index: int, action: str) -> None:
        ledger, row = self._refs[index]
        if action == ACCEPT:
            self._accept(row, ledger)
        elif action == REJECT:
            _set_decision(row, REJECT)
            row["reject_code"] = infer_reject_code(row)
        elif action == "clear":
            _clear_row(row)
        elif action.startswith("check:"):
            _prefix, field, value = action.split(":")
            if field == "chk_planted" and not _is_synthetic_row(row):
                raise ValueError("planted check is not applicable to a real item")
            row[field] = value
        else:
            raise ValueError(f"unsupported {self.kind} action: {action}")
        _save(ledger.path, ledger.rows, ledger.fieldnames)

    @staticmethod
    def _accept(row: dict[str, str], ledger: _Worksheet) -> None:
        if row.get("review_profile") == TRANSLATION_PROFILE:
            failed = [field for field in CHECK_COLS if row.get(field) not in ("", PASS)]
            if failed:
                raise ValueError("translation acceptance conflicts with failed checks")
            for field in CHECK_COLS:
                if not row.get(field):
                    row[field] = PASS
        if not _edit_still_grounds(ledger.corpus_root, row):
            raise ValueError("edited answer no longer matches its corpus span")
        _set_decision(row, ACCEPT)
        row["reject_code"] = ""


class KnowledgeCutoffAdapter(GoldsetVerifyAdapter):
    """Named adapter for the cutoff translation bundle's verification profile."""

    kind = "knowledge-cutoff-ua"

    def __init__(self, bundle: Path | str) -> None:
        from llb.bench.knowledge_cutoff.translation_artifacts import WORKSHEET_FILENAME

        root = Path(bundle)
        worksheet = root if root.is_file() else root / WORKSHEET_FILENAME
        super().__init__(worksheet)
        self.path = root
        self._worksheets[0].label = "cutoff translation"


class DraftCompareAdapter(GoldsetVerifyAdapter):
    """Composite adapter over every lane named by a comparison report."""

    kind = "draft-compare"

    def __init__(self, comparison: Path | str) -> None:
        from llb.prep.ontology.compare_gate import comparison_worksheets

        value = Path(comparison)
        report = value / "comparison.json" if value.is_dir() else value
        raw_paths = comparison_worksheets(report)
        worksheets = [(label, self._resolve(report, path)) for label, path in raw_paths.items()]
        loaded = self._from_paths(value, worksheets)
        self.path = loaded.path
        self._worksheets = loaded._worksheets
        self._refs = loaded._refs

    @staticmethod
    def _resolve(report: Path, worksheet: Path) -> Path:
        if worksheet.is_absolute() or worksheet.exists():
            return worksheet
        return report.parent / worksheet
