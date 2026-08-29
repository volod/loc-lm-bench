"""Replaying a fold length at the span the fold it stands for actually offered.

A per-family guard fit replays a deterministic walk with a summarizer that writes the family's own
measured number of characters. The measurement is free because the one-fold control already folds
once per case -- but that fold covers the WHOLE ten-entry transcript, and the cell the length is
replayed at folds three- and four-entry spans. A summarizer offered less writes less, so a length
carried across unchanged is a length measured against the wrong offer, and the error it makes is
guard-dependent: a tighter guard folds sooner, over fewer entries, and is wrong by more.

The fix costs no episode either. The ladder already runs a second never-fitted cell whose folds
cover much shorter spans; run it before the fitted cell and the run holds a SECOND measured
(offered span, written length) point. Two points give a SLOPE -- how many characters this family
writes per character it is offered -- and the slope is what turns one measured length into the
length the fitted cell's own span implies.

What the two points do NOT give is a curve, so this module refuses to be one. The slope is applied
only between the spans that were measured; outside them the nearer measured span stands, because
an extrapolated fold length is a number no cell of this run ever offered. And a run that measured
one span only keeps the flat replay it always had, named as such (`SPAN_LENGTH_SINGLE`) rather
than handed a slope of zero pretending to be a measurement.

The per-case LEVEL still comes from the control: each case's own measured fold length says how
verbose that case's summarizer was, and the slope -- one number for the family -- says how that
level moves with the span. Level per case, slope per family, so the fit keeps predicting a case
COUNT rather than collapsing twelve cases into one replayed episode.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

from llb.bench.context_policy.guard_band import median_int

ARM_TYPED_MARKER = "typed_marker"

SPAN_LENGTH_INTERPOLATED = "the_fold_length_is_replayed_across_two_measured_spans"
SPAN_LENGTH_SINGLE = "only_one_fold_span_was_measured_so_the_fold_length_is_replayed_flat"
SPAN_LENGTH_UNMEASURED = "no_fold_span_was_measured_to_replay_a_fold_length_against"


@dataclass(frozen=True, slots=True)
class FoldLengthModel:
    """How long a summary this family writes when a fold offers it a given span.

    `anchor_span` is the span the per-case levels were measured at, so `length_at(level, span)`
    returns the level unchanged there and moves away from it only as far as the measured slope
    and the measured span range allow.
    """

    anchor_span: int
    anchor_length: int
    second_span: int
    second_length: int
    chars_per_offered_char: float
    reading: str

    @property
    def span_low(self) -> int:
        return min(self.anchor_span, self.second_span) if self.second_span else self.anchor_span

    @property
    def span_high(self) -> int:
        return max(self.anchor_span, self.second_span)

    def length_at(self, level: int, offered_chars: int) -> int:
        """The length this family writes at one offered span, from a case's measured level."""
        if self.chars_per_offered_char == 0.0:
            return max(0, level)
        held = min(max(offered_chars, self.span_low), self.span_high)
        return max(0, round(level + self.chars_per_offered_char * (held - self.anchor_span)))

    @classmethod
    def from_record(cls, record: dict[str, object]) -> "FoldLengthModel":
        """Rebuild the model a fit ran with, from the fields it recorded."""
        return cls(
            anchor_span=int(cast(int, record.get("anchor_fold_span_chars", 0))),
            anchor_length=int(cast(int, record.get("anchor_fold_length_chars", 0))),
            second_span=int(cast(int, record.get("second_fold_span_chars", 0))),
            second_length=int(cast(int, record.get("second_fold_length_chars", 0))),
            chars_per_offered_char=float(
                cast(float, record.get("chars_written_per_offered_char", 0.0))
            ),
            reading=str(record.get("fold_span_reading", SPAN_LENGTH_UNMEASURED)),
        )

    def as_record(self) -> dict[str, object]:
        """The fit-record fields that make a replayed length auditable after the fact."""
        return {
            "fold_span_reading": self.reading,
            "anchor_fold_span_chars": self.anchor_span,
            "anchor_fold_length_chars": self.anchor_length,
            "second_fold_span_chars": self.second_span,
            "second_fold_length_chars": self.second_length,
            "chars_written_per_offered_char": round(self.chars_per_offered_char, 5),
            "replayed_span_range": [self.span_low, self.span_high] if self.anchor_span else [],
        }


def shipped_arm_cases(
    rows: list[dict[str, object]], source_cell_id: str
) -> Iterator[dict[str, object]]:
    """Every case one cell measured under the SHIPPED marker-preserving policy.

    The ablation arm is deliberately excluded from every measurement a fit reads: it runs a
    different summarizer, so its lengths belong to a policy the ladder does not ship.
    """
    for row in rows:
        if row["cell_id"] != source_cell_id or row["arm"] != ARM_TYPED_MARKER:
            continue
        yield from cast(list[dict[str, object]], row["cases"])


def measured_fold_points(
    rows: list[dict[str, object]], source_cell_id: str
) -> list[tuple[int, int]]:
    """Every (offered span, written length) pair one cell's shipped-policy arm measured.

    One pair per FOLD: the offered span is what the policy handed the summarizer at that fold and
    the length is what it wrote back, so a three-fold case contributes three points rather than an
    episode-level average that no single fold ever stood at.
    """
    return [
        (int(span), int(chars))
        for case in shipped_arm_cases(rows, source_cell_id)
        for span, chars in zip(
            cast(list[int], case.get("summary_fold_input_chars", [])),
            cast(list[int], case.get("summary_output_chars", [])),
            strict=False,
        )
    ]


def fold_length_span_model(
    anchor: list[tuple[int, int]], second: list[tuple[int, int]]
) -> FoldLengthModel:
    """The family's written-length slope, from the two cells whose folds cover different spans.

    Each cell contributes ONE aggregate point -- the median span its folds offered and the median
    length written against them -- rather than a scatter, because what is wanted is the difference
    between two regimes and not a regression through the noise inside either of them.
    """
    if not anchor:
        return FoldLengthModel(0, 0, 0, 0, 0.0, SPAN_LENGTH_UNMEASURED)
    anchor_span, anchor_length = _aggregate_point(anchor)
    if not second:
        return FoldLengthModel(anchor_span, anchor_length, 0, 0, 0.0, SPAN_LENGTH_SINGLE)
    second_span, second_length = _aggregate_point(second)
    if second_span == anchor_span:
        return FoldLengthModel(
            anchor_span, anchor_length, second_span, second_length, 0.0, SPAN_LENGTH_SINGLE
        )
    slope = (second_length - anchor_length) / (second_span - anchor_span)
    return FoldLengthModel(
        anchor_span, anchor_length, second_span, second_length, slope, SPAN_LENGTH_INTERPOLATED
    )


def _aggregate_point(points: list[tuple[int, int]]) -> tuple[int, int]:
    return median_int([span for span, _chars in points]), median_int(
        [chars for _span, chars in points]
    )
