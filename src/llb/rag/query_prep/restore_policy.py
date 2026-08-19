"""The restoration constraints' design constants, as one sweepable policy object.

`restore` ships three constants that were chosen to be conservative rather than measured: how far
a candidate's re-noised surface may sit from the form the user typed, how short a token has to be
before a length change or an unresolved tie is refused, and whether morphology or local query
context breaks a distance tie first. Each one trades recovered recall against a wrong rewrite of
the user's question, and neither side of that trade is knowable without measuring it, so the three
live here as one frozen policy the sweep can vary and a run config can pin.

The defaults below ARE the shipped behavior; an unset policy is exactly the behavior that was in
place before the policy existed.
"""

from dataclasses import dataclass
from typing import Final

# A candidate must reproduce the typed surface EXACTLY under the reversed transform. The lossy
# part of romanization (dropped soft sign / apostrophe) is what leaves several corpus surfaces
# compatible at all; anything beyond that is a different word, not a restoration of this one. A
# budget of 1 additionally admits a token that was BOTH transliterated and mistyped.
SURFACE_MAX_DISTANCE = 0
# At or below this length a distance-1 neighborhood is dense with unrelated function words, so an
# otherwise-unresolved tie is refused instead of broken alphabetically, and an insertion/deletion
# candidate is refused outright: at three or four characters, dropping or adding a letter yields a
# DIFFERENT short word (`якв` -> `кв`, `зто` -> `то`) rather than a repair of this one. Only a
# transliteration provenance licenses a short length change, because that is exactly the character
# romanization is known to have dropped (the soft sign and the apostrophe).
AMBIGUOUS_TOKEN_MAX_CHARS = 4

# Which signal breaks an edit-distance tie first. `morphology` asks whether the candidate is a real
# word form that keeps the typed ending before asking whether the query's other tokens co-occur
# with it; `context` asks the corpus first and lets morphology break what context leaves tied.
RANK_MORPHOLOGY: Final = "morphology"
RANK_CONTEXT: Final = "context"
RESTORATION_RANK_ORDERS: Final[tuple[str, ...]] = (RANK_MORPHOLOGY, RANK_CONTEXT)

# Bounds for a swept or operator-set value. A surface budget above two admits candidates no lossy
# transform could have produced; a cutoff above eight would lock the length of ordinary long words.
SURFACE_MAX_DISTANCE_LIMIT = 2
AMBIGUOUS_TOKEN_MAX_CHARS_LIMIT = 8


@dataclass(frozen=True)
class RestorationPolicy:
    """The three constants the restoration constraints are parameterized by."""

    surface_max_distance: int = SURFACE_MAX_DISTANCE
    ambiguous_token_max_chars: int = AMBIGUOUS_TOKEN_MAX_CHARS
    rank_order: str = RANK_MORPHOLOGY

    def __post_init__(self) -> None:
        if not 0 <= self.surface_max_distance <= SURFACE_MAX_DISTANCE_LIMIT:
            raise ValueError(
                f"surface_max_distance must be between 0 and {SURFACE_MAX_DISTANCE_LIMIT}"
            )
        if not 0 <= self.ambiguous_token_max_chars <= AMBIGUOUS_TOKEN_MAX_CHARS_LIMIT:
            raise ValueError(
                f"ambiguous_token_max_chars must be between 0 and {AMBIGUOUS_TOKEN_MAX_CHARS_LIMIT}"
            )
        if self.rank_order not in RESTORATION_RANK_ORDERS:
            raise ValueError(
                f"rank_order must be one of {list(RESTORATION_RANK_ORDERS)}, "
                f"got {self.rank_order!r}"
            )

    @property
    def context_first(self) -> bool:
        """Whether local query context outranks morphology on an edit-distance tie."""
        return self.rank_order == RANK_CONTEXT

    @property
    def label(self) -> str:
        """Stable one-line setting id used as a report row label and an artifact key."""
        return (
            f"surface={self.surface_max_distance},"
            f"short={self.ambiguous_token_max_chars},"
            f"rank={self.rank_order}"
        )

    def as_metadata(self) -> dict[str, object]:
        """The policy as plain JSON values for a report header or a run artifact."""
        return {
            "surface_max_distance": self.surface_max_distance,
            "ambiguous_token_max_chars": self.ambiguous_token_max_chars,
            "rank_order": self.rank_order,
        }


DEFAULT_RESTORATION_POLICY = RestorationPolicy()
