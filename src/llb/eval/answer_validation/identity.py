"""Which surfaces the answer gate treats as ONE thing, and on whose authority.

Every false rejection this gate has produced so far is an IDENTITY failure rather than an axiom
failure: the axiom was right and the two surfaces it compared were the same thing written twice.
So identity is decided HERE, in one place, by three folds ordered by how much the corpus itself
vouches for each:

  1. **The alias the extraction ledger RECORDS.** The corpus's own statement that two surfaces
     name one entity. An alias two different entity names both claim is dropped rather than
     resolved -- collapsing an ambiguous surface would invent an identity no reviewer accepted.
  2. **The node cluster the entity-resolution lane PROPOSES**
     (`llb.graph.resolution.overlay`), when a run supplies an overlay. Reusing that overlay is the
     point: the graph lane already decides which nodes are one entity, and a second notion of it
     here would let the gate refuse an answer the graph merged.
  3. **Value equivalence** for the three types whose members are values
     (`llb.eval.answer_validation.equivalence`). This is what an alias map structurally cannot
     carry: nothing gives an extractor a reason to record `2,9 млн осіб` as an alias of
     `2.9 мільйона осіб`, and every `functional` / `inverse_functional` / `max_cardinality` axiom
     breaks on that gap wherever a value has more than one written form.

Two rules hold across all three. **Every fold lands on a surface the CORPUS uses**, never on a
synthetic canonical form, so a violation the gate reports still quotes text a reviewer can find in
the documents. And **only the ANSWER's endpoints are folded** -- the corpus ledger is left exactly
as it was recorded, because the gate subtracts the violations the corpus already had on its own
(`gate._answer_violations`) and rewriting that side would change what "the corpus already broke
this" means.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

from llb.eval.answer_validation.equivalence import (
    VALUE_TYPES,
    Lemmatizer,
    default_lemmatizer,
    value_key,
)
from llb.prep.ontology.models import DocExtraction
from llb.prep.ontology.naming import normalize_name

_LOG = logging.getLogger(__name__)


class SurfaceIdentity:
    """The corpus's aliases, an optional proposed node overlay, and value equivalence, as one fold.

    Built once per run beside the ledger: the maps are corpus-wide because an alias and a merge
    are identities the corpus states once, not evidence a retrieved chunk has to repeat.
    """

    def __init__(
        self,
        extractions: Sequence[DocExtraction],
        *,
        overlay: Path | str | None = None,
        lemmatize: Lemmatizer | None = None,
    ) -> None:
        self._lemmatize = lemmatize
        self._aliases = _alias_map(extractions)
        self._overlay = _overlay_map(overlay)
        self._values = self._value_map(extractions)

    @property
    def n_aliases(self) -> int:
        return len(self._aliases)

    @property
    def n_merged_surfaces(self) -> int:
        """How many surfaces the supplied overlay folds onto another one (0 without an overlay)."""
        return len(self._overlay)

    @property
    def n_values(self) -> int:
        """How many distinct VALUES the corpus's typed value surfaces resolve to."""
        return len(self._values)

    def fold(self, name: str, entity_type: str = "") -> str:
        """The surface the corpus records this endpoint under, given the type it was declared as.

        The value fold is TYPE-SCOPED and runs last: `1990 рік` read as a `DATE` and read as a
        `QUANTITY` are different claims, and an endpoint whose value does not parse keeps whatever
        the alias and overlay folds made of it.
        """
        folded = self._by_name(name)
        if entity_type in VALUE_TYPES:
            key = value_key(folded, entity_type, self._lemmatizer())
            if key is not None:
                return self._values.get(key, folded)
        return folded

    def _lemmatizer(self) -> Lemmatizer:
        """Resolved on first use: a ledger with no value-typed endpoint never loads an analyzer."""
        if self._lemmatize is None:
            self._lemmatize = default_lemmatizer()
        return self._lemmatize

    def _by_name(self, name: str) -> str:
        """The alias fold, then the overlay's proposed canonical for whatever it produced."""
        folded = self._aliases.get(normalize_name(name), name)
        return self._overlay.get(normalize_name(folded), folded)

    def _value_map(self, extractions: Sequence[DocExtraction]) -> dict[str, str]:
        """value key -> the one corpus surface every written form of that value folds onto.

        A key claimed by two different names is RESOLVED here rather than dropped, which is the
        opposite of the alias rule and deliberate: two surfaces with the same value key are the
        same value -- that is what a value key means -- while two entities claiming one alias are
        an ambiguity the corpus never settled. The pick is the smallest folded surface, so the
        choice is deterministic and both sides of a comparison make it identically.
        """
        claims: dict[str, set[str]] = {}
        for extraction in extractions:
            for entity in extraction.entities:
                if entity.type not in VALUE_TYPES:
                    continue
                for surface in (entity.name, *entity.aliases):
                    key = value_key(surface, entity.type, self._lemmatizer())
                    if key:
                        claims.setdefault(key, set()).add(self._by_name(surface))
        return {key: min(names) for key, names in claims.items() if names}


def _alias_map(extractions: Sequence[DocExtraction]) -> dict[str, str]:
    """folded alias surface -> the entity NAME the corpus records it under.

    Corpus-wide on purpose: an alias is an identity the corpus states once, not evidence the
    retrieved chunk has to repeat. An alias claimed by two different names is dropped rather than
    resolved -- collapsing an ambiguous surface would invent the very identity a reviewer has not
    accepted.
    """
    claims: dict[str, set[str]] = {}
    for extraction in extractions:
        for entity in extraction.entities:
            for surface in (entity.name, *entity.aliases):
                key = normalize_name(surface)
                if key:
                    claims.setdefault(key, set()).add(entity.name)
    return {key: next(iter(names)) for key, names in claims.items() if len(names) == 1}


def _overlay_map(overlay: Path | str | None) -> dict[str, str]:
    """folded member surface -> the canonical surface the resolution overlay proposes.

    The overlay is READ, never computed here: producing one is the entity-resolution lane's job
    and its threshold is a decision that lane records. A run that supplies none folds on the
    corpus's own aliases exactly as before.
    """
    if overlay is None:
        return {}
    from llb.graph.resolution.overlay import read_overlay_surfaces

    surfaces = read_overlay_surfaces(Path(overlay))
    folded = {
        normalize_name(member): canonical
        for member, canonical in surfaces.items()
        if normalize_name(member) != normalize_name(canonical)
    }
    _LOG.info(
        "[answer-gate] node overlay %s folds %d surface(s) onto a proposed canonical",
        overlay,
        len(folded),
    )
    return folded


__all__ = ["SurfaceIdentity"]
