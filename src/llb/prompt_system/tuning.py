"""RAG prompt-system comparison prompt-tuning loop -- search prompt-system variants over the editable knobs.

Enumerates a deterministic grid over the tuning axes the plan names -- prompt variant (role /
instruction), metadata density, graph-reference style, and anthology size -- rendering one
manifest-addressable candidate per combination so their benchmark scores can be compared by
prompt-system id without ever mixing prompt systems. Pure + deterministic; the actual scoring rides
the normal benchmark path.
"""

from dataclasses import replace
from itertools import product

from llb.prompt_system.budget import ContextBudget, Tokenizer
from llb.prompt_system.corpus import CorpusPackage
from llb.prompt_system.review import PromptCandidate, make_candidate
from llb.prompt_system.template import (
    METADATA_DENSITIES,
    TemplateFields,
)


def variant_grid(
    base: TemplateFields,
    *,
    anthology_sizes: list[int],
    metadata_densities: list[str] | None = None,
    graph_styles: list[str],
) -> list[TemplateFields]:
    """The cartesian grid of template fields to search (deduped, stable order)."""
    densities = metadata_densities if metadata_densities is not None else list(METADATA_DENSITIES)
    seen: set[tuple[int, str, str]] = set()
    grid: list[TemplateFields] = []
    for size, density, style in product(anthology_sizes, densities, graph_styles):
        key = (size, density, style)
        if key in seen:
            continue
        seen.add(key)
        fields = replace(
            base,
            anthology_size=size,
            metadata_density=density,
            graph_reference_style=style,
        )
        fields.validate()
        grid.append(fields)
    return grid


def generate_candidates(
    corpus: CorpusPackage,
    grid: list[TemplateFields],
    budget: ContextBudget,
    tokenizer: Tokenizer,
) -> list[PromptCandidate]:
    """Render one budget-fitted candidate per grid point (deduped by prompt-system id)."""
    candidates: list[PromptCandidate] = []
    seen: set[str] = set()
    for fields in grid:
        candidate = make_candidate(corpus, fields, budget, tokenizer)
        if candidate.prompt_system_id in seen:
            continue
        seen.add(candidate.prompt_system_id)
        candidates.append(candidate)
    return candidates
