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
from llb.prompt_system.knowledge_tree_render import KnowledgeTreeRender, render_knowledge_tree
from llb.prompt_system.knowledge_tree_source import KnowledgeTreeSource
from llb.prompt_system.manifest import prompt_system_id
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


def with_knowledge_tree_variants(
    baseline: list[TemplateFields],
    *,
    depths: list[int],
    budgets: list[int],
) -> list[TemplateFields]:
    """Keep every no-tree control and add the depth/budget cartesian variants."""
    grid = list(baseline)
    for fields in baseline:
        for depth, budget in product(depths, budgets):
            variant = replace(
                fields,
                knowledge_tree_depth=depth,
                knowledge_tree_budget=budget,
            )
            variant.validate()
            grid.append(variant)
    return grid


def generate_candidates(
    corpus: CorpusPackage,
    grid: list[TemplateFields],
    budget: ContextBudget,
    tokenizer: Tokenizer,
    *,
    knowledge_tree_source: KnowledgeTreeSource | None = None,
) -> list[PromptCandidate]:
    """Render one budget-fitted candidate per grid point (deduped by prompt-system id)."""
    candidates: list[PromptCandidate] = []
    seen: set[str] = set()
    tree_cache: dict[tuple[int, int], KnowledgeTreeRender] = {}
    for fields in grid:
        tree_text = ""
        tree_report: dict[str, object] = {}
        if fields.knowledge_tree_depth > 0:
            if knowledge_tree_source is None:
                raise ValueError("knowledge-tree fields need a loaded knowledge-tree source")
            key = (fields.knowledge_tree_depth, fields.knowledge_tree_budget)
            rendered = tree_cache.get(key)
            if rendered is None:
                rendered = render_knowledge_tree(
                    knowledge_tree_source,
                    depth=fields.knowledge_tree_depth,
                    budget_tokens=min(fields.knowledge_tree_budget, budget.prompt_budget),
                    tokenizer=tokenizer,
                )
                tree_cache[key] = rendered
            tree_text = rendered.text
            tree_report = rendered.report()
            tree_report["requested_budget_tokens"] = fields.knowledge_tree_budget
            control_fields = replace(
                fields,
                knowledge_tree_depth=0,
                knowledge_tree_budget=0,
            )
            tree_report["baseline_prompt_system_id"] = prompt_system_id(corpus, control_fields)
        candidate = make_candidate(
            corpus,
            fields,
            budget,
            tokenizer,
            knowledge_tree_text=tree_text,
            knowledge_tree_report=tree_report,
        )
        if candidate.prompt_system_id in seen:
            continue
        seen.add(candidate.prompt_system_id)
        candidates.append(candidate)
    return candidates
