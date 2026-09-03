"""The data-prep directories described as datasets: a staged corpus and a draft bundle.

The gold rows index into the corpus copy, the provenance names the corpus version, the citation
sidecars locate a span on a page -- none of them mean anything read alone. This module names the
project-owned members of each directory; `llb.artifacts.datasets` binds, reads, surveys, and
upgrades them.
"""

from pathlib import Path

from llb.artifacts.datasets import MemberSpec, describe_dataset
from llb.artifacts.default_registry import DEFAULT_REGISTRY
from llb.artifacts.registry import ContractRegistry
from llb.core.contracts.artifacts import DatasetManifest

CORPUS_DATASET_ID = "llb-staged-corpus"
DRAFT_DATASET_ID = "llb-draft-bundle"
PDF_CITATION_SUFFIX = ".citations.json"
OVERLAY_RELATIVE_PATH = ".llb/conflict_overlay.json"

CORPUS_MEMBERS: tuple[MemberSpec, ...] = (
    MemberSpec("corpus-manifest", "corpus_manifest.json", "llb.corpus-manifest"),
    MemberSpec("pdf-manifest", "pdf_corpus_manifest.json", "llb.pdf-corpus-manifest"),
    MemberSpec("pdf-quality", "pdf_corpus_quality.json", "llb.pdf-corpus-manifest"),
    MemberSpec("conflict-overlay", OVERLAY_RELATIVE_PATH, "llb.conflict-overlay"),
)

DRAFT_MEMBERS: tuple[MemberSpec, ...] = (
    MemberSpec("gold-items", "goldset.jsonl", "llb.gold-item", "jsonl"),
    MemberSpec("gold-chains", "chains.jsonl", "llb.gold-chain", "jsonl"),
    MemberSpec("ontology", "ontology.json", "llb.ontology"),
    MemberSpec("extraction", "extraction.jsonl", "llb.ontology-extraction", "jsonl"),
    MemberSpec("provenance", "provenance.json", "llb.ontology-provenance"),
    MemberSpec("import-report", "import_report.json", "llb.external-draft-provenance"),
    MemberSpec("item-provenance", "item_provenance.jsonl", "llb.external-draft-item", "jsonl"),
)


def corpus_bundle_manifest(
    corpus_root: Path | str, registry: ContractRegistry = DEFAULT_REGISTRY
) -> DatasetManifest:
    """Describe a staged corpus, including one citation sidecar member per converted PDF."""
    root = Path(corpus_root)
    citations = tuple(
        MemberSpec(f"citations-{path.name}", path.name, "llb.pdf-citations")
        for path in sorted(root.glob(f"*{PDF_CITATION_SUFFIX}"))
    )
    return describe_dataset(
        root,
        CORPUS_DATASET_ID,
        "One staged corpus: its ingestion manifests, citation sidecars, and applied overlay.",
        (*CORPUS_MEMBERS, *citations),
        registry,
    )


def draft_bundle_manifest(
    bundle_dir: Path | str, registry: ContractRegistry = DEFAULT_REGISTRY
) -> DatasetManifest:
    """Describe a draft bundle: its gold rows, chains, ontology, extraction, and provenance."""
    return describe_dataset(
        Path(bundle_dir),
        DRAFT_DATASET_ID,
        "One draft bundle: the drafted items and everything that says what produced them.",
        DRAFT_MEMBERS,
        registry,
    )
