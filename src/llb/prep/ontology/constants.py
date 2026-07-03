"""Tunable constants for the ontology-assisted gold-set drafting pipeline.

Kept in one place (AGENTS.md: no magic numbers) so the stage modules read declaratively and
a single edit re-tunes the pipeline.
"""

from llb.goldset.schema import Provenance

# --- stage 1: inventory ----------------------------------------------------------------------
SUPPORTED_SUFFIXES = (".txt", ".md")

# --- stage 2: extraction ---------------------------------------------------------------------
# Cap how much of a long document is sent to the extractor in one call (chars). Documents
# longer than this are CHUNKED into overlapping windows (verified-data hardening) -- each window is extracted and
# the per-window extractions merged -- so a long doc is no longer one truncated call. Offsets
# still index the full original text (grounding runs against the full doc).
EXTRACT_MAX_CHARS = 12000
EXTRACT_CHUNK_OVERLAP = 600  # overlap between extraction windows so a span on a seam survives
# Extraction windows per document; 1 preserves deterministic sequential calls.
EXTRACT_CONCURRENCY = 1

# --- stage 3: ontology induction -------------------------------------------------------------
# A "constrained" candidate: keep only the most-supported types, and drop hapax types.
MAX_ENTITY_TYPES = 24
MAX_RELATION_TYPES = 32
MIN_TYPE_COUNT = 1
N_TYPE_EXAMPLES = 3
# Confidence blends normalized count with normalized DOCUMENT frequency (verified-data hardening): a type spread
# across documents is more reliable than one of equal count concentrated in a single document.
CONFIDENCE_COUNT_WEIGHT = 0.5
CONFIDENCE_DOCFREQ_WEIGHT = 0.5
# The high-confidence induced types carried into the drafting prompt as explicit constraints.
ONTOLOGY_CONSTRAINT_MIN_CONFIDENCE = 0.5
N_CONSTRAINT_TYPES = 8

# --- stage 4: coverage sampling --------------------------------------------------------------
DEFAULT_MAX_ITEMS = 60  # upper bound on drafted QA items per run
# Difficulty heuristic: short, frequent evidence is easy; long or rare evidence is hard.
DIFFICULTY_EASY_MAX_CHARS = 80
DIFFICULTY_HARD_MIN_CHARS = 200
RARE_RELATION_MAX_COUNT = 1  # a relation seen this many times or fewer is "rare" -> harder

# --- stage 5: drafting -----------------------------------------------------------------------
# Window of context (chars on each side of the evidence span) handed to the drafter.
DRAFT_CONTEXT_RADIUS = 600

# --- output ----------------------------------------------------------------------------------
METHOD_DIR = "prepare-goldset"  # $DATA_DIR/prepare-goldset/<timestamp>/
GOLDSET_FILENAME = "goldset.jsonl"
ONTOLOGY_FILENAME = "ontology.json"
EXTRACTION_FILENAME = "extraction.jsonl"
PROVENANCE_FILENAME = "provenance.json"
CORPUS_DIRNAME = "corpus"
PDF_ONTOLOGY_REPORT_FILENAME = "pdf_ontology_report.json"
PROMPT_DICTIONARY_FILENAME = "prompt_dictionary_candidates.jsonl"
NEEDLE_GOLDSET_FILENAME = "needle_items.jsonl"
PROMPT_DICTIONARY_MAX_EXAMPLES = 5
# Per-document, per-window extraction journal + its settings sidecar. The journal lets an
# interrupted multi-hour draft resume the extraction stage instead of re-spending model calls;
# the meta sidecar pins the determinism-critical settings so `--resume` reproduces the same
# windows, seeds, and kept items.
EXTRACTION_JOURNAL_FILENAME = "extraction_journal.jsonl"
EXTRACTION_JOURNAL_META_FILENAME = "extraction_journal.meta.json"
EXTRACTION_JOURNAL_META_KIND = "extraction-journal-meta"

PROVENANCE_KIND: Provenance = "ontology-drafted"
