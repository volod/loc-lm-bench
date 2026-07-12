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
# Retry malformed or non-object model output once before leaving a window for resume.
EXTRACT_PARSE_RETRIES = 1

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
# Coverage-target drafting (yield-max): instead of a flat item cap, draft up to this many seeds
# per stratum bucket (relation / entity_type / section / semantic kind). When unset, the flat
# `max_items` cap applies. `max_items` still bounds the total as a safety ceiling.
DEFAULT_COVERAGE_TARGET = None  # per-stratum-bucket target; None -> flat-cap mode

# --- question-type + difficulty labels (yield-max) -------------------------------------------
# Closed question-type taxonomy recorded per item (in item provenance / needle rows, NOT the
# GoldItem schema). Reviewers and analyzers filter drafts on these labels.
QUESTION_TYPE_FACTOID = "factoid"
QUESTION_TYPE_DEFINITION = "definition"
QUESTION_TYPE_PROCEDURAL = "procedural"
QUESTION_TYPE_NUMERIC = "numeric"
QUESTION_TYPE_COMPARATIVE = "comparative"
QUESTION_TYPE_MULTI_HOP = "multi-hop"
QUESTION_TYPES = (
    QUESTION_TYPE_FACTOID,
    QUESTION_TYPE_DEFINITION,
    QUESTION_TYPE_PROCEDURAL,
    QUESTION_TYPE_NUMERIC,
    QUESTION_TYPE_COMPARATIVE,
    QUESTION_TYPE_MULTI_HOP,
)
DEFAULT_QUESTION_TYPE = QUESTION_TYPE_FACTOID

# --- multi-hop graph-path seeds (yield-max) --------------------------------------------------
# A 2-hop chain A -r1-> B -r2-> C drafted from the knowledge graph, grounded in the two edges'
# evidence spans (multi-span, cross-section/document). Bounded so a large graph does not explode
# the draft set; deterministic ordering keeps a resume reproducible.
MULTI_HOP_DEPTH = 2
DEFAULT_MULTI_HOP_MAX_PATHS = 40
MULTI_HOP_DIFFICULTY = "hard"  # a chain question is inherently harder than a single-span factoid
MULTI_HOP_MIN_SPANS = 2  # a multi-hop item must carry at least this many grounded spans

# --- near-duplicate suppression against prior bundles (yield-max) -----------------------------
# Drop a drafted question whose pinned-E5 cosine similarity to ANY prior-bundle question exceeds
# this threshold, so a coverage-target rerun does not re-draft paraphrases already reviewed.
NEAR_DUP_COSINE_THRESHOLD = 0.9

# --- item id namespaces ----------------------------------------------------------------------
ONTOLOGY_ID_PREFIX = "onto"  # flat single-span ontology-drafted items
MULTI_HOP_ID_PREFIX = "mhop"  # multi-hop chain items
CHAIN_ID_PREFIX = "chain"  # ordered chain-of-questions items

# --- stage 5: drafting -----------------------------------------------------------------------
# Window of context (chars on each side of the evidence span) handed to the drafter.
DRAFT_CONTEXT_RADIUS = 600
# Output-language gate: require an unambiguous Ukrainian letter and make Cyrillic the clear
# majority. This rejects untranslated source quotations while allowing occasional Latin proper
# names inside otherwise Ukrainian questions and answers.
UKRAINIAN_SPECIFIC_LETTERS = frozenset("іїєґ")
UKRAINIAN_MIN_CYRILLIC_FRACTION = 0.6

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
CHAINS_FILENAME = "chains.jsonl"
PROMPT_DICTIONARY_MAX_EXAMPLES = 5
# Per-document, per-window extraction journal + its settings sidecar. The journal lets an
# interrupted multi-hour draft resume the extraction stage instead of re-spending model calls;
# the meta sidecar pins the determinism-critical settings so `--resume` reproduces the same
# windows, seeds, and kept items.
EXTRACTION_JOURNAL_FILENAME = "extraction_journal.jsonl"
EXTRACTION_JOURNAL_META_FILENAME = "extraction_journal.meta.json"
EXTRACTION_JOURNAL_META_KIND = "extraction-journal-meta"

PROVENANCE_KIND: Provenance = "ontology-drafted"
