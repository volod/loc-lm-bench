"""Tunable constants for the M4.4 ontology-assisted gold-set drafting pipeline.

Kept in one place (AGENTS.md: no magic numbers) so the stage modules read declaratively and
a single edit re-tunes the pipeline.
"""

from llb.goldset.schema import Provenance

# --- stage 1: inventory ----------------------------------------------------------------------
SUPPORTED_SUFFIXES = (".txt", ".md")

# --- stage 2: extraction ---------------------------------------------------------------------
# Cap how much of a long document is sent to the extractor in one call (chars). Documents
# longer than this are truncated for extraction; offsets still index the full original text.
EXTRACT_MAX_CHARS = 12000

# --- stage 3: ontology induction -------------------------------------------------------------
# A "constrained" candidate: keep only the most-supported types, and drop hapax types.
MAX_ENTITY_TYPES = 24
MAX_RELATION_TYPES = 32
MIN_TYPE_COUNT = 1
N_TYPE_EXAMPLES = 3

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

PROVENANCE_KIND: Provenance = "ontology-drafted"
