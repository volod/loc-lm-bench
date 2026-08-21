"""Artifact names, defaults, and the comparison vocabulary of the record-linkage seam."""

METHOD = "link-records"
LINKAGE_SUBDIR = "linkage"

SETTINGS_FILE = "settings.json"
MATCH_PARAMETERS_FILE = "match_parameters.json"
BLOCKING_COUNTS_FILE = "blocking_counts.json"
PAIRS_FILE = "pairs.jsonl"
CLUSTERS_FILE = "clusters.jsonl"
# Written only when a reviewer label table was supplied -- an operating point exists only where
# labels do, so an absent file is the honest report of an unlabelled domain.
ACCURACY_FILE = "accuracy.json"
MODEL_FILE = "model.json"

# The re-scoring artifact: a saved Splink model re-scores the same pairs without re-fitting.
REPLAY_FILES = (MODEL_FILE, SETTINGS_FILE)

DEFAULT_UNIQUE_ID_COLUMN = "unique_id"
DEFAULT_MATCH_THRESHOLD = 0.9
# `estimate_u_using_random_sampling` draws this many random pairs for the non-match parameters.
DEFAULT_MAX_PAIRS = 1_000_000
DEFAULT_SEED = 7
DEFAULT_EM_MAX_ITERATIONS = 25
DEFAULT_EM_CONVERGENCE = 1e-4
# Splink's own prior for "two records drawn at random match". Overridden per run from the spec.
DEFAULT_RANDOM_MATCH_PROBABILITY = 1e-4
# DuckDB's parallel aggregations sum in thread-completion order, so a multi-threaded fit lands
# last-ULP differences in the m/u estimates and therefore in every published probability. One
# thread is what makes two fits of the same table BYTE-identical, which is the whole point of a
# seam whose output is meant to be replayable; raise it per run only when a table is large enough
# that the trade is worth stating in `settings.json`.
DEFAULT_DUCKDB_THREADS = 1

# Expectation-maximisation on a small table has a degenerate boundary solution: a comparison level
# no pair of the match class exhibits gets m driven to machine zero, and every pair landing on such
# a level then scores at the SAME floor weight. The ranking below the top class collapses into one
# tie, which is precisely the part of it a reviewer reads. Flooring m and u at a small pseudo-count
# says "nobody observed this level", not "this level cannot occur among matches", and restores the
# order. Off by default (0.0): a fit that trains every level does not need it, and turning it on
# per run keeps every reading taken without it reproducible.
DEFAULT_MIN_LEVEL_PROBABILITY = 0.0

# The linkage boundary: a method that combines several weak agreement signals needs several
# fields. One column is a single-feature threshold wearing a probability, which is what this
# capability exists to replace.
MIN_COMPARISONS = 2

# Comparison-level vocabulary. One seam serves three domains because these cover the fields all
# three carry; adding a kind means adding a row here and a builder in `comparisons.py`.
KIND_EXACT = "exact"
KIND_LEVENSHTEIN = "levenshtein"
KIND_JARO_WINKLER = "jaro_winkler"
KIND_ARRAY_INTERSECT = "array_intersect"
KIND_JACCARD = "jaccard"
KIND_DATE_DIFFERENCE = "date_difference"
KIND_COSINE = "cosine"
KIND_SET_OVERLAP = "set_overlap"

COMPARISON_KINDS = (
    KIND_EXACT,
    KIND_LEVENSHTEIN,
    KIND_JARO_WINKLER,
    KIND_ARRAY_INTERSECT,
    KIND_JACCARD,
    KIND_DATE_DIFFERENCE,
    KIND_COSINE,
    KIND_SET_OVERLAP,
)
# Kinds whose comparison levels are cut points: a spec that omits them has no levels to score.
KINDS_NEEDING_THRESHOLDS = tuple(k for k in COMPARISON_KINDS if k != KIND_EXACT)
# Kinds whose thresholds are similarity scores in (0, 1]; the rest are distances/sizes.
KINDS_WITH_SCORE_THRESHOLDS = (KIND_JARO_WINKLER, KIND_JACCARD, KIND_COSINE, KIND_SET_OVERLAP)
# The one kind whose ladder is built from TWO measures of the same set pair, so it is the one kind
# `containment_thresholds` applies to (see `ComparisonSpec`).
KINDS_WITH_CONTAINMENT = (KIND_SET_OVERLAP,)
# Kinds whose thresholds must be whole numbers (edit distance, intersection size, day count).
KINDS_WITH_INTEGER_THRESHOLDS = (KIND_LEVENSHTEIN, KIND_ARRAY_INTERSECT, KIND_DATE_DIFFERENCE)
# Splink already emits an exact-match level above every ladder, so a zero edit distance or a
# zero-size intersection is that same level written twice -- and a duplicated level never fires,
# which shows up much later as an m value that would not train.
KINDS_WITH_POSITIVE_THRESHOLDS = (KIND_LEVENSHTEIN, KIND_ARRAY_INTERSECT)

DATE_METRICS = ("second", "minute", "hour", "day", "month", "year")
DEFAULT_DATE_METRIC = "day"
# The seam stores dates as text so a JSONL record table round-trips byte-for-byte.
DEFAULT_DATE_FORMAT = "%Y-%m-%d"

# DuckDB column types implied by each comparison kind (the record table's DDL is derived, not
# guessed, so the same JSONL always produces the same table).
DUCKDB_TYPES = {
    KIND_EXACT: "VARCHAR",
    KIND_LEVENSHTEIN: "VARCHAR",
    KIND_JARO_WINKLER: "VARCHAR",
    KIND_ARRAY_INTERSECT: "VARCHAR[]",
    KIND_JACCARD: "VARCHAR",
    KIND_DATE_DIFFERENCE: "VARCHAR",
    # The cosine column is the exception: DuckDB's `array_cosine_similarity` is defined on
    # FIXED-size arrays, so the type carries the embedding dimension (see `column_types`).
    KIND_COSINE: "DOUBLE[{dimension}]",
    KIND_SET_OVERLAP: "VARCHAR[]",
}

# A column an EXPLODING blocking rule generates candidates from is an array in the record table
# whatever else it is, so its DDL type is fixed rather than derived from a comparison kind.
EXPLODED_COLUMN_TYPE = "VARCHAR[]"

RECORDS_TABLE = "llb_records"
LABELS_TABLE = "llb_labels"

EXTRA_HINT = 'uv pip install -e ".[linkage]"'
SPLINK_MISSING = "ERROR: the record-linkage seam needs the [linkage] extra. Run: " + EXTRA_HINT

# `estimate_m_from_pairwise_labels` reads every row of its input as a match, so the positives-only
# view of the reviewer label table lives under this suffix beside the full one.
MATCH_LABELS_SUFFIX = "_matches"

# The labelled accuracy curve is reported at rounded match weights; without rounding a large label
# set produces one row per distinct weight, which is a chart's problem, not a report's.
ACCURACY_WEIGHT_ROUNDING = 0.5

# Column names Splink's own clustering SQL introduces (`connected_components.py`): the final
# cluster table selects `cc.cluster_id` beside every input column, so a record table carrying one
# of these names produces an ambiguous-reference binder error deep inside the clustering step,
# long after the fit succeeded. The seam refuses them up front instead.
RESERVED_COLUMNS = ("cluster_id", "node_id", "representative")
