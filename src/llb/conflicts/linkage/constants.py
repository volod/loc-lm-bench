"""Column names, agreement ladders, floors, and artifact names of the edition-linkage lane."""

from llb.conflicts.constants import REL_DUPLICATE

LINKAGE_MODE = "corpus-edition-linkage"
SUMMARY_FILE = "edition_summary.json"
RECORDS_FILE = "document_records.jsonl"
EDITIONS_FILE = "editions.jsonl"

# Record-table columns. `shingles` is the document's WHOLE shingle set, which is what the two
# overlap measures are defined on; `block_shingles` is the discriminative subset the audit's own
# inverted index blocks on, so the candidate list is the lexical tier's and not a second one.
DOC_ID_COLUMN = "unique_id"
SHINGLES_COLUMN = "shingles"
BLOCK_SHINGLES_COLUMN = "block_shingles"
TITLE_COLUMN = "title"
SOURCE_SYSTEM_COLUMN = "source_system"
EFFECTIVE_DATE_COLUMN = "effective_date"
# One value for every record, so the rule it names generates every pair. Expectation-maximisation
# needs a rule it can run, and no ordinary agreement blocks a re-ingested edition against its
# predecessor: an edition changes its date, and it very often changes its source system too.
BLOCK_KEY_COLUMN = "block_key"
BLOCK_KEY_VALUE = "corpus"
RETAIN_COLUMNS = (BLOCK_KEY_COLUMN,)

# The second rung of each overlap ladder sits this far below the tier's own cutoff, so the fit can
# price a pair the cutoff rejects instead of collapsing every rejected pair into one level.
LADDER_STEP = 0.3
TITLE_SIMILARITY_THRESHOLDS = (0.9, 0.7)
# Day gaps: about a year, then about four. An edition is dated later than the document it revises,
# so a date GAP is the ordinary case and an exact match is the re-upload.
DATE_GAP_DAYS = (400, 1500)

# See `llb.linkage.constants.DEFAULT_MIN_LEVEL_PROBABILITY`: at corpus scale the match class is a
# handful of pairs, and without a floor every pair outside it scores at one tie.
LEVEL_PROBABILITY_FLOOR = 1e-4

# What share of a corpus's true duplicate pairs the hash tier is assumed to settle, used to turn
# its settled count into the prior "two random documents are the same document". Deliberately
# pessimistic: the hash tier catches only byte-identical and normalization-equivalent copies, and
# an edition that changed one sentence is not either.
HASH_TIER_ASSUMED_RECALL = 0.5

# Below this the lane declines rather than publish u estimates drawn from a handful of pairs -- the
# same floor and the same reason as the gold-item shadow lane.
MIN_LINKAGE_DOCUMENTS = 20
# Above this the pair table is larger than a reviewer would read and the exploded blocking join is
# the run's dominant cost.
MAX_LINKAGE_DOCUMENTS = 1000
# The exploded blocking table's row count: every discriminative shingle of every document becomes
# one row before the self-join.
MAX_EXPLODED_SHINGLES = 2_000_000

# The relations a MATCH PROBABILITY is a probability of. `subsumed_by` is deliberately absent: a
# note a regulation absorbed whole is evidence about two DIFFERENT documents, so a fit that leaves
# it below the identity cut has not disagreed with the tier that reported it.
DUPLICATE_RELATIONS = (REL_DUPLICATE,)

# How many ranked pairs the console summary and the report section print.
REPORT_EXAMPLES = 10
