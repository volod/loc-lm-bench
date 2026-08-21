"""Column names, agreement ladders, and lane limits of the gold-item linkage specification."""

ROLE_PRIOR = "prior"
ROLE_CANDIDATE = "candidate"

# Compared columns.
QUESTION_VECTOR_COLUMN = "question_vector"
ANSWER_VECTOR_COLUMN = "answer_vector"
SOURCE_DOC_COLUMN = "source_doc_id"
SPAN_BLOCKS_COLUMN = "span_blocks"
QUESTION_TYPE_COLUMN = "question_type"

# Retained columns: provenance a reviewer needs beside a proposed drop, plus the constant column
# the blocking rule compares every pair through.
ITEM_ID_COLUMN = "item_id"
ROLE_COLUMN = "role"
SPLIT_COLUMN = "split"
BLOCK_KEY_COLUMN = "block_key"
BLOCK_KEY_VALUE = "gold-item"

# Source-span character overlap is priced as shared blocks of this many characters. A coarse grid
# keeps two citations of one sentence agreeing when their offsets differ by a word, which raw
# offset equality would miss, and it turns a continuous overlap into the discrete set intersection
# the comparison vocabulary already scores.
SPAN_BLOCK_CHARS = 50

QUESTION_COSINE_THRESHOLDS = (0.95, 0.9, 0.8)
ANSWER_COSINE_THRESHOLDS = (0.95, 0.85)
SPAN_BLOCK_SIZES = (3, 1)

# The blocking rule compares every pair (all records agree on the constant block key), because a
# drafted paraphrase of a prior question is not required to share a document, a type, or a span --
# the shipped constant compares against every prior question, and a shadow lane that scored fewer
# pairs could not reproduce its decisions. The cost is therefore the table size, and this is the
# cap past which the lane declines rather than generating a pair table nobody asked for.
MAX_SHADOW_RECORDS = 1500

# Below this the lane declines instead of fitting: `estimate_u_using_random_sampling` draws its
# non-match parameters from random pairs, and a handful of records has neither the pairs to draw
# nor the level coverage to train m on -- the numbers would be noise wearing a probability's
# clothes.
MIN_SHADOW_RECORDS = 20

SHADOW_MODE = "gold-item-dedup-shadow"
