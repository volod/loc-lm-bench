"""Column names, agreement ladders, lane limits, and artifact names of graph node resolution."""

METHOD = "graph-entity-resolution"
RESOLUTION_MODE = "graph-node-overlay"

# Compared columns: the five signals a fragmented entity still agrees on.
NAME_COLUMN = "name"
SURFACE_FORMS_COLUMN = "surface_forms"
ENTITY_TYPE_COLUMN = "entity_type"
DOC_IDS_COLUMN = "doc_ids"
MENTION_VECTOR_COLUMN = "mention_vector"

# Retained columns: the blocking keys and the provenance a reader needs beside a proposed merge.
NODE_ID_COLUMN = "graph_node_id"
HEAD_KEY_COLUMN = "head_key"
TAIL_KEY_COLUMN = "tail_key"
N_MENTIONS_COLUMN = "n_mentions"

RETAIN_COLUMNS = (
    NODE_ID_COLUMN,
    HEAD_KEY_COLUMN,
    TAIL_KEY_COLUMN,
    N_MENTIONS_COLUMN,
)

# Ukrainian names differ by ENDING, so both blocking keys are morphological stems rather than whole
# tokens: `tail_key` blocks a surname against its own full name and its inflected forms ("Франко" /
# "Іван Франко" / "І. Франко" / "Франка" all end on the same stem), and `head_key` blocks the
# multiword institution names that agree on their first word instead. `entity_type` is the third
# rule: it is what proposes a merge between two surface forms that share no stem at all
# ("ЗСУ" against "Збройні Сили України"), which is the fragmentation the name key cannot see.

NAME_SIMILARITY_THRESHOLDS = (0.92, 0.85, 0.75)
# Alias-array agreement, priced as shared surface forms (the node's own name included, normalized).
# Size 2 is a node agreeing on a name AND an alias; size 1 is the alias-hits-name case that
# fragments an entity in the first place.
SURFACE_FORM_SIZES = (2, 1)
# Documents the node's mentions occur in. Weak alone on a small corpus -- most nodes cite one
# document and many share it -- and informative in combination, which is the method.
DOC_ID_SIZES = (2, 1)
# Embedding cosine over the node's surface forms plus its mention text. E5 similarities are
# compressed high, so the ladder sits where a same-entity pair actually separates on this corpus.
MENTION_COSINE_THRESHOLDS = (0.97, 0.94, 0.9)

# How much node text is embedded: the name, the aliases, then mention text up to this budget. A
# node's mentions are its surface spans, so a handful already carries every form it appears in.
MAX_NODE_TEXT_CHARS = 512

# The candidate cuts a run prices unless told otherwise. The grid spans the whole decision range
# on purpose: 0.99 merges only what nothing else could explain, 0.9 is the seam's own default, and
# 0.6 is close enough to the model's coin flip that a cut which lifts nothing THERE has said
# something about the corpus rather than about the grid.
DEFAULT_THRESHOLDS = (0.99, 0.9, 0.75, 0.6)

# Below this the lane declines rather than publish u estimates drawn from a handful of pairs --
# the same floor the gold-item lane refuses under, for the same reason.
MIN_RESOLUTION_NODES = 20
# Above this it declines rather than block a pair table nobody asked for: the entity-type rule is
# near-quadratic within a type, and a graph this large is a different engineering problem.
MAX_RESOLUTION_NODES = 3000

# Artifacts under `$DATA_DIR/graph-entity-resolution/<run>/`.
RECORDS_FILE = "node_records.jsonl"
OVERLAYS_DIR = "overlays"
OVERLAY_FILE_TEMPLATE = "overlay_{threshold}.jsonl"
COMPARISON_FILE = "comparison.json"
SUMMARY_FILE = "summary.json"
REPORT_FILE = "resolution_report.md"
PRE_MERGE_DIR = "pre_merge_graph"

# Lane labels in the paired comparison.
LANE_BASE_TEMPLATE = "graph/{strategy}"
LANE_OVERLAY_TEMPLATE = "graph/{strategy}+overlay@{threshold}"
LANE_VECTOR = "faiss"
