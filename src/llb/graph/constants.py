"""Constants for the Milestone 6 GraphRAG knowledge-graph + narrative retrieval backend.

Named so the build, community detection, and retrieval strategies share one source of truth
(AGENTS.md: avoid magic numbers).
"""

# Retrieval backend + strategy identifiers (recorded per run in the manifest).
BACKEND_GRAPH = "graph"
STRATEGY_LOCAL_KHOP = "local_khop"
STRATEGY_GLOBAL_COMMUNITY = "global_community"
STRATEGIES = (STRATEGY_LOCAL_KHOP, STRATEGY_GLOBAL_COMMUNITY)

# Serialized-chunk record kinds (the `metadata.kind` of an emitted offset-bearing context).
KIND_NODE_MENTION = "node_mention"  # an entity mention span
KIND_EDGE_FACT = "edge_fact"  # an SRO-fact evidence span

# Morphology-aware entity linking (M6 residual 1). A Ukrainian name and its inflected forms
# (Франко -> Франка / Франком / Франкові) differ only in the ending, so linking ALSO matches on a
# shared leading stem of this length -- two tokens whose first MIN_STEM_LEN chars agree share a
# prefix of at least that length. Keeps the linker pure + deterministic (no embedder, no lemmatizer).
MIN_STEM_LEN = 4
# A morphological (stem) link is worth less than an exact token match, so exact hits still rank
# first; a node still links on a stem match alone (the recall gain on inflected questions).
STEM_MATCH_WEIGHT = 0.5
# Tiny per-node confidence tie-break folded into the link score (does not outweigh a token match).
CONFIDENCE_TIE_BOOST = 0.01

# local_khop defaults.
DEFAULT_KHOP_DEPTH = 2  # hops expanded around the entity-linked seed nodes
DEFAULT_N_SEED_NODES = 5  # max question-linked seed nodes per query

# global_community defaults.
DEFAULT_N_COMMUNITIES = 2  # max question-relevant communities serialized per query

# Community detection (deterministic, seeded label propagation -- no graph-analytics dep).
COMMUNITY_MAX_ITERS = 20  # label-propagation passes before forcing convergence
COMMUNITY_SEED = 13  # tie-break seed so a corpus always partitions identically

# Persisted store layout (under the config's graph_dir()).
NODES_FILE = "nodes.jsonl"
EDGES_FILE = "edges.jsonl"
META_FILE = "graph_meta.json"
SUMMARIES_FILE = "community_summaries.json"  # tagged DIAGNOSTIC; never span-scored
DUCKDB_FILE = "graph.duckdb"
