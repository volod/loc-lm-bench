# Entity Resolution and Record Linkage

Three of this project's record tables carry an identity the pipeline used to settle by a single
hand-set constant: graph entity nodes on a normalized name, drafted gold items on one question
cosine, re-ingested document editions on a content hash or a shingle cutoff. `src/llb/linkage/` is
the shared seam those decisions are meant to move onto: give it a record table, a comparison
specification, and blocking rules, and it returns pairwise match probabilities plus identity
clusters, with the trained model written into the run bundle so the answer can be replayed without
re-fitting.

The confidence contract from
[the spec](../../design/spec.md#entity-resolution-and-record-linkage) is what the seam is FOR, not
a caveat on it. Linkage answers "are these two records the same thing", never "do these two records
contradict each other"; no output of this package may be presented as a conflict verdict. A
proposed cluster never rewrites a corpus, a gold set, or a stored graph -- adopting a merge is a
separate decision taken on retrieval or review evidence.

## What it is built on

Splink 4 (MIT) supplies the Fellegi-Sunter engine on the DuckDB backend
[the graph store](graphrag-backend.md) already requires, so the package is an adapter rather than a
new algorithm. It is installed by the `linkage` extra
(`uv pip install -e ".[linkage]"`, in the default `make venv` EXTRAS); the base install never pulls
it, and `make ci-github` stays green with it absent because every test that needs it is marked
`heavy_env` behind `pytest.importorskip("splink")`.

| Module | Owns |
| --- | --- |
| `llb/linkage/comparison_spec.py` | One compared field's agreement ladder and the ladders it refuses |
| `llb/linkage/spec.py` | The blocking rules, the whole specification, and its validation |
| `llb/linkage/constants.py` | Artifact names, defaults, the comparison vocabulary, DuckDB types |
| `llb/linkage/records.py` | JSONL record and label reading, derived column types, DuckDB registration |
| `llb/linkage/comparisons.py` | Vocabulary -> Splink comparison creators; the deterministic model id |
| `llb/linkage/fitting.py` | Blocking counts, u/m training, the labelled accuracy curve |
| `llb/linkage/engine.py` | Fit or replay a record table into scored pairs and clusters |
| `llb/linkage/clustering.py` | Re-cluster ONE fit's pairs at any threshold, without Splink |
| `llb/linkage/artifacts.py` | The `linkage/` bundle: write it, read it back |
| `llb/linkage/run.py` | Publishing a run and the operator-readable report |
| `llb/cli/linkage.py` | The `llb link-records` command |

Only `comparisons.py`, `fitting.py`, and `engine.py` know Splink's shapes; everything above them
speaks `LinkageSpec` in and `LinkageResult` out, so a consumer of the seam never imports Splink and
never handles a Splink dataframe.

## The comparison vocabulary

One seam serves three domains because these seven kinds cover the fields all three carry. A
comparison names a column, a kind, and the cut points its agreement ladder is built at; Splink adds
an exact-match level above every ladder and an "all other comparisons" level below it.

| Kind | Scores | Thresholds | DuckDB column type |
| --- | --- | --- | --- |
| `exact` | string equality | none | `VARCHAR` |
| `levenshtein` | edit distance | whole numbers >= 1 | `VARCHAR` |
| `jaro_winkler` | prefix-weighted string similarity | scores in (0, 1] | `VARCHAR` |
| `jaccard` | character-shingle set overlap | scores in (0, 1] | `VARCHAR` |
| `set_overlap` | Jaccard of two element sets, then containment of the smaller | scores in (0, 1], in two ladders | `VARCHAR[]` |
| `array_intersect` | shared elements of an alias list | whole sizes >= 1 | `VARCHAR[]` |
| `date_difference` | absolute difference in `date_metric` units | whole numbers >= 0 | `VARCHAR` |
| `cosine` | embedding cosine similarity | scores in (0, 1] | `DOUBLE[dimension]` |

`set_overlap` is the one kind that carries a SECOND ladder over one column
(`containment_thresholds`), because mutual overlap and one-sided coverage measure different
relations between two sets and a short set inside a long one has low Jaccard by construction. Its
levels are emitted mutual-first, so a pair reads the way a caller testing Jaccard before falling
back to containment already reads it.

Column types are DERIVED from the kinds rather than inferred from the data, so the same JSONL
always materialises the same table. Two consequences worth knowing: a `cosine` comparison must
declare its `dimension`, because DuckDB's `array_cosine_similarity` is defined on fixed-width
arrays; and dates are stored as text (`input_is_string`), so a record table round-trips byte for
byte.

The specification also refuses what the method cannot price, before any table is read: fewer than
two comparisons (a one-feature threshold is exactly what this seam replaces), two comparisons on
one column, a comparison on the identifier column, no blocking rule, repeated thresholds, a
zero `levenshtein` or `array_intersect` threshold (which silently repeats the exact-match level and
shows up much later as an m value that would not train), and a column named `cluster_id`,
`node_id`, or `representative` -- Splink's clustering SQL introduces those itself, so a record
table carrying one fails as an ambiguous-reference binder error deep inside the clustering step,
after the model has already trained. A column a blocking expression references
without comparing goes in `retain_columns`; a blocking rule over a column the table does not carry
fails with the list of columns it could have used.

A blocking rule may also EXPLODE an array column (`arrays_to_explode`): two records are compared
when they share at least one element, which is how an inverted index a caller already computes in
Python becomes the rule Splink runs. Such a column is typed `VARCHAR[]` from the rule rather than
from a comparison, and it may not be a TRAINING rule -- Splink refuses an exploding rule for
expectation-maximisation, and it fixes no comparison for the others to be trained against.

## What a run does

1. **Count the blocks.** Every prediction rule's comparison count is recorded BEFORE the fit, so a
   rule that generates more pairs than the host can score is visible as an artifact rather than as
   a run that dies halfway through.
2. **Fit u by random sampling.** `estimate_u_using_random_sampling` at the specification's seed.
   The construction is sound for identity -- two records drawn at random are almost surely not the
   same record -- and it is precisely what is unavailable for contradiction, which is why a
   calibrated linkage probability does not reopen the closed per-pair semantic question.
3. **Fit m.** From reviewer labels when a label table is supplied
   (`estimate_m_from_pairwise_labels`), otherwise by expectation-maximisation, one pass per
   training rule.
4. **Score and cluster.** `predict` over the blocked pairs, then
   `cluster_pairwise_predictions_at_threshold` for the connected-components resolution at the
   specification's `match_threshold`.
5. **Read the operating point.** With labels, `accuracy_analysis_from_labels_table` yields the
   precision/recall curve a threshold is READ OFF rather than picked. The seam keeps the numbers
   and drops Splink's interactive chart. Without labels there is no accuracy file and the report
   says so: the output is a ranked candidate list on the same terms as an unlabelled conflict tier.
6. **Score the run's own cut exactly.** Curve rows sit at rounded match weights, so the number an
   operator acts on is computed from the run's decisions instead, and reported twice: `pairwise`
   (pairs the model scored at or above the cut) and `after clustering` (pairs the run actually
   merged). The second is wider, because connected components merge a pair scoring below the cut
   when a third record links them -- keeping them separate is what makes that visible instead of
   folded into one number. A labelled pair no blocking rule generated counts as a non-match: a
   pair never compared is a merge never proposed.

Levels the fit could not estimate are recorded as `null` and counted in the summary
(`n_untrained_levels`) rather than silently defaulted -- a label set covers only the levels its
matches exhibit, so a label-fitted model legitimately leaves the rest unmeasured.

## Pricing several cuts from one fit

Step 4 resolves the specification's OWN threshold through Splink. A consumer that has to price
several candidate cuts -- read a threshold off a curve, or measure what each cut would do
downstream -- calls `clustering.cluster_pairs(record_ids, pairs, threshold)` instead: the same
connected-components rule (an edge per pair at or above the cut, the smallest member id as the
cluster id) applied to the pairs a fit already published. One fit, many cuts, so a per-threshold
reading varies the threshold and nothing else. Re-fitting per cut would move the model as well,
which is exactly the confound such a reading exists to avoid. A test holds the two together: at
the specification's own threshold `cluster_pairs` reproduces what the engine clustered.

## Reproducibility

The seam's output is meant to be replayable, so two things are pinned:

- **One DuckDB thread** (`duckdb_threads`, default 1). Parallel aggregations sum in
  thread-completion order, which lands last-ULP differences in the m/u estimates and therefore in
  every published probability. Single-threaded, two fits of the same table are byte-identical.
  Raise it per run only when a table is large enough that the trade is worth stating in
  `settings.json`.
- **A pseudo-count floor, off by default.** `min_level_probability` pulls every fitted m and u into
  `[floor, 1 - floor]` before scoring, and a level the fit left as `null` stays null. It exists
  because expectation-maximisation on a small table drives the m of an unobserved level to machine
  zero, which collapses the whole ranking below the top class into one tie; a reading taken without
  it is unchanged, because the default is 0.
- **A derived model id.** Splink's default `linker_uid` is random, which would make two identical
  fits produce two different `model.json` files. It is a hash of the specification instead.

## Artifacts

A run bundle is `$DATA_DIR/<method>/<run>/linkage/` (`link-records` is the default method; a
consumer passes its own so the linkage bundle nests inside that domain's run):

| File | Contents |
| --- | --- |
| `settings.json` | The specification, the run summary, and the caller's metadata |
| `blocking_counts.json` | Comparisons each blocking rule generates, recorded before the fit |
| `match_parameters.json` | Fitted m and u per comparison level (`null` where untrained) |
| `model.json` | The trained model -- the artifact `--replay-from` re-scores from |
| `pairs.jsonl` | Pair, match probability, match weight, per-level agreement |
| `clusters.jsonl` | Cluster id, size, and member record ids at the run's threshold |
| `accuracy.json` | The labelled curve plus the run's own cut scored pairwise and after clustering; written only when labels were supplied |

`settings.json` is the registered `llb.linkage-settings` contract at `2.0.0`: it states every
tuning knob the run used, so a replay re-scores from the run's settings rather than from the
reading build's defaults. A bundle written before the contract left those knobs absent and is
migrated on read. See [data-prep contracts](artifact-contracts/data-prep-contracts.md).

## Commands

```bash
make install-extras EXTRAS=linkage      # or it arrives with the default `make venv`
make link-records                       # the committed sample
make link-records RECORDS=<jsonl> LINK_SPEC=<json> LINK_LABELS=<jsonl>
make replay-linkage RECORDS=<jsonl> LINK_BUNDLE=<a previous run directory>
# Low-level equivalent:
llb link-records --records <jsonl> --spec <json> [--labels <jsonl>] [--replay-from <bundle>]
```

`LINK_METHOD` and `LINK_RUN` name the artifact directory; `LINK_EXAMPLES` sets how many clusters
and pairs the report prints. No GPU and no network.

## The committed sample and what it proves

`samples/linkage/` ([its README](../../../samples/linkage/README.md)) holds 36 Ukrainian
institution records over 12 real entities, the specification they are linked under, and 20 reviewer
labels sampled across the probability range. The variation is what a Ukrainian corpus actually
produces: genitive name forms, Latin `i` homoglyphs for Cyrillic `і`, abbreviated address prefixes,
single-digit registry-code typos, and distinct institutions that share a city and most of their
name.

The fixture's point is that no single feature separates it -- the name similarity of two records of
one entity overlaps the name similarity of two different Lviv universities, and so does the address
similarity. Six weak signals combined do separate it. On the committed specification
(`jaro_winkler` on name, `jaccard` on address, `levenshtein` on registry code, `array_intersect` on
aliases, `exact` on city, `date_difference` on the effective date, threshold 0.9), both the
unsupervised and the label-fitted run recover all 12 entities exactly: every cluster holds one
entity and every entity lands in one cluster, with no cross-entity pair above the threshold. Two of
the twelve are reached only transitively -- a pair scoring 0.69 on its own is merged through a
third record that clears the cut, which a single pairwise threshold could not do.

Each record's `truth_entity_id` rides along in the JSONL but is neither compared nor retained, so
the model never sees it; the tests read it only to check what was recovered.

Against the 19 reviewer labels the curve is flat at precision 1.0 from threshold 0.85 upward and
best F1 (1.0) sits at 0.85. The run's declared 0.9 cut scores precision 1.0 / recall 0.818 pairwise
(two labelled matches fall below it) and precision 1.0 / recall 1.0 after clustering, because those
two are the transitive merges. That gap is the fixture's point restated in the labelled numbers,
and it is also the honest limit of a 19-pair label set: it prices the threshold on this fixture,
not on any production table.

## The gold-item lane

Drafted gold items are the first of the three identity decisions to move onto the seam, and they
move onto it in SHADOW: the shipped policy still decides every drop, and the fitted model is scored
beside it. That is deliberate -- the confidence contract publishes a match probability with the
labelled set it was scored against, and until `entity-merge-labelled-set` produces one there is no
defensible ground on which to flip a default. What the lane changes today is what a drop REPORT
says: a rejection arrives with a match probability, the level agreements behind it, and the prior
item it lost to, instead of one cosine.

| Module | Owns |
| --- | --- |
| `llb/prep/ontology/linkage/records.py` | Gold items -> the record table, and the comparison specification |
| `llb/prep/ontology/linkage/shadow.py` | Running the fit beside the constant and publishing the report |
| `llb/prep/ontology/linkage/verdicts.py` | Both policies' verdict per item, the operating points, the provisional cut |
| `llb/prep/ontology/linkage/agreements.py` | Reading a scored pair back in words; pair and neighbour indexes |
| `llb/prep/ontology/linkage/constants.py` | Column names, agreement ladders, the fit floor and all-pairs cap |

### What a gold item is compared on

| Column | Kind | Ladder | Why it carries information |
| --- | --- | --- | --- |
| `question_vector` | `cosine` | 0.95 / 0.9 / 0.8 | The shipped signal, kept on the pinned E5 embedding so "similar" still means what the retriever sees |
| `answer_vector` | `cosine` | 0.95 / 0.85 | Two distinct questions about one narrow fact share an answer; one question with two answers does not |
| `source_doc_id` | `exact` | -- | A repeat is drafted from the document it repeats |
| `span_blocks` | `array_intersect` | 3 / 1 | Source-span character overlap, priced as shared 50-character grid cells of `<doc-id>:<cell>` so two citations of one sentence agree when their offsets differ by a word |
| `question_type` | `exact` | -- | Weak on its own (roughly half of random pairs agree) and useful in combination, which is the whole method |

`split` is RETAINED and not compared. The drafting pipeline assigns splits after deduplication, so
every candidate carries the same placeholder at drop time; agreement on it would price when a field
gets filled in rather than whether two items are the same question. That is a departure from the
field list the task was written with, and it is the honest one.

One blocking rule compares every pair (all records agree on a constant `block_key`), because the
shipped constant compares a drafted question against every prior question -- a lane that scored
fewer pairs could not reproduce its decisions. The cost is therefore the table size, and it is
bounded at both ends: below `MIN_SHADOW_RECORDS` (20) the lane declines rather than publish u
estimates drawn from a handful of pairs, and above `MAX_SHADOW_RECORDS` (1500, about 1.1M pairs) it
declines rather than generate a pair table nobody asked for. Each decline is reported with its
reason instead of silently skipped. The two expectation-maximisation passes block on
`source_doc_id` and on `question_type`, so each holds one comparison fixed and they cover each
other.

### What the run reports

The drafting bundle's `dedup` block gains a `linkage_shadow` entry, and every row of
`dropped_detail` gains the pair behind it -- `match_probability`, `match_weight`, and `agreements`
naming the level each comparison landed on ("Cosine similarity of question_vector >= 0.95",
"Array intersection size >= 3"). A drop against a prior bundle also names the prior ITEM
(`nearest_prior_id`), which is what makes the decision replayable against the written pair table.

`linkage_shadow` carries the two numbers the policies are compared on -- the lowest match weight
among the items the constant DROPPED and the highest among those it KEPT -- plus their margin, and
a list of operating points. Each operating point states what that cut would decide and lists every
item where the two policies disagree, with the nearest record and the agreements behind it. That
list is the input `entity-merge-labelled-set` samples from.

Both comparisons are made on the match WEIGHT, not the probability. A well-separated fit pushes a
duplicate and a merely-similar item to probabilities that both round to 1.0, so the margin between
them survives only in the log-odds the probability was computed from; the published cut carries
both forms plus `probability_cut_reproduces_shipped_drops`, which says whether the probability form
still decides what the weight does. The provisional cut is the seam's default (0.9) whenever that
already drops exactly what the constant drops, and otherwise the lowest-scoring drop -- the
tightest cut that preserves every shipped decision. It is deliberately NOT the midpoint of the two
weights: non-match weights run to tens of negative bits, so their arithmetic midpoint lands far
below any value an operator would adopt and moves with every unrelated pair added to the table.

### Running it

```bash
make prepare-goldset-draft DRAFT_CORPUS=<dir> DRAFT_MODEL=<model> \
  DRAFT_DEDUP_AGAINST=<prior-bundle> DRAFT_DEDUP_LINKAGE_SHADOW=1
llb prepare-goldset-draft --corpus-root <dir> --model <model> \
  --dedup-against <prior-bundle> --dedup-linkage-shadow
```

The flag requires `--dedup-against` (there is nothing to score against without it) and needs the
`linkage` extra; without it the lane records `enabled: false` with the missing packages rather than
failing the drafting run. The fit lands in the drafting bundle's own `linkage/` directory, with the
same artifact set every linkage run writes, and its `settings.json` metadata records
`mode: gold-item-dedup-shadow` and the provisional cut, so a reader cannot mistake the bundle for
an adopted policy.

### What the fixture runs showed (2026-08-21)

Two runs over the committed gold-set fixtures, both scoring the same drafted batch (exact repeats,
near-paraphrases, one intra-batch repeat, and unrelated questions from a second fixture):

- With the deterministic hashed-bag embedder the tests inject (143 records, 10153 scored pairs), a
  cut exists that reproduces all 10 shipped drops exactly, with a 0.26-bit margin; the seam's
  default 0.9 does NOT -- it would drop 5 more items, each a paraphrase the question cosine alone
  kept while every other field agreed. Those 5 rows are the disagreement list.
- With the pinned E5 embedder on the CUDA host (80 records, 3160 pairs, 12 s including model load),
  the default 0.9 cut reproduces all 12 shipped drops with a 162-bit margin and no disagreement:
  real E5 already scores those paraphrases above the shipped cosine, so the constant dropped them
  too.

The pair of readings is the point: which side the constant and the model differ on depends on the
embedding, and both runs leave the shipped constant in charge. Levels the fit never observed are
reported as `n_untrained_levels` rather than defaulted.

## The graph node lane

Graph entity nodes are the second identity decision to move onto the seam. `_GraphBuilder`
keys a node on `_norm(name)`, a full-string equality test, so in Ukrainian a surname alone, a full
name, an initialed form, and an inflected case form key differently and one entity becomes several
nodes -- splitting its mentions, its degree, and its community membership. That is upstream of
every graph-lane number, because seed linking scores a node by the question tokens it covers and a
node holding a third of its own mentions covers proportionally fewer.

The lane resolves those nodes as an OVERLAY beside the built graph and MEASURES what applying one
would do. Nothing here rewrites a stored graph: the pass proposes canonical clusters, applies each
to a copy, reruns the graph lane over the identical item set at a fixed seed, and reports the
difference. The readings are in
[GraphRAG](graphrag-backend/entity-resolution-evidence.md).

| Module | Owns |
| --- | --- |
| `llb/graph/resolution/records.py` | Graph nodes -> the record table, and the comparison specification |
| `llb/graph/resolution/overlay.py` | The node-cluster overlay: propose it, apply it to a copy, write and read it |
| `llb/graph/resolution/compare.py` | Building the lane set per strategy and running the paired rerun |
| `llb/graph/resolution/verdict.py` | The recommend-or-negative rule and the per-threshold rows |
| `llb/graph/resolution/run.py` | Fitting once, pricing every cut, and the decline reasons |
| `llb/graph/resolution/artifacts.py` | The run bundle, including the retained pre-merge graph |
| `llb/graph/resolution/report.py` | The Markdown artifact and the console summary |
| `llb/cli/rag/graph_resolution.py` | The `llb resolve-graph-entities` command |

### What a graph node is compared on

| Column | Kind | Ladder | Why it carries information |
| --- | --- | --- | --- |
| `name` | `jaro_winkler` | 0.92 / 0.85 / 0.75 | Prefix-weighted, so an inflected ending costs less than a different word |
| `surface_forms` | `array_intersect` | 2 / 1 | Every form the node was seen under, its own name included -- the builder merges an alias onto the node whose NAME matches, so the second node's name is often the first node's alias |
| `entity_type` | `exact` | -- | Weak alone (a third of this corpus's nodes are `MISC`) and informative in combination |
| `doc_ids` | `array_intersect` | 2 / 1 | Two fragments of one entity are cited from the same documents |
| `mention_vector` | `cosine` | 0.97 / 0.94 / 0.9 | The pinned E5 embedding of the node's surface forms plus its mention text -- the only signal that reaches an initialism against its spelled-out form |

The `name` column's exact-match level, which Splink adds above every ladder, is UNREACHABLE by
construction: the builder already keys a node on its normalized name, so two records with an equal
name would be one node. It is reported in `untrained_levels` with that sentence rather than left as
a bare count.

Three blocking rules generate the candidates. `tail_key` and `head_key` are the leading
`MIN_STEM_LEN` characters of the name's last and first token -- the same dependency-free
`graph.linking.morph_key` the question linker already uses, so a candidate pair is generated by
exactly the morphology the retrieval path assumes. The tail stem is what blocks a surname against
its own full name and against every inflected form of either; the head stem blocks the multiword
institution names that agree on their first word instead. `entity_type` is the third rule, and it
is what proposes a merge between two forms sharing no stem at all.

The expectation-maximisation passes block on `entity_type` and on `tail_key`, in that order, and
the order is the finding. A stem key is very nearly a function of the name, so a pass blocked on
one sees no name variation to learn from and lands m BELOW u on the very level a same-entity pair
agrees at -- which on the first CUDA-host attempt scored every real duplicate at a negative match
weight. Blocking on the type leaves the name free to vary, and the tail-stem pass then trains the
type the first pass held fixed.

The lane declines rather than publish noise below 20 nodes (too few pairs for the u estimate) and
above 3000 (the entity-type rule is near-quadratic within a type). Each decline is written with its
reason.

### The overlay and what applying it does

An overlay names, per candidate cut, which node ids the model put in one identity and which member
is canonical -- the most grounded (most mentions), then the longest name, then the lowest id, so a
merged node reads as the spelled-out form rather than the initialism. Every cut is re-clustered
from the SAME fit, so a per-threshold reading varies the threshold and nothing else.

Applying an overlay builds a new `KnowledgeGraph`:

- the canonical node carries every member's surface forms as aliases and every member's mention
  spans, deduplicated by exact span -- which is the recall mechanism, because the seed linker keys
  on a node's name plus aliases and serializes all of its mentions;
- edges are remapped onto canonical endpoints and deduplicated by fact plus evidence span; a
  self-loop that survives the remap is KEPT, because its evidence is a grounded fact the serializer
  still emits;
- communities are RE-DETECTED, because merging changes the adjacency label propagation reads and a
  carried `community_id` would describe a graph that no longer exists.

The source graph is never mutated, and the run copies the pre-merge store into the bundle so the
reading can be redone without the overlay.

A written overlay is also readable WITHOUT a graph: `read_overlay_surfaces` returns the member ->
canonical NAME map its rows already carry for their reader. That is what lets a consumer holding
only an extraction ledger reuse the identity this lane proposed instead of inventing a second one
-- the ontology answer gate folds a declared endpoint through it under
`ONTOLOGY_OVERLAY=<overlay.jsonl>` ([RAG core](rag-core/answer-validation.md#which-surfaces-are-one-thing)).

### Artifacts

`$DATA_DIR/graph-entity-resolution/<run>/`:

| File | Contents |
| --- | --- |
| `linkage/` | The seam's standard bundle for the node fit (`mode: graph-node-overlay`) |
| `node_records.jsonl` | The record table the fit read, one row per graph node |
| `overlays/overlay_<cut>.jsonl` | One overlay per priced cut: a header row, then one row per proposed identity with the canonical and member NAMES |
| `comparison.json` | The paired lane rerun, one `compare-retrieval` report per graph strategy |
| `summary.json` | The counts, the per-cut rows with their deltas, and the verdict |
| `resolution_report.md` | The same reading as Markdown |
| `pre_merge_graph/` | The graph the reading was taken over, copied |

### Running it

```bash
make resolve-graph-entities GOLDSET=<goldset.jsonl>
make resolve-graph-entities GOLDSET=<goldset.jsonl> RESOLVE_THRESHOLDS=0.5,0.3,0.1 \
  RESOLVE_WITH_VECTOR=1 CORPUS=<corpus-dir>
llb resolve-graph-entities --goldset <goldset.jsonl> --k 10 [--thresholds 0.99,0.9,0.75,0.6] \
  [--strategies local_khop,global_community] [--no-mention-embeddings] [--with-vector]
```

It reads the built graph store at the config's `graph_dir()` and needs the `linkage` extra;
`--mention-embeddings` (the default) also needs the `[rag]` extra for the pinned embedder, and
`--no-mention-embeddings` drops the cosine comparison rather than zero-filling it.
`--with-vector` adds the built FAISS lane as a reference row that the verdict can never adopt --
this run decides an overlay, not a backend.

## The document-edition lane

Re-ingested document editions are the third identity decision to move onto the seam, and the only
one where the shipped decision is TWO hand-set constants rather than one. The conflict audit's
lexical tier reports a `duplicate` when the shingle Jaccard clears 0.8 and a `subsumed_by` when the
containment of the smaller document clears 0.9 -- two numbers on two scales, so the list a reviewer
opens has no order across the relations in it, and neither cutoff can be priced. The hash tier
below it settles byte-identical and normalization-equivalent documents for free and is untouched by
this lane.

What the lane adds is a probability per document pair and an edition GROUP: the connected component
a document belongs to, in edition order, with the current edition named where the governance fields
can name it. What it does not add is a finding. The audit's relations stay the tiers'; the lane is
opt-in (`--linkage`), runs where the two model-free tiers run, and publishes beside them.

| Module | Owns |
| --- | --- |
| `llb/conflicts/linkage/records.py` | Corpus documents -> the record table, and the comparison specification |
| `llb/conflicts/linkage/run.py` | The prior, the fit, the provisional cut, the decline reasons, the verdict |
| `llb/conflicts/linkage/editions.py` | Clusters read as edition groups, ordered by `compare_editions` |
| `llb/conflicts/linkage/rankings.py` | The two-ranking comparison: recovery, ordering discordance, decisions |
| `llb/conflicts/linkage/artifacts.py` | The `linkage/` subdirectory of the audit bundle |
| `llb/conflicts/linkage/report.py` | The report section and the console lines |

### What a document pair is compared on

| Column | Kind | Ladder | Why it carries information |
| --- | --- | --- | --- |
| `shingles` | `set_overlap` | Jaccard 0.8 / 0.5, then containment 0.9 / 0.6 | The lexical tier's own two measures, over the tier's own word 5-gram shingles, at the run's own cutoffs |
| `title` | `jaro_winkler` | 0.9 / 0.7 | The first heading, case-folded -- an upper-cased re-issue is a title AGREEMENT, not a disagreement |
| `source_system` | `exact` | -- | Weak alone, and informative in combination: a re-ingest often arrives from a different system |
| `effective_date` | `date_difference` | 400 / 1500 days | An edition is dated later than the document it revises, so a gap is the ordinary case and an exact match is the re-upload |

`set_overlap` is the comparison kind the seam grew for this lane: one column, two measures, and the
levels emitted mutual-overlap-first then containment -- the order the lexical tier's own decision
rule already applies. The top rung of each ladder IS that run's cutoff, so the fit and the
thresholds read the same two numbers and a difference between them is a difference in how the
evidence is COMBINED, never in what was measured.

### Blocking is the tier's inverted index, not a second candidate list

One blocking rule, and it EXPLODES an array: two documents are compared when they share at least
one discriminative shingle. That is `candidate_pairs` restated as SQL, so the fit prices exactly the
pairs the lexical tier examines -- a test asserts the two agree in Python and a second asserts the
engine scores that many pairs. Banded LSH is still not an option here for the reason the tier
documents: a short note inside a long document has low Jaccard by construction.

The record table therefore carries two shingle arrays. `shingles` is the WHOLE set, which is what
the two overlap measures are defined on; `block_shingles` is the doc-frequency-filtered subset the
inverted index blocks on. Splitting them is what keeps the measured numbers identical to the tier's
while the candidate list stays identical to the tier's too.

Expectation-maximisation trains on ONE rule that generates every pair. No ordinary agreement blocks
a re-ingested edition against its predecessor -- an edition changes its date and often its source
system -- so a training rule over either would hold fixed the very field an edition varies. (Splink
refuses an exploding rule for training outright, and the specification refuses it before Splink
does.)

### The prior, and the floor

Two settings decide whether the published probabilities mean anything at corpus scale, and both are
measured rather than assumed.

**The prior comes from the hash tier.** `probability_two_random_records_match` moves every
probability, and the seam's generic 1e-4 default is orders of magnitude off on a corpus of a few
dozen documents. The pairs the hash tier settled are duplicates nobody had to judge, so their share
of the corpus's document pairs, divided by the share of duplicates a byte-and-normalization test can
be expected to catch (`HASH_TIER_ASSUMED_RECALL`, 0.5), is the prior. On the planted fixture that
gives 0.0369 against the plant's true duplicate-pair rate of 0.046 -- close enough to price a cut,
and the construction is stated in the bundle so a reader can re-derive it. A corpus the hash tier
settled nothing in falls back to the seam default and says so.

**The floor is what makes a ranking exist.** Expectation-maximisation on a corpus-sized table has a
degenerate boundary solution: a comparison level no pair of the match class exhibits gets m driven
to machine zero, and every pair landing on such a level then scores at the same floor weight. The
lane sets `min_level_probability` to 1e-4, a pseudo-count floor the seam applies to every fitted m
and u before scoring -- "nobody observed this level", not "this level cannot occur among matches".
Measured on the planted fixture, with the floor off the lowest-ranked reported relation sits at rank
135 of 140 and 115 unrelated pairs outrank a reported relation; at any floor from 1e-6 to 1e-2 it
sits at rank 21 with one, and the edition grouping and the cut are the same. The floor decides
whether the ranking exists; its value does not decide the order.

### The cut and the edition groups

The fit resolves the seam's own 0.9 threshold, and the lane then re-clusters the SAME pairs at a
provisional cut: 0.9 whenever that already merges every duplicate the current thresholds report,
and otherwise the lowest probability among those duplicates -- the tightest cut that preserves every
decision the corpus is audited under today. It is deliberately not the midpoint between the classes,
which on a well-separated fit lands far below anything an operator would adopt and moves with every
unrelated pair the corpus gains.

An edition group is a multi-document cluster read as a history: members oldest first under
`compare_editions`, and `current` naming the members no other member is newer than. That is a LIST
because two byte-identical re-uploads carrying one date are one edition held twice -- naming either
would invent a precedence the governance fields do not record. Singletons are not groups. Each
decision group in `groups.json` gains the `edition_groups` its documents fall into, so a reviewer
reading four rows about four files can see they are four editions of one document.

### What the lane does NOT do

A subsumption is not an identity. A note a regulation absorbed whole is scored, ranked, and
reported -- and left out of the regulation's edition group, because the two are different documents.
The lane's verdict is therefore about RECALL only: relations no blocking rule generated, and
duplicates the cut does not merge. Pairs the fit ranks that the thresholds reject are counted and
listed as the candidate surface a ranking adds, not as a disagreement the run settles.

### Artifacts

The audit bundle's `linkage/` subdirectory, which is the seam's standard bundle
(`mode: corpus-edition-linkage`) plus three files:

| File | Contents |
| --- | --- |
| `edition_summary.json` | The prior, the cut, the edition counts, the recovery table, the ordering discordance, the decisions, and the verdict |
| `document_records.jsonl` | The record table, with each shingle array replaced by its size |
| `editions.jsonl` | One row per proposed edition group: members oldest first, the current edition, and the field that ordered it |

The audit's own `summary.json` carries the same reading under `edition_linkage`, without the
per-relation rows and the group list -- those have files of their own.

### Running it

```bash
make audit-corpus-conflicts CORPUS=<corpus-dir> EFFORT=lexical LINKAGE=1
llb audit-corpus-conflicts --corpus <corpus-dir> --effort lexical --linkage
```

It needs the `linkage` extra, no GPU and no network. The lane declines rather than publish noise
below 20 documents (too few pairs to estimate a non-match parameter from, and the levels no pair
exhibits would not train), above 1000 documents, or above 2,000,000 discriminative shingles (the
exploded blocking join). Each decline is written with its reason.

### What the runs showed (2026-08-21)

Three corpora, `--effort lexical --linkage`, under `$DATA_DIR/corpus-conflicts/`:

- **The planted fixture** (`samples/corpora/editions_uk_v1/`, 26 documents, 140 candidate pairs,
  measured 2026-08-21): every one of the 20 relations the current thresholds recover is
  scored, the 15 duplicates among them are all merged, and the clustering forms exactly the plant's
  6 edition families -- 16 documents grouped, largest 4, 5 groups with a current edition the dates
  could name. The 20 relations occupy ranks 1-21 of 140; the one pair between them is the note whose
  containment in the 2019 appeals edition is 0.735, which the tier's 0.9 cutoff rejects. Two
  comparison levels were never observed and are reported rather than defaulted
  (`shingles/Jaccard >= 0.5`, `title/Jaro-Winkler >= 0.9`).
- **The two ranking disagree, and that is the finding.** Over the 110 comparable orderings of those
  20 relations the two rankings disagree on 77 (Kendall tau -0.4). Nearly all of it is one thing:
  the tier scores every planted subsumption at containment 1.0 and every revision at a Jaccard
  around 0.9, so its ordering puts the notes above the editions; the fit puts every duplicate above
  every subsumption. A tier score is a Jaccard for one relation and a containment for the other, so
  the order it implies across relations was never a quantity -- the discordance is measuring that,
  not a disagreement about evidence.
- **The two quickstart corpora decline** (the HR corpus at 8 documents and the goods corpus at 5,
  both measured 2026-08-21). Both are far below the 20-document floor, and both carry
  zero hash and lexical findings, so there was nothing for a fit to reproduce either. That is the
  honest state of this host's operator corpora for a DOCUMENT-pair fit: they are a handful of large
  PDFs, and the method needs a table of documents. The lane records the reason rather than
  publishing an untrainable model.

The reading is adopted for nothing. The lexical tier's two cutoffs still decide every finding, and
the lane publishes a ranking and a proposed grouping beside them.

## Tests

`tests/llb/linkage/` -- the specification, record-table, and result contracts run in the base
install; the fit, replay, artifact, and vocabulary tests are `heavy_env`. What they hold:

- the fit recovers the fixture's known cluster structure, and no cross-entity pair clears the
  threshold;
- two fits of the same table produce byte-identical pairs, clusters, and model;
- a saved model re-scores the same pairs to identical probabilities, including a replay read back
  from a written bundle;
- every comparison kind in the vocabulary fits and scores on a real table (the two the sample does
  not use -- `levenshtein` on a code column and `cosine` on an embedding column -- on a synthetic
  one);
- the labelled run's cut is scored exactly, and the clustering's recall exceeds the pairwise cut's
  on the fixture's transitive merges;
- the bundle holds every documented artifact, and an unlabelled run writes no accuracy curve;
- `cluster_pairs` reproduces what the engine clustered at the specification's own threshold,
  merges a pair scoring below a cut when a third record links them, and never splits a cluster
  as the cut is lowered.

`tests/llb/prep/ontology/linkage/` -- the gold-item lane. The record table, the derived question
type, the span-block grid, and the specification's shape run in the base install; the fixture runs
are `heavy_env`. What they hold: the shadow lane changes no drop (the kept and dropped ids match a
run with the lane off); every shipped drop is scored and a cut exists that reproduces all of them;
every drop row names its agreements and the prior item it lost to; the default cut's disagreement
list is non-empty and complete enough to label from; the bundle holds the fit; and a table below
the fit floor or a batch with no drafted items declines with a reason instead of publishing noise.

`tests/llb/conflicts/linkage/` -- the document-edition lane. The record table, the blocking-column
equality with `candidate_pairs`, the specification's ladders, the prior, the provisional cut, the
edition grouping, and the two-ranking comparison run in the base install; the fit over the planted
corpus is `heavy_env`. What they hold: the exploding blocking column generates exactly the pairs
`candidate_pairs` returns, in Python and in the engine; every relation the thresholds recover is
scored; every duplicate among them is merged; the edition groups ARE the plant's six families and no
group crosses two of them; an absorbed note is scored and ranked and never merged into the document
that absorbed it; every duplicate ranks above every subsumption; two fits publish identical pairs;
the audit's findings are byte-identical with the lane on and off; `groups.json` names edition groups
only when the lane ran; the bundle holds every documented artifact and records shingle COUNTS rather
than shingles; and a corpus below the document floor declines with its reason.

`tests/llb/graph/resolution/` -- the graph node lane. The record table, the blocking keys, the
overlay, the verdict rule, and both report renderings run in the base install; the whole pass over
the planted graph is `heavy_env`. The plant is 31 nodes over 6 fragmented entities plus 16
distractors that share a type, a document, and a leading word with each other, and it deliberately
does NOT plant an epithet nothing else mentions -- resolving that is coreference, which this pass
is not. What the tests hold: one cut recovers the planted clustering EXACTLY; no cut proposes a
cross-entity merge; a tighter cut never merges more than a looser one; the paired rerun scores the
identical items under every lane against the pre-overlay baseline; a recovered overlay costs the
lane neither recall nor MRR; a merged node carries every member's mentions; the bundle holds every
documented artifact including the retained pre-merge graph; applying an overlay never mutates the
source graph; the merged node links on a form only a fragment carried; and a flat reading arrives
labelled as the negative result it is. The floor and cap declines are checked by their reasons.

## Boundary

Linkage needs records carrying several weakly correlated fields. It is not applied to a single
free-text column, it is not a retriever, and it does not rank chunks for a query. Chunk-level
near-duplicate collapse stays in [the retrieval store's](rag-core/retrieval-store.md) collapse
tiers, and linkage does not replace the exact and normalized hash tiers that already settle the
cases they settle for free. The Spark, Athena, and Postgres Splink backends are out of scope, as
are Splink's interactive HTML charts -- the numbers behind them are recorded instead.

The graph lane draws its own boundary inside that one. It produces an overlay BESIDE the built
graph and never rewrites the stored one in place; it does not change the closed node vocabulary or
an entity's typing; it does not merge edges or relations as identities of their own (edges are
remapped onto merged endpoints, which is a consequence of a node merge, not a decision about the
edge); and it is not a coreference model -- a form that agrees with its entity on nothing but the
type and the document is outside what the method can price, and the planted fixture says so.
