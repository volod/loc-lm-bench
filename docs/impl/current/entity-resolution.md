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
| `llb/linkage/spec.py` | The typed comparison/blocking specification and its validation |
| `llb/linkage/constants.py` | Artifact names, defaults, the comparison vocabulary, DuckDB types |
| `llb/linkage/records.py` | JSONL record and label reading, derived column types, DuckDB registration |
| `llb/linkage/comparisons.py` | Vocabulary -> Splink comparison creators; the deterministic model id |
| `llb/linkage/fitting.py` | Blocking counts, u/m training, the labelled accuracy curve |
| `llb/linkage/engine.py` | Fit or replay a record table into scored pairs and clusters |
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
| `array_intersect` | shared elements of an alias list | whole sizes >= 1 | `VARCHAR[]` |
| `date_difference` | absolute difference in `date_metric` units | whole numbers >= 0 | `VARCHAR` |
| `cosine` | embedding cosine similarity | scores in (0, 1] | `DOUBLE[dimension]` |

Column types are DERIVED from the kinds rather than inferred from the data, so the same JSONL
always materialises the same table. Two consequences worth knowing: a `cosine` comparison must
declare its `dimension`, because DuckDB's `array_cosine_similarity` is defined on fixed-width
arrays; and dates are stored as text (`input_is_string`), so a record table round-trips byte for
byte.

The specification also refuses what the method cannot price, before any table is read: fewer than
two comparisons (a one-feature threshold is exactly what this seam replaces), two comparisons on
one column, a comparison on the identifier column, no blocking rule, repeated thresholds, and a
zero `levenshtein` or `array_intersect` threshold (which silently repeats the exact-match level and
shows up much later as an m value that would not train). A column a blocking expression references
without comparing goes in `retain_columns`; a blocking rule over a column the table does not carry
fails with the list of columns it could have used.

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

## Reproducibility

The seam's output is meant to be replayable, so two things are pinned:

- **One DuckDB thread** (`duckdb_threads`, default 1). Parallel aggregations sum in
  thread-completion order, which lands last-ULP differences in the m/u estimates and therefore in
  every published probability. Single-threaded, two fits of the same table are byte-identical.
  Raise it per run only when a table is large enough that the trade is worth stating in
  `settings.json`.
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
- the bundle holds every documented artifact, and an unlabelled run writes no accuracy curve.

## Boundary

Linkage needs records carrying several weakly correlated fields. It is not applied to a single
free-text column, it is not a retriever, and it does not rank chunks for a query. Chunk-level
near-duplicate collapse stays in [the retrieval store's](rag-core/retrieval-store.md) collapse
tiers, and linkage does not replace the exact and normalized hash tiers that already settle the
cases they settle for free. The Spark, Athena, and Postgres Splink backends are out of scope, as
are Splink's interactive HTML charts -- the numbers behind them are recorded instead.
