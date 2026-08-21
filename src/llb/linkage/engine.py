"""Turn a record table into an identity decision: fit or replay, score, cluster.

Together with `fitting.py` this is the only place that knows Splink's shapes. Everything above it
speaks `LinkageSpec` in and `LinkageResult` out, so the three identity decisions that consume the
seam never import Splink and never see a Splink dataframe.
"""

import logging
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any, Iterator

from llb.core.contracts.common import JsonObject
from llb.linkage.comparisons import build_settings, require_splink
from llb.linkage.constants import DEFAULT_DUCKDB_THREADS, SPLINK_MISSING
from llb.linkage.fitting import (
    count_blocking_comparisons,
    labelled_accuracy,
    match_parameters,
    train,
)
from llb.linkage.model import (
    BlockingCount,
    LinkageCluster,
    LinkagePair,
    LinkageResult,
    pairs_above,
    pairs_co_clustered,
    score_labels,
    sorted_clusters,
    sorted_pairs,
)
from llb.linkage.records import LabelTables, ReviewerLabel, register_labels, register_records
from llb.linkage.spec import LinkageSpec

_LOG = logging.getLogger(__name__)

_AGREEMENT_PREFIX = "gamma_"
_CLUSTER_COLUMN = "cluster_id"


@contextmanager
def duckdb_connection(threads: int = DEFAULT_DUCKDB_THREADS) -> Iterator[Any]:
    """An in-memory DuckDB the record table and Splink's working tables share.

    Pinned to one thread by default: see `DEFAULT_DUCKDB_THREADS` for why a parallel fit is not
    byte-reproducible.
    """
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise SystemExit(SPLINK_MISSING) from exc
    con = duckdb.connect(":memory:")
    con.execute(f"SET threads TO {int(threads)}")
    try:
        yield con
    finally:
        con.close()


@contextmanager
def _prepared(
    records: Sequence[JsonObject],
    spec: LinkageSpec,
    settings: Any,
    labels: Sequence[ReviewerLabel] | None = None,
) -> Iterator[tuple[Any, tuple[BlockingCount, ...], LabelTables | None]]:
    """Register the record table, count the blocks, and build the linker over them.

    The prefix a fit and a replay share: both need the same table, the same block counts, and a
    linker -- they differ only in whether the settings they are given were trained yet.
    """
    require_splink()
    from splink import DuckDBAPI, Linker

    with duckdb_connection(spec.duckdb_threads) as con:
        table = register_records(con, records, spec)
        counts = count_blocking_comparisons(con, spec, table)
        _LOG.info(
            "[linkage] %d records, %d comparisons to score",
            len(records),
            sum(count.post_filter for count in counts),
        )
        label_tables = register_labels(con, labels) if labels else None
        linker = Linker(
            table,
            settings,
            db_api=DuckDBAPI(connection=con),
            set_up_basic_logging=False,
            validate_settings=True,
        )
        yield linker, counts, label_tables


def _pairs(rows: Sequence[JsonObject], spec: LinkageSpec) -> tuple[LinkagePair, ...]:
    identifier = spec.unique_id_column
    built = [
        LinkagePair(
            left_id=str(row[f"{identifier}_l"]),
            right_id=str(row[f"{identifier}_r"]),
            match_probability=float(row["match_probability"]),
            match_weight=float(row["match_weight"]),
            agreement={
                key[len(_AGREEMENT_PREFIX) :]: int(value)
                for key, value in row.items()
                if key.startswith(_AGREEMENT_PREFIX) and value is not None
            },
        )
        for row in rows
    ]
    return sorted_pairs(built)


def _clusters(rows: Sequence[JsonObject], spec: LinkageSpec) -> tuple[LinkageCluster, ...]:
    identifier = spec.unique_id_column
    members: dict[str, list[str]] = {}
    for row in rows:
        members.setdefault(str(row[_CLUSTER_COLUMN]), []).append(str(row[identifier]))
    built = [
        LinkageCluster(cluster_id=cluster_id, record_ids=tuple(sorted(ids)))
        for cluster_id, ids in members.items()
    ]
    return sorted_clusters(built)


def _score(
    linker: Any, spec: LinkageSpec
) -> tuple[tuple[LinkagePair, ...], tuple[LinkageCluster, ...]]:
    predictions = linker.inference.predict()
    pairs = _pairs(predictions.as_record_dict(), spec)
    clustered = linker.clustering.cluster_pairwise_predictions_at_threshold(
        predictions, threshold_match_probability=spec.match_threshold
    )
    return pairs, _clusters(clustered.as_record_dict(), spec)


def run_linkage(
    records: Sequence[JsonObject],
    spec: LinkageSpec,
    labels: Sequence[ReviewerLabel] | None = None,
) -> LinkageResult:
    """Fit a linkage model on a record table and resolve it into identity clusters."""
    spec.validate()
    with _prepared(records, spec, build_settings(spec), labels) as (linker, counts, label_tables):
        from_labels = train(linker, spec, label_tables)
        model = linker.misc.save_model_to_json()
        pairs, clusters = _score(linker, spec)
        accuracy = labelled_accuracy(linker, label_tables.judged) if label_tables else ()
    cut = spec.match_threshold
    return LinkageResult(
        spec=spec,
        blocking_counts=counts,
        match_parameters=match_parameters(model),
        pairs=pairs,
        clusters=clusters,
        trained_model=model,
        n_records=len(records),
        trained_from_labels=from_labels,
        accuracy=accuracy,
        pair_operating_point=(
            score_labels(pairs_above(pairs, cut), labels, cut) if labels else None
        ),
        cluster_operating_point=(
            score_labels(pairs_co_clustered(clusters), labels, cut) if labels else None
        ),
    )


def replay_linkage(
    records: Sequence[JsonObject],
    spec: LinkageSpec,
    model: JsonObject,
    *,
    trained_from_labels: bool = False,
) -> LinkageResult:
    """Re-score a record table from a saved model, with no fitting of any kind.

    This is what makes an identity decision reproducible: the same records and the same model
    artifact must give back the same probabilities, whoever runs it and whenever. How the model was
    originally fitted is not recoverable from the model file, so the caller carries it forward from
    the bundle's summary rather than the replay guessing.
    """
    spec.validate()
    with _prepared(records, spec, dict(model)) as (linker, counts, _):
        pairs, clusters = _score(linker, spec)
    return LinkageResult(
        spec=spec,
        blocking_counts=counts,
        match_parameters=match_parameters(model),
        pairs=pairs,
        clusters=clusters,
        trained_model=dict(model),
        n_records=len(records),
        trained_from_labels=trained_from_labels,
    )
