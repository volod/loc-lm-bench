"""The governance field names a staged corpus records, in one place.

Two lanes write these fields and two more read them back, so the tuples live apart from the
readers: `governance.py` renders a row, `fingerprints.py` folds the fingerprinted subset into the
manifest item row, and the store meta publishes the full set. A module holding only names has no
imports of its own, which is what lets both sides depend on it.

`OPERATOR_GOVERNANCE_FIELDS` are authored on this side -- by an operator's defaults, a sidecar, or
front matter. `ACQUIRED_GOVERNANCE_FIELDS` are rendered by an upstream acquisition service into the
per-document sidecar described by `docs/design/acquired-corpus-projection.md`; this project
reads them and never authors them.
"""

OPERATOR_GOVERNANCE_FIELDS = (
    "language",
    "version",
    "effective_date",
    "ingestion_time",
    "source_system",
    "acl_label",
)

ACQUIRED_GOVERNANCE_FIELDS = (
    "source_uri",
    "capture_time",
    "capture_id",
    "payload_digest",
    "licence",
    "acquisition_run_id",
    "revision_of",
)

GOVERNANCE_FIELDS = OPERATOR_GOVERNANCE_FIELDS + ACQUIRED_GOVERNANCE_FIELDS

# `ingestion_time` records when THIS project staged the document -- a local event that must not
# move a corpus fingerprint, so it is the one governance field the fingerprinted item row omits.
LOCAL_GOVERNANCE_FIELDS = ("ingestion_time",)

FINGERPRINTED_GOVERNANCE_FIELDS = tuple(
    field for field in GOVERNANCE_FIELDS if field not in LOCAL_GOVERNANCE_FIELDS
)
