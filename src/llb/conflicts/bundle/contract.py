"""The registry seam for the conflict bundle's per-document record.

The record keeps the compact integer schema it has always written -- a bundle is read by this
project and by nothing else, and re-encoding every archived one to carry a string identity would
change bytes that a store generation is fingerprinted against for no reader's benefit. What
changes is that the integer is now one encoding of a REGISTERED version: this module maps between
the two, so a bundle at any form reaches the current contract through the registered migration,
and a form this build does not know is refused by name instead of read as an empty record.
"""

from pathlib import Path

from llb.artifacts.data_prep.families import contract_version, local_stage_inputs_version
from llb.artifacts.default_registry import DEFAULT_REGISTRY
from llb.artifacts.serialization import stated_sections
from llb.conflicts.bundle.record import SCHEMA_KEY, STAGE_INPUTS_SCHEMA_VERSION
from llb.core.contracts.common import JsonObject
from llb.core.contracts.data_prep.conflicts import STAGE_INPUTS_SCHEMA_ID

# The earliest bundles carry no version key at all; the registry's legacy read version names them.
EARLIEST_LOCAL_SCHEMA_VERSION = 1


def local_version_of(record: JsonObject) -> int:
    """The record's own integer schema, defaulting to the form that predates the version key."""
    version = record.get(SCHEMA_KEY)
    return version if isinstance(version, int) else EARLIEST_LOCAL_SCHEMA_VERSION


def stage_inputs_at_current(record: JsonObject, *, source: str | Path = "<bundle>") -> JsonObject:
    """One stage-inputs record re-encoded at the current form, whatever form it was written at.

    The result is in the bundle's own encoding, so every reader of a finished record --
    `documents_of`, `recorded_inputs`, the stage replay -- reads it unchanged.
    """
    current = DEFAULT_REGISTRY.read_as(
        STAGE_INPUTS_SCHEMA_ID,
        record,
        version=contract_version(local_version_of(record)),
        source=str(source),
    )
    fields: JsonObject = stated_sections(current)
    fields.pop("schema_id")
    return {
        SCHEMA_KEY: local_stage_inputs_version(str(fields.pop("schema_version"))),
        **fields,
    }


def is_current_form(record: JsonObject) -> bool:
    """Whether the record is already at the form this build writes."""
    return local_version_of(record) == STAGE_INPUTS_SCHEMA_VERSION
