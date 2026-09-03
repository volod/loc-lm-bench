"""How a contract record becomes the bytes an artifact file holds.

Two dumps are wanted and they are not the same. A record that STATES an absence -- a document row
whose acquisition provenance the corpus never carried -- must write that `null`, because the row's
whole point is that the question was asked. A record whose optional SECTION is absent must not
write a `null` section, because a run that never reached the coverage pass is not a run that found
no coverage. So the choice is made per record, at the top level only, and never inside a row.
"""

from pydantic import BaseModel

from llb.core.contracts.common import JsonObject


def stated_sections(record: BaseModel) -> JsonObject:
    """The record with its absent top-level sections omitted and every nested field kept."""
    payload: JsonObject = record.model_dump(mode="json")
    return {key: value for key, value in payload.items() if value is not None}
