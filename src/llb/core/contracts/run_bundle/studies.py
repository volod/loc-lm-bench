"""Study records: what a benchmark study predeclared, and what it read out afterwards.

A prospective design is the whole point of these studies -- the sample, the families, the effect
the run must reach, and the gates that decide adoption are fixed BEFORE the run, and a design
edited afterwards is not a design. So the durable form of both records is the study's own JSON
object, byte-for-byte: a published aggregate is cited by digest, and re-encoding the record to
carry an identity would move the bytes the citation resolves against.

The identity therefore lives on the model rather than in the file, exactly as the conflict bundle
in [data-prep contracts] keeps its integer version: the producer builds and validates this
contract and then writes the local form, and a reader stamps the identity back on before dispatch.
What the contract adds over an unread JSON blob is the part a cross-cutting reader needs -- which
study a record belongs to, which of the two records it is, and that the body is an object at all.
"""

from typing import Final, Literal

from pydantic import Field, JsonValue

from llb.core.contracts.artifacts import ArtifactContract
from llb.core.contracts.common import JsonObject

STUDY_DESIGN_SCHEMA_ID: Final = "llb.study-design"
STUDY_ANALYSIS_SCHEMA_ID: Final = "llb.study-analysis"

# The key a study's local form uses for its own integer version, and the two identity keys a
# stamped record carries. A local design states its version; an analysis states none.
LOCAL_VERSION_KEY = "schema_version"


class StudyRecord(ArtifactContract):
    """One study record: the study it belongs to and the body that study owns."""

    study_id: str = Field(min_length=1)
    study_kind: str | None = None
    # The integer version the local form carries, where it carries one. A design states it and is
    # refused by its own validator when it does not match; an analysis is written beside the design
    # that produced it and states nothing, so this is absent there.
    local_version: int | None = Field(default=None, ge=1)
    # The predeclared parameters or the resolved readings, as a record or as a table of them.
    # Its keys belong to the study that wrote them -- one study declares a task-family floor and a
    # paired cost gate, the next declares a seed roster and an adoption rule -- so the contract
    # binds WHOSE they are rather than inventing a union no reader could hold to.
    body: JsonObject | list[JsonValue]


class StudyDesignDocument(StudyRecord):
    """What the study fixed before it ran: sample, roster, effect, and adoption gates."""

    schema_id: Literal["llb.study-design"] = STUDY_DESIGN_SCHEMA_ID
    schema_version: Literal["1.0.0"] = "1.0.0"


class StudyAnalysisDocument(StudyRecord):
    """What the study read out of the run it had already committed to."""

    schema_id: Literal["llb.study-analysis"] = STUDY_ANALYSIS_SCHEMA_ID
    schema_version: Literal["1.0.0"] = "1.0.0"
