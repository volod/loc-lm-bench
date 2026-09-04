"""Declare what a category or study bundle publishes beside its manifest and its rows.

Every benchmark in this suite writes the same three shapes of additional file: the design it fixed
before the run, the analysis it read out afterwards, and a rendered comparison a human reads. The
declaration is made ONCE here rather than at each of the twenty-odd call sites, because two studies
declaring the same file differently is exactly the confusion the run contract exists to remove.

The rule is stated on the content, not on the filename. A `.md` file is a rendered report and is
declared exempt with the reason. A `.json` file is a study record: a DESIGN when it predeclares
itself -- naming the study it belongs to and the integer version its validator checks -- and a
reading the study TOOK otherwise. Anything else has no declaration and cannot be published, which
is the refusal that keeps the third shape from quietly appearing.
"""

import json
from collections.abc import Mapping

from llb.artifacts.errors import DatasetReadError
from llb.artifacts.run_bundle.run_artifacts import RunArtifact, human_report, study_artifact
from llb.core.contracts.run_bundle.studies import STUDY_ANALYSIS_SCHEMA_ID, STUDY_DESIGN_SCHEMA_ID

MARKDOWN_SUFFIX = ".md"
JSON_SUFFIX = ".json"


def declared_artifacts(
    artifacts: Mapping[str, str] | None, *, study_id: str | None
) -> tuple[RunArtifact, ...]:
    """Declare each additional file a benchmark publishes, refusing one this rule cannot place."""
    return tuple(_declared(name, content, study_id) for name, content in (artifacts or {}).items())


def _declared(name: str, content: str, study_id: str | None) -> RunArtifact:
    if name.endswith(MARKDOWN_SUFFIX):
        return human_report(name, content)
    if not name.endswith(JSON_SUFFIX):
        raise DatasetReadError(
            f"{name}: a benchmark publishes a JSON study record or a Markdown report, "
            "and this is neither"
        )
    schema_id = STUDY_DESIGN_SCHEMA_ID if _predeclares_itself(content) else STUDY_ANALYSIS_SCHEMA_ID
    if schema_id == STUDY_ANALYSIS_SCHEMA_ID and study_id is None:
        raise DatasetReadError(
            f"{name}: a study reading must name the study it belongs to; pass study_id"
        )
    return study_artifact(name, schema_id, content, study_id=study_id or "")


def _predeclares_itself(content: str) -> bool:
    """A design names its study and states the integer version its own validator checks."""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return False
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("study_id"), str)
        and isinstance(payload.get("schema_version"), int)
    )
