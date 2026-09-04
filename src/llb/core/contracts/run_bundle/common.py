"""Shapes shared by more than one run-bundle contract."""

from pydantic import BaseModel, ConfigDict


class RunBundleRow(BaseModel):
    """Strict nested row shared by the run, study, board, and orchestration contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
