"""Query-preparation resource contracts.

`query_glossary.json` is built from a draft bundle's dictionary candidates and then read by every
query-prep run that expands aliases. Its `version` tag says which glossary format produced it, and
the reader has always trusted it; registering the file gives that tag a dispatch key, so a
glossary written by a newer build refuses rather than expanding queries with entries this build
reads only half of.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from llb.core.contracts.artifacts import ArtifactContract

QUERY_GLOSSARY_SCHEMA_ID = "llb.query-glossary"


class GlossaryEntryRecord(BaseModel):
    """One canonical term and the surface forms a query may use instead."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    canonical: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)


class QueryGlossary(ArtifactContract):
    """`query_glossary.json`: the alias-expansion table a query-prep run applies."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_id: Literal["llb.query-glossary"]
    schema_version: Literal["1.0.0"]
    version: str = Field(min_length=1)
    entries: list[GlossaryEntryRecord] = Field(default_factory=list)
    source_bundle: str | None = None
