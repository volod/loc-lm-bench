"""Small strict semantic-version parser used by the artifact registry."""

import re
from dataclasses import dataclass

_SEMVER = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")


@dataclass(frozen=True, order=True)
class SemanticVersion:
    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, value: str) -> "SemanticVersion":
        match = _SEMVER.fullmatch(value)
        if match is None:
            raise ValueError(f"invalid semantic version {value!r}; expected MAJOR.MINOR.PATCH")
        return cls(*(int(part) for part in match.groups()))

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"
