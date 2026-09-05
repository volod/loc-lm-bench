"""Named refusals raised before an artifact reaches downstream code."""


class ArtifactContractError(ValueError):
    """Base class for operator-facing artifact contract refusals."""


class MissingIdentityError(ArtifactContractError):
    pass


class UnknownContractError(ArtifactContractError):
    pass


class UnsupportedVersionError(ArtifactContractError):
    pass


class UnsupportedFutureVersionError(UnsupportedVersionError):
    pass


class InvalidSourceRecordError(ArtifactContractError):
    pass


class MigrationPathError(ArtifactContractError):
    pass


class AmbiguousMigrationError(MigrationPathError):
    pass


class DeclaredCompatibilityRefusal(ArtifactContractError):
    pass


class DatasetReadError(ArtifactContractError):
    pass
