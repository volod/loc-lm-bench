"""Suite-wide fixtures. The only one here is the no-GPU guard for the lightweight tier.

The rationale, the escape-hatch markers, and the `LLB_GPU_GUARD` modes live in
`llb.quality.gpu_guard`; this module is the pytest wiring and the reporting decision.
"""

import warnings
from collections.abc import Iterable, Iterator

import pytest

from llb.quality import gpu_guard


def device_guard_steps(test_id: str, marker_names: Iterable[str]) -> Iterator[None]:
    """Snapshot the device before a test, and act on what it did after -- the fixture's whole body.

    Kept a plain generator so the reporting decision itself is drivable from a test rather than
    only from the suite it wraps.
    """
    guard = gpu_guard.DeviceGuard.start(marker_names)
    yield
    if guard is None:
        return
    outcome = guard.verdict(test_id)
    if outcome is None:
        return
    mode, message = outcome
    if mode == gpu_guard.MODE_REFUSE:
        pytest.fail(message, pytrace=False)
    warnings.warn(message, stacklevel=1)


@pytest.fixture(autouse=True)
def light_tier_device_guard(request: pytest.FixtureRequest) -> Iterator[None]:
    """Fail an unmarked test that initialized a CUDA context or imported `flashinfer`.

    Autouse and declared in the root test conftest, so it is set up first and torn down LAST: the
    after-snapshot sees what a test's own fixtures did on the way out, not only its body.
    """
    yield from device_guard_steps(
        request.node.nodeid, [marker.name for marker in request.node.iter_markers()]
    )
