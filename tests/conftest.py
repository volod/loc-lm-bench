"""Suite-wide no-GPU and no-download guards for the lightweight tier.

The rationale, escape hatches, and modes live in `llb.quality.gpu_guard.guard` and
`llb.quality.download_guard`. This module owns the pytest wiring and reporting decisions.
"""

import warnings
from collections.abc import Iterable, Iterator
from contextlib import nullcontext

import pytest

from llb.quality import download_guard
from llb.quality.gpu_guard import guard as gpu_guard
from llb.quality.gpu_guard import spawn as gpu_guard_spawn


def device_guard_steps(test_id: str, marker_names: Iterable[str]) -> Iterator[None]:
    """Deny the device to children, watch this process, and act on what the test did.

    Kept a plain generator so the reporting decision itself is drivable from a test rather than
    only from the suite it wraps.
    """
    guard = gpu_guard.DeviceGuard.start(marker_names)
    if guard is None:
        yield
        return
    with gpu_guard_spawn.denied_children() if guard.denies_children else nullcontext():
        yield
    outcome = guard.verdict(test_id)
    if outcome is None:
        return
    mode, message = outcome
    if mode == gpu_guard.MODE_REFUSE:
        pytest.fail(message, pytrace=False)
    warnings.warn(message, stacklevel=1)


def download_guard_steps(test_id: str, marker_names: Iterable[str]) -> Iterator[None]:
    """Refuse non-loopback connections for an undeclared lightweight-tier test."""
    guard = download_guard.DownloadGuard.start(test_id, marker_names)
    if guard is None:
        yield
        return
    with guard.connections():
        yield


@pytest.fixture(autouse=True)
def light_tier_device_guard(request: pytest.FixtureRequest) -> Iterator[None]:
    """Fail an unmarked test that reaches the device, and keep its children off the device.

    Autouse and declared in the root test conftest, so it is set up first and torn down LAST: the
    after-snapshot sees what a test's own fixtures did on the way out, not only its body.
    """
    yield from device_guard_steps(
        request.node.nodeid, [marker.name for marker in request.node.iter_markers()]
    )


@pytest.fixture(autouse=True)
def light_tier_download_guard(request: pytest.FixtureRequest) -> Iterator[None]:
    """Fail an unmarked test before it can connect to a non-loopback destination."""
    yield from download_guard_steps(
        request.node.nodeid, [marker.name for marker in request.node.iter_markers()]
    )
