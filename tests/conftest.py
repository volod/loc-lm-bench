"""Suite-wide fixtures. The only one here is the no-GPU guard for the lightweight tier.

The rationale, the escape-hatch markers, the child-process denial, and the `LLB_GPU_GUARD` modes
live in `llb.quality.gpu_guard`; this module is the pytest wiring and the reporting decision.
"""

import subprocess
import warnings
from collections.abc import Iterable, Iterator
from contextlib import contextmanager, nullcontext

import pytest

from llb.quality import gpu_guard


@contextmanager
def deny_the_device_to_children() -> Iterator[None]:
    """Hand every `subprocess` child an empty `CUDA_VISIBLE_DEVICES` for the duration.

    Patching the module attribute rather than this process's environment is the whole point: the
    parent keeps its device (see `gpu_guard`, where the caching that forces that is written down).
    """
    original = subprocess.Popen
    subprocess.Popen = gpu_guard.denying_popen(original)  # type: ignore[misc]
    try:
        yield
    finally:
        subprocess.Popen = original  # type: ignore[misc]


def device_guard_steps(test_id: str, marker_names: Iterable[str]) -> Iterator[None]:
    """Deny the device to children, watch this process, and act on what the test did.

    Kept a plain generator so the reporting decision itself is drivable from a test rather than
    only from the suite it wraps.
    """
    guard = gpu_guard.DeviceGuard.start(marker_names)
    if guard is None:
        yield
        return
    with deny_the_device_to_children() if guard.denies_children else nullcontext():
        yield
    outcome = guard.verdict(test_id)
    if outcome is None:
        return
    mode, message = outcome
    if mode == gpu_guard.MODE_REFUSE:
        pytest.fail(message, pytrace=False)
    warnings.warn(message, stacklevel=1)


@pytest.fixture(autouse=True)
def light_tier_device_guard(request: pytest.FixtureRequest) -> Iterator[None]:
    """Fail an unmarked test that reaches the device, and keep its children off the device.

    Autouse and declared in the root test conftest, so it is set up first and torn down LAST: the
    after-snapshot sees what a test's own fixtures did on the way out, not only its body.
    """
    yield from device_guard_steps(
        request.node.nodeid, [marker.name for marker in request.node.iter_markers()]
    )
