"""Keep the lightweight pytest tier offline without breaking local fake servers.

The non-slow suite is the no-GPU, no-download tier. Download clients are too numerous to patch
one at a time, and client-specific offline variables do not cover an ordinary HTTP request. This
guard therefore watches the common effect: ``socket.socket.connect`` / ``connect_ex`` to a
non-loopback destination. It refuses the connection before the operating system sees it.

Loopback IPv4/IPv6, ``localhost``, and Unix-domain sockets remain available for fake-server tests.
Mark a test ``slow`` when real network work belongs to its expensive integration flow, or
``network_env`` when a quick test must use a non-local service. ``LLB_DOWNLOAD_GUARD=report`` lets
the connection through with a warning, while ``off`` disables the guard. Refusal is the default:
an offline-tier violation should fail on the commit that introduces it, without waiting for a cold
GitHub runner to expose it.

The socket methods are patched around each test rather than replacing ``socket.socket``. Clients
that captured the class before pytest setup therefore still pass through the guarded methods.
"""

import ipaddress
import os
import socket
import warnings
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from llb.core import env

EXEMPT_MARKERS = ("slow", "network_env")

MODE_REFUSE = "refuse"
MODE_REPORT = "report"
MODE_OFF = "off"
GUARD_MODES = (MODE_REFUSE, MODE_REPORT, MODE_OFF)
DEFAULT_MODE = MODE_REFUSE

_NETWORK_FAMILIES = (socket.AF_INET, socket.AF_INET6)
_CONNECT_METHODS = ("connect", "connect_ex")
_REMEDY = (
    "the non-slow suite is the no-download tier: inject a fake at the client seam, use a "
    "loopback fake server, or mark the test slow / network_env to declare network access "
    f"({env.DOWNLOAD_GUARD}=report downgrades this to a warning)"
)


class DownloadGuardError(RuntimeError):
    """A lightweight-tier test attempted a non-loopback connection."""


def guard_mode(environ: Mapping[str, str] | None = None) -> str:
    """Resolve the guard mode; refuse an unknown value instead of silently disabling the gate."""
    raw = (environ if environ is not None else os.environ).get(env.DOWNLOAD_GUARD)
    if raw is None or not raw.strip():
        return DEFAULT_MODE
    mode = raw.strip().lower()
    if mode not in GUARD_MODES:
        raise ValueError(f"{env.DOWNLOAD_GUARD}={raw!r} is not one of {', '.join(GUARD_MODES)}")
    return mode


def exempting_marker(marker_names: Iterable[str]) -> str | None:
    """Return the marker declaring network access, if the test carries one."""
    names = set(marker_names)
    return next((marker for marker in EXEMPT_MARKERS if marker in names), None)


def is_loopback_destination(family: int, address: Any) -> bool:
    """Whether a socket destination is local enough for an offline fake-server test."""
    if family not in _NETWORK_FAMILIES:
        return True
    host = _address_host(address)
    if host is None:
        return False
    normalized = host.rstrip(".").lower()
    if normalized == "localhost":
        return True
    address_text = normalized.split("%", maxsplit=1)[0]
    try:
        ip = ipaddress.ip_address(address_text)
    except ValueError:
        return False
    if ip.is_loopback:
        return True
    return bool(ip.version == 6 and ip.ipv4_mapped and ip.ipv4_mapped.is_loopback)


def destination_label(address: Any) -> str:
    """A stable, useful destination label for the finding."""
    if isinstance(address, tuple) and len(address) >= 2:
        return f"{address[0]}:{address[1]}"
    return repr(address)


@dataclass(frozen=True)
class DownloadGuard:
    """The mode and test attribution for one test's connection guard."""

    mode: str
    test_id: str

    @classmethod
    def start(
        cls,
        test_id: str,
        marker_names: Iterable[str],
        *,
        environ: Mapping[str, str] | None = None,
    ) -> "DownloadGuard | None":
        """Create a guard unless the test is exempt or the operator disabled it."""
        mode = guard_mode(environ)
        if mode == MODE_OFF or exempting_marker(marker_names) is not None:
            return None
        return cls(mode=mode, test_id=test_id)

    @contextmanager
    def connections(self) -> Iterator[None]:
        """Guard both socket connection methods for the duration of this test."""
        originals = {name: getattr(socket.socket, name) for name in _CONNECT_METHODS}
        replacements = {name: self._guarded_connector(originals[name]) for name in _CONNECT_METHODS}
        try:
            for name, replacement in replacements.items():
                setattr(socket.socket, name, replacement)
            yield
        finally:
            for name, original in originals.items():
                setattr(socket.socket, name, original)

    def _guarded_connector(self, connector: Any) -> Any:
        def guarded(sock: socket.socket, address: Any) -> Any:
            if is_loopback_destination(sock.family, address):
                return connector(sock, address)
            message = guard_message(self.test_id, address)
            if self.mode == MODE_REFUSE:
                raise DownloadGuardError(message)
            warnings.warn(message, stacklevel=2)
            return connector(sock, address)

        return guarded


def guard_message(test_id: str, address: Any) -> str:
    """Explain the denied effect and every supported escape hatch."""
    return (
        f"[download-guard] {test_id} attempted a non-loopback connection to "
        f"{destination_label(address)}. {_REMEDY}"
    )


def _address_host(address: Any) -> str | None:
    if not isinstance(address, tuple) or not address:
        return None
    host = address[0]
    if isinstance(host, bytes):
        try:
            return host.decode("ascii")
        except UnicodeDecodeError:
            return None
    return host if isinstance(host, str) else None
