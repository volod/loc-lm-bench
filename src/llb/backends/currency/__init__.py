"""Upstream currency for the model roster: is the generation we carry still the newest one offered?

The register in `llb.backends.roster` states which generation of each family the roster carries.
This package asks the registries those artifacts come from -- the Ollama library and the Hugging
Face model API -- what they currently offer for the same family, and reports the gap. It reports
only: adopting a generation invalidates every measurement taken against the one it replaces, so the
decision stays with an operator.
"""

from llb.backends.currency.registries import Cassette, Fetcher, Response, live_fetch
from llb.backends.currency.report import (
    BEHIND,
    CURRENT,
    UNKNOWN,
    VERDICTS,
    FamilyCurrency,
    RegistryReading,
    probe_family,
    probe_register,
)
from llb.backends.currency.render import counts, render_json, render_text, report_payload

__all__ = [
    "BEHIND",
    "CURRENT",
    "UNKNOWN",
    "VERDICTS",
    "Cassette",
    "FamilyCurrency",
    "Fetcher",
    "RegistryReading",
    "Response",
    "counts",
    "live_fetch",
    "probe_family",
    "probe_register",
    "render_json",
    "render_text",
    "report_payload",
]
