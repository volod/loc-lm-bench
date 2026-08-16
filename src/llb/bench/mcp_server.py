"""tooling benchmark MCP transport -- serve the SAME tool catalog via the official `mcp` Python SDK.

The tooling benchmark already drives a candidate two ways from one source (the `ToolDef` catalog):
the universal text-in-prompt protocol and the native OpenAI `tools=` caller (`bench.tooling`). This
module adds the third transport from the same source -- a Model Context Protocol server that exposes
each catalog tool over MCP, so an MCP-speaking client (or model harness) discovers the identical
tools. The catalog is the single source of truth across all three transports.

`mcp_tool_specs` is the PURE mapping `ToolDef -> MCP tool descriptor` (name / description /
inputSchema), unit-tested without the dependency. `build_mcp_server` lazily imports the `mcp` SDK
(an opt-in extra, kept out of the base install) and builds a low-level server whose `list_tools`
returns the catalog and whose `call_tool` echoes the call (the catalog is CALL-ONLY here -- tool
EXECUTION is the agentic sandbox, not this transport).

SDK major: this transport targets `mcp` 1.x, the version `uv.lock` pins (the `[mcp]` extra carries
the matching `<2` bound). mcp 2.x is not a rename to absorb -- it REPLACES the low-level server
API this module is built on: `Server.list_tools()` / `Server.call_tool()` no longer exist, tool
registration moved to `add_request_handler` / the new `MCPServer`, and `Tool` renamed `inputSchema`
to `input_schema` (keeping the old spelling only as a validation alias). Carrying two server
implementations for an opt-in transport nothing else in the benchmark line depends on is not worth
its maintenance, so `build_mcp_server` REFUSES an unsupported major with a named error instead of
failing with an `AttributeError` inside a decorator. The one place the rename does reach --
building a `Tool` from a descriptor -- goes through `model_validate`, so the descriptor dict stays
the single source under either spelling.
"""

import importlib.metadata as metadata
import json
import logging
from pathlib import Path
from typing import Any

from llb.core.contracts.benchmarks import ToolDef

_LOG = logging.getLogger(__name__)

SUPPORTED_MCP_MAJOR = 1
MCP_EXTRA_HINT = 'uv pip install -e ".[mcp]"'


def mcp_tool_specs(catalog: dict[str, ToolDef]) -> list[dict[str, Any]]:
    """Map the tool catalog onto MCP tool descriptors (name / description / inputSchema)."""
    return [
        {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "inputSchema": tool.get("parameters", {}) or {"type": "object", "properties": {}},
        }
        for tool in catalog.values()
    ]


def load_catalog(path: Path | str) -> dict[str, ToolDef]:
    """Load a tooling bundle's tool catalog (reusing the bench-tooling loader)."""
    from llb.bench.tooling import load_catalog_file

    catalog, _cases = load_catalog_file(path)
    return catalog


def installed_sdk_major(version: str) -> int:
    """Major of an installed `mcp` version string (`"1.28.1"` -> 1)."""
    return int(version.split(".", 1)[0])


def require_supported_sdk(version: str) -> None:
    """Refuse an `mcp` major this transport was not built against, with the fix in the message."""
    major = installed_sdk_major(version)
    if major != SUPPORTED_MCP_MAJOR:
        raise SystemExit(
            f"ERROR: the MCP transport targets mcp {SUPPORTED_MCP_MAJOR}.x and found {version}. "
            f"mcp {major}.x does not carry the low-level Server tool decorators this transport is "
            f"built on. Install the locked SDK: {MCP_EXTRA_HINT}"
        )


def build_mcp_server(catalog: dict[str, ToolDef], *, name: str = "loc-lm-bench-tools") -> Any:
    """Build a low-level MCP `Server` exposing the catalog (lazy `mcp` import; opt-in extra)."""
    try:
        from mcp.server import Server
        from mcp.types import TextContent, Tool
    except ImportError as exc:  # pragma: no cover - exercised only with the optional dep
        raise SystemExit(
            f"ERROR: the MCP transport needs the [mcp] extra. Run: {MCP_EXTRA_HINT}"
        ) from exc
    require_supported_sdk(metadata.version("mcp"))

    specs = mcp_tool_specs(catalog)
    server: Any = Server(name)

    @server.list_tools()
    async def _list_tools() -> list[Any]:
        # `model_validate` keeps the descriptor dict authoritative: `inputSchema` is the field name
        # under 1.x and the validation alias under 2.x, so neither spelling is hard-coded here.
        return [Tool.model_validate(spec) for spec in specs]

    @server.call_tool()
    async def _call_tool(tool_name: str, arguments: dict[str, Any]) -> list[Any]:
        # Call-only catalog: echo the recognized call (real execution is the agentic benchmark sandbox).
        if tool_name not in catalog:
            raise ValueError(f"unknown tool: {tool_name}")
        echo = {"tool": tool_name, "arguments": arguments}
        return [TextContent(type="text", text=json.dumps(echo, ensure_ascii=False))]

    _LOG.info("[mcp] built MCP server %r serving %d tools", name, len(specs))
    return server


def serve_stdio(catalog: dict[str, ToolDef], *, name: str = "loc-lm-bench-tools") -> None:
    """Run the catalog MCP server over stdio (blocks). Lazy deps; for an MCP client to connect."""
    import anyio
    from mcp.server.stdio import stdio_server

    server = build_mcp_server(catalog, name=name)

    async def _run() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(_run)
