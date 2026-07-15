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
"""

import json
import logging
from pathlib import Path
from typing import Any

from llb.core.contracts.benchmarks import ToolDef

_LOG = logging.getLogger(__name__)


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


def build_mcp_server(catalog: dict[str, ToolDef], *, name: str = "loc-lm-bench-tools") -> Any:
    """Build a low-level MCP `Server` exposing the catalog (lazy `mcp` import; opt-in extra)."""
    try:
        from mcp.server import Server
        from mcp.types import TextContent, Tool
    except ImportError as exc:  # pragma: no cover - exercised only with the optional dep
        raise SystemExit(
            'ERROR: the MCP transport needs the [mcp] extra. Run: uv pip install -e ".[mcp]"'
        ) from exc

    specs = mcp_tool_specs(catalog)
    server: Any = Server(name)

    @server.list_tools()
    async def _list_tools() -> list[Any]:
        return [
            Tool(
                name=spec["name"],
                description=spec["description"],
                inputSchema=spec["inputSchema"],
            )
            for spec in specs
        ]

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
