"""mcp-tool-rag as an MCP server.

Point it at the MCP servers you already run. An agent connects to this one
server instead and gets two tools:

    search_tools(query)                      the few relevant tools this caller may use
    call_tool(server, tool, arguments)       run one of them on its real server

    python -m mcp_tool_rag serve config.json

config.json:

    {
      "name": "tool-rag",
      "servers": [
        {"name": "warehouse", "url": "http://localhost:8001/mcp",
         "headers": {"x-api-key": "env:WAREHOUSE_KEY"}},
        {"name": "billing",   "url": "http://localhost:8002/mcp"}
      ],
      "permissions": {"mode": "none"},
      "limit": 8,
      "examples": "examples.json"
    }

Secrets never go in the file: a header value "env:NAME" is read from the
environment variable NAME.

Permissions ("permissions.mode"):
    "none"         every caller may find and call every tool
    "gatekeeper"   per-person, from mcp-gatekeeper. Callers authenticate with
                   their mcp-gatekeeper key; search_tools only returns tools
                   that key allows, call_tool checks again, and the caller's
                   key is passed through to the upstream server so its own
                   gate checks and audits the PERSON, not this router.
                   Needs "store": a keys.json path or a postgresql:// DSN.
                   Each key also needs this server ("name") in its
                   permissions, with tools ["search_tools", "call_tool"].
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from .catalog import Tool
from .retrievers import BM25, Hybrid
from .router import ToolRouter, allowed_for
from .upstream import Upstream, UpstreamError

log = logging.getLogger("mcp_tool_rag")

try:                                                    # mcp >= 2
    from mcp.server.mcpserver import Context, MCPServer as _Server
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:                                     # mcp 1.x
    from mcp.server.fastmcp import Context, FastMCP as _Server
    from mcp.server.fastmcp.exceptions import ToolError


def _resolve(value):
    if isinstance(value, str) and value.startswith("env:"):
        v = os.environ.get(value[4:])
        if v is None:
            raise SystemExit("environment variable %s is not set" % value[4:])
        return v
    return value


def load_config(path: str | Path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    for s in cfg.get("servers", []):
        s["headers"] = {k: _resolve(v) for k, v in (s.get("headers") or {}).items()}
    perms = cfg.setdefault("permissions", {"mode": "none"})
    if perms.get("mode") not in ("none", "gatekeeper"):
        raise SystemExit('permissions.mode must be "none" or "gatekeeper"')
    if perms["mode"] == "gatekeeper" and not perms.get("store"):
        raise SystemExit('permissions.mode "gatekeeper" needs a "store"')
    if "store" in perms:
        perms["store"] = _resolve(perms["store"])
    return cfg


def _make_retriever(catalog: list[Tool], examples: dict | None):
    """Best method from the evaluation that the installed extras allow:
    keyword + embeddings (+ example requests), else keyword only."""
    if examples:
        kw = BM25([Tool(t.server, t.name, t.description + ". Example requests: "
                        + " | ".join(examples.get(t.name, examples.get(t.tool_name, []))),
                        t.params, original=t.original) for t in catalog])
    else:
        kw = BM25(catalog)
    try:
        from .retrievers import Embeddings, EnrichedEmbeddings
        emb = Embeddings(catalog)
    except ImportError:
        log.warning("fastembed not installed: keyword search only "
                    '(pip install "mcp-tool-rag[embeddings]")')
        return kw
    if examples:
        keyed = {t.name: examples.get(t.name, examples.get(t.tool_name, [])) for t in catalog}
        emb = EnrichedEmbeddings(emb, catalog, keyed)
    return Hybrid(kw, emb)


class ToolRag:
    """The router over live upstream servers. Separate from the MCP wiring so
    it can be used and tested directly."""

    def __init__(self, cfg: dict, store=None, retriever_factory=_make_retriever):
        self.cfg = cfg
        self.limit = int(cfg.get("limit", 8))
        self.upstreams = {s["name"]: Upstream(s["name"], s["url"], s.get("headers"))
                          for s in cfg.get("servers", [])}
        self.store = store
        self.schemas: dict[str, dict] = {}
        catalog = []
        for name, up in self.upstreams.items():
            try:
                tools = up.list_tools()
            except UpstreamError as e:
                log.error("skipping %s: %s", name, e)
                continue
            for t in tools:
                qualified = "%s.%s" % (name, t["name"])
                props = (t.get("inputSchema") or {}).get("properties") or {}
                catalog.append(Tool(server=name, name=qualified, original=t["name"],
                                    description=(t.get("description") or "").strip(),
                                    params={p: (v or {}).get("description", "") or ""
                                            for p, v in props.items()}))
                self.schemas[qualified] = t.get("inputSchema") or {"type": "object"}
        if not catalog:
            raise SystemExit("no tools found on any upstream server")
        examples = None
        if cfg.get("examples"):
            examples = json.loads(Path(cfg["examples"]).read_text(encoding="utf-8"))
        self.catalog = catalog
        self.by_name = {t.name: t for t in catalog}
        self.router = ToolRouter(catalog, retriever_factory(catalog, examples))
        log.info("indexed %d tools from %d servers", len(catalog), len(self.upstreams))

    # -- permissions ---------------------------------------------------------

    def allowed(self, api_key: str | None):
        """The caller's permission check, or None when permissions are off.
        Fails closed: no key, unknown key, or a store error allows nothing."""
        if self.cfg["permissions"]["mode"] == "none":
            return lambda tool: True
        from mcp_gatekeeper.keys import hash_secret
        if not api_key:
            return lambda tool: False
        try:
            rec = self.store.find_key_by_hash(hash_secret(api_key))
        except Exception as e:
            log.error("permission lookup failed: %s", e)
            return lambda tool: False
        if rec is None or not rec.active:
            return lambda tool: False
        return allowed_for(rec.permissions)

    # -- the two tools -------------------------------------------------------

    def search(self, query: str, api_key: str | None, limit: int | None = None) -> list[dict]:
        k = max(1, min(int(limit or self.limit), 25))
        tools = self.router.search(query, self.allowed(api_key), k=k)
        return [{"server": t.server, "tool": t.tool_name, "description": t.description,
                 "input_schema": self.schemas.get(t.name, {})} for t in tools]

    def call(self, server: str, tool: str, arguments: dict | None, api_key: str | None,
             forward: dict | None) -> dict:
        t = self.by_name.get("%s.%s" % (server, tool))
        if t is None:
            raise ToolError("Unknown tool %s on server %s. Use search_tools to find tools."
                            % (tool, server))
        if not self.allowed(api_key)(t):
            raise ToolError("ACCESS DENIED: your key may not use %s on %s." % (tool, server))
        try:
            return self.upstreams[server].call_tool(tool, arguments or {}, auth=forward)
        except UpstreamError as e:
            raise ToolError("Upstream error: %s" % e)


def _caller(ctx):
    """The caller's key and the headers to pass upstream, from the request."""
    from mcp_gatekeeper.middleware import extract_key
    request = getattr(getattr(ctx, "request_context", None), "request", None)
    raw = getattr(request, "headers", None)
    if raw is None:
        return None, None
    headers = [(k.encode().lower(), v.encode()) for k, v in raw.items()]
    key = extract_key(headers) or None
    forward = {}
    for h in ("x-api-key", "authorization"):
        if raw.get(h):
            forward[h] = raw.get(h)
    return key, forward


def build_app(cfg: dict, rag: ToolRag | None = None):
    """The ASGI app: the MCP server, behind mcp-gatekeeper when permissions
    are on."""
    store = None
    if cfg["permissions"]["mode"] == "gatekeeper":
        s = cfg["permissions"]["store"]
        if s.startswith("postgres"):
            from mcp_gatekeeper.postgres import PostgresStore
            store = PostgresStore(s)
        else:
            from mcp_gatekeeper import JsonFileStore
            store = JsonFileStore(s)
    rag = rag or ToolRag(cfg, store=store)
    # Pass the caller's own key to upstream servers (so their gates check and
    # audit the person) whenever permissions are per-person.
    forward_auth = cfg["permissions"].get("forward_auth", store is not None)

    server = _Server(cfg.get("name", "tool-rag"))

    @server.tool()
    def search_tools(query: str, ctx: Context, limit: int = 8) -> str:
        """Find the tools for a task. Describe what you want to do in plain words;
        returns the few most relevant tools you are allowed to use, with the
        server, tool name and input schema to pass to call_tool."""
        key, _ = _caller(ctx)
        return json.dumps(rag.search(query, key, limit), indent=1)

    @server.tool()
    def call_tool(server: str, tool: str, ctx: Context, arguments: dict | None = None) -> str:
        """Run a tool found with search_tools, on its own server."""
        key, forward = _caller(ctx)
        result = rag.call(server, tool, arguments, key, forward if forward_auth else None)
        text = "\n".join(c.get("text", json.dumps(c)) for c in result.get("content", []))
        if result.get("isError"):
            raise ToolError(text or "the tool reported an error")
        return text

    try:
        app = server.streamable_http_app(stateless_http=True, json_response=True)
    except TypeError:                                   # mcp 1.x
        server.settings.stateless_http = True
        server.settings.json_response = True
        app = server.streamable_http_app()

    if store is not None:
        from mcp_gatekeeper import Gatekeeper
        app = Gatekeeper(app, server_name=cfg.get("name", "tool-rag"), store=store)
    return app
