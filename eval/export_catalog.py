"""Pull a live MCP server's tool list into a catalog file. No dependencies.

    python eval/export_catalog.py --url https://your-server/mcp --key sk_... \
        --server warehouse --out data/real/catalog.json

Run it once per server with the same --out: each run adds that server's tools
and replaces any it exported earlier. Then review the file before sharing it —
tool names and descriptions can say a lot about what a company does.

Speaks streamable HTTP directly: initialize, the initialized notification,
then tools/list (following pagination). Works whether the server answers with
plain JSON or with a server-sent event stream.
"""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

PROTOCOL = "2025-06-18"


def post(url, key, body, session=None, bearer=False):
    headers = {"content-type": "application/json",
               "accept": "application/json, text/event-stream",
               "mcp-protocol-version": PROTOCOL}
    if key:
        headers["authorization" if bearer else "x-api-key"] = ("Bearer " + key) if bearer else key
    if session:
        headers["mcp-session-id"] = session
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode("utf-8")
            sid = r.headers.get("mcp-session-id") or session
            ctype = r.headers.get("content-type", "")
    except urllib.error.HTTPError as e:
        raise SystemExit("HTTP %d from server: %s" % (e.code, e.read()[:300]))
    if not raw.strip():
        return None, sid
    if "text/event-stream" in ctype:
        # Take the JSON-RPC response for our id out of the event stream.
        for line in raw.splitlines():
            if line.startswith("data:"):
                msg = json.loads(line[5:].strip())
                if msg.get("id") == body.get("id"):
                    return msg, sid
        return None, sid
    return json.loads(raw), sid


def fetch_tools(url, key, bearer=False):
    init = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": PROTOCOL, "capabilities": {},
                       "clientInfo": {"name": "mcp-tool-rag-export", "version": "0"}}}
    resp, sid = post(url, key, init, bearer=bearer)
    if not resp or "error" in resp:
        raise SystemExit("initialize failed: %s" % resp)
    post(url, key, {"jsonrpc": "2.0", "method": "notifications/initialized"}, sid, bearer)

    tools, cursor, n = [], None, 2
    while True:
        params = {"cursor": cursor} if cursor else {}
        resp, sid = post(url, key, {"jsonrpc": "2.0", "id": n, "method": "tools/list",
                                    "params": params}, sid, bearer)
        if not resp or "error" in resp:
            raise SystemExit("tools/list failed: %s" % resp)
        result = resp["result"]
        tools += result.get("tools", [])
        cursor = result.get("nextCursor")
        n += 1
        if not cursor:
            return tools


def to_catalog(server, tools):
    out = []
    for t in tools:
        props = (t.get("inputSchema") or {}).get("properties") or {}
        out.append({"server": server, "name": t["name"],
                    "description": (t.get("description") or "").strip(),
                    "params": {p: (spec or {}).get("description", "") for p, spec in props.items()}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="the server's MCP endpoint, e.g. https://host/mcp")
    ap.add_argument("--key", default="", help="API key, sent as x-api-key (or Bearer with --bearer)")
    ap.add_argument("--bearer", action="store_true")
    ap.add_argument("--server", required=True, help="short server name to file the tools under")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tools = to_catalog(a.server, fetch_tools(a.url, a.key, a.bearer))
    out = Path(a.out)
    existing = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    merged = [t for t in existing if t["server"] != a.server] + tools
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print("%s: %d tools -> %s (%d in total)" % (a.server, len(tools), out, len(merged)))


if __name__ == "__main__":
    main()
