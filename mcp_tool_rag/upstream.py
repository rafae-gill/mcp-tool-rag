"""A minimal client for the MCP servers the router sits in front of.

Speaks streamable HTTP directly (initialize, then tools/list or tools/call),
so it works whatever SDK or language the upstream server is written in, and
whether it answers with plain JSON or an event stream. No dependencies.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

PROTOCOL = "2025-06-18"


class UpstreamError(RuntimeError):
    pass


class Upstream:
    """One upstream MCP server. Thread-safe; keeps one session per caller key."""

    def __init__(self, name: str, url: str, headers: dict | None = None, timeout: float = 60):
        self.name, self.url, self.timeout = name, url, timeout
        self.headers = dict(headers or {})
        self._sessions: dict[str, str | None] = {}
        self._lock = threading.Lock()
        self._next_id = 1

    # -- public --------------------------------------------------------------

    def list_tools(self, auth: dict | None = None) -> list[dict]:
        tools, cursor = [], None
        while True:
            result = self._request("tools/list", {"cursor": cursor} if cursor else {}, auth)
            tools += result.get("tools", [])
            cursor = result.get("nextCursor")
            if not cursor:
                return tools

    def call_tool(self, tool: str, arguments: dict, auth: dict | None = None) -> dict:
        return self._request("tools/call", {"name": tool, "arguments": arguments or {}}, auth)

    # -- plumbing ------------------------------------------------------------

    def _request(self, method: str, params: dict, auth: dict | None) -> dict:
        headers = {**self.headers, **(auth or {})}
        key = json.dumps(sorted(headers.items()))
        try:
            resp, _ = self._post({"jsonrpc": "2.0", "id": self._id(), "method": method,
                                  "params": params}, headers, self._session(key, headers))
        except UpstreamError:
            # The upstream may have restarted and forgotten our session:
            # start a fresh one and try once more.
            with self._lock:
                self._sessions.pop(key, None)
            resp, _ = self._post({"jsonrpc": "2.0", "id": self._id(), "method": method,
                                  "params": params}, headers, self._session(key, headers))
        if resp is None:
            raise UpstreamError("%s: no response to %s" % (self.name, method))
        if "error" in resp:
            raise UpstreamError("%s: %s" % (self.name, resp["error"].get("message", resp["error"])))
        return resp.get("result") or {}

    def _session(self, key: str, headers: dict) -> str | None:
        with self._lock:
            if key in self._sessions:
                return self._sessions[key]
        init = {"jsonrpc": "2.0", "id": self._id(), "method": "initialize",
                "params": {"protocolVersion": PROTOCOL, "capabilities": {},
                           "clientInfo": {"name": "mcp-tool-rag", "version": "0"}}}
        resp, sid = self._post(init, headers, None)
        if not resp or "error" in resp:
            raise UpstreamError("%s: initialize failed: %s" % (self.name, resp))
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"}, headers, sid)
        with self._lock:
            self._sessions[key] = sid
        return sid

    def _id(self) -> int:
        with self._lock:
            self._next_id += 1
            return self._next_id

    def _post(self, body: dict, headers: dict, session: str | None):
        h = {"content-type": "application/json",
             "accept": "application/json, text/event-stream",
             "mcp-protocol-version": PROTOCOL, **headers}
        if session:
            h["mcp-session-id"] = session
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=h,
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read().decode("utf-8")
                sid = r.headers.get("mcp-session-id") or session
                ctype = r.headers.get("content-type", "")
        except urllib.error.HTTPError as e:
            raise UpstreamError("%s: HTTP %d %s" % (self.name, e.code, e.read()[:200]))
        except urllib.error.URLError as e:
            raise UpstreamError("%s: unreachable (%s)" % (self.name, e.reason))
        if not raw.strip():
            return None, sid
        if "text/event-stream" in ctype:
            for line in raw.splitlines():
                if line.startswith("data:"):
                    msg = json.loads(line[5:].strip())
                    if msg.get("id") == body.get("id"):
                        return msg, sid
            return None, sid
        return json.loads(raw), sid
