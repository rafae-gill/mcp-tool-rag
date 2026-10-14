"""mcp-tool-rag as an MCP server, end to end, against real upstream MCP servers.

Two upstream servers run for real on local ports: "billing" behind its own
mcp-gatekeeper, "deploy" without one. The router sits in front of both.
Keyword search only here, so the test needs no model download.
"""
import json
import socket
import threading
import time

import pytest

pytest.importorskip("mcp")
uvicorn = pytest.importorskip("uvicorn")
from starlette.testclient import TestClient  # noqa: E402

from mcp_gatekeeper import Gatekeeper, JsonFileStore, ListAudit, mint_key  # noqa: E402
from mcp_tool_rag import BM25  # noqa: E402
from mcp_tool_rag.server import ToolRag, build_app  # noqa: E402

try:
    from mcp.server.mcpserver import MCPServer as Server
except ImportError:
    from mcp.server.fastmcp import FastMCP as Server


def http_app(srv):
    try:
        return srv.streamable_http_app(stateless_http=True, json_response=True)
    except TypeError:
        srv.settings.stateless_http = True
        srv.settings.json_response = True
        return srv.streamable_http_app()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(app):
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(200):
        if server.started:
            return server, "http://127.0.0.1:%d/mcp" % port
        time.sleep(0.02)
    raise RuntimeError("upstream didn't start")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    store = JsonFileStore(str(tmp_path_factory.mktemp("keys") / "keys.json"))
    rag_tools = {"tools": ["search_tools", "call_tool"]}
    admin, _ = mint_key(store, name="Router", email="router@example.com", permissions={
        "billing": {"tools": ["*"]}, "deploy": {"tools": ["*"]}})
    finance, _ = mint_key(store, name="Fin", email="fin@example.com", permissions={
        "tool-rag": rag_tools, "billing": {"tools": ["refund_payment", "get_invoice"]}})
    support, _ = mint_key(store, name="Sup", email="sup@example.com", permissions={
        "tool-rag": rag_tools, "billing": {"tools": ["get_invoice"]},
        "deploy": {"tools": ["list_deploys"]}})

    billing = Server("billing")

    @billing.tool()
    def refund_payment(payment_id: str) -> str:
        """Refund a customer payment, in full."""
        return "refunded %s" % payment_id

    @billing.tool()
    def get_invoice(invoice_id: str) -> str:
        """Fetch one invoice with its line items."""
        return "invoice %s: 2 items" % invoice_id

    deploy = Server("deploy")

    @deploy.tool()
    def list_deploys(service: str) -> str:
        """Recent deploys of a service."""
        return "%s: v41, v42" % service

    @deploy.tool()
    def rollback_deploy(service: str) -> str:
        """Roll a service back to its previous release."""
        return "rolled back %s" % service

    billing_audit = ListAudit()
    s1, billing_url = serve(Gatekeeper(http_app(billing), server_name="billing", store=store,
                                       audit=billing_audit))
    s2, deploy_url = serve(http_app(deploy))

    cfg = {"name": "tool-rag", "limit": 3,
           "servers": [{"name": "billing", "url": billing_url, "headers": {"x-api-key": admin}},
                       {"name": "deploy", "url": deploy_url}],
           "permissions": {"mode": "gatekeeper", "store": "unused"}}
    rag = ToolRag(cfg, store=store, retriever_factory=lambda catalog, ex: BM25(catalog))
    cfg["permissions"]["store"] = str(store.path)
    app = build_app(cfg, rag=rag)
    # One client for the whole module: the SDK's session manager can only be
    # started once per app.
    with TestClient(app, base_url="http://127.0.0.1:8090") as client:
        yield {"client": client, "rag": rag, "finance": finance, "support": support,
               "billing_audit": billing_audit}
    s1.should_exit = s2.should_exit = True


def mcp_call(client, key, _tool_name, **arguments):
    h = {"content-type": "application/json", "accept": "application/json, text/event-stream",
         "mcp-protocol-version": "2025-06-18"}
    if key:
        h["x-api-key"] = key
    r = client.post("/mcp", headers=h, json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                             "params": {"name": _tool_name, "arguments": arguments}})
    return r


def test_indexes_every_upstream_tool(world):
    assert sorted(world["rag"].by_name) == ["billing.get_invoice", "billing.refund_payment",
                                            "deploy.list_deploys", "deploy.rollback_deploy"]


def test_search_respects_each_persons_permissions(world):
    c = world["client"]
    if True:
        fin = json.loads(mcp_call(c, world["finance"], "search_tools",
                                  query="refund this customer payment").json()["result"]["content"][0]["text"])
        sup = json.loads(mcp_call(c, world["support"], "search_tools",
                                  query="refund this customer payment").json()["result"]["content"][0]["text"])
    assert fin[0]["tool"] == "refund_payment" and fin[0]["server"] == "billing"
    assert "properties" in fin[0]["input_schema"]
    assert "refund_payment" not in [t["tool"] for t in sup]
    assert {t["tool"] for t in sup} <= {"get_invoice", "list_deploys"}


def test_call_runs_upstream_as_the_caller(world):
    """The caller's key is passed through, so the upstream gate audits the
    PERSON, not the router."""
    c = world["client"]
    if True:
        r = mcp_call(c, world["finance"], "call_tool", server="billing", tool="refund_payment",
                     arguments={"payment_id": "pay_7"})
    result = r.json()["result"]
    assert result["isError"] is False
    assert "refunded pay_7" in result["content"][0]["text"]
    last = world["billing_audit"].events[-1]
    assert (last.actor, last.action) == ("Fin", "refund_payment")


def test_call_of_a_forbidden_tool_is_refused_by_the_router(world):
    before = len(world["billing_audit"].events)
    c = world["client"]
    if True:
        r = mcp_call(c, world["support"], "call_tool", server="billing", tool="refund_payment",
                     arguments={"payment_id": "pay_7"})
    result = r.json()["result"]
    assert result["isError"] is True and "ACCESS DENIED" in result["content"][0]["text"]
    assert len(world["billing_audit"].events) == before      # never reached billing


def test_no_key_never_reaches_the_router(world):
    c = world["client"]
    if True:
        assert mcp_call(c, None, "search_tools", query="refund").status_code == 401


def test_unknown_tool_is_a_clear_error(world):
    c = world["client"]
    if True:
        r = mcp_call(c, world["finance"], "call_tool", server="billing", tool="nope")
    assert "Unknown tool" in r.json()["result"]["content"][0]["text"]
