"""Router behaviour that must hold whatever the retriever's quality."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp_tool_rag import (BM25, Hybrid, Tool, ToolRouter, allowed_for,  # noqa: E402
                          split_identifier, tokenize)

CATALOG = [
    Tool("billing", "refund_payment", "Refund a customer payment in full or in part.",
         {"payment_id": "Payment to refund", "amount": "Amount to refund"}),
    Tool("billing", "get_invoice", "Fetch one invoice with its line items.", {"invoice_id": "Invoice"}),
    Tool("warehouse", "run_query", "Run a read-only SQL query against the warehouse.", {"sql": "SQL"}),
    Tool("warehouse", "describe_table", "Columns and types of a table or view.", {"table": "Name"}),
    Tool("slack", "search_messages", "Search Slack messages across channels.", {"query": "Text"}),
    Tool("deploy", "rollbackDeploy", "Roll a service back to its previous release.", {"service": "Name"}),
]
SUPPORT = {"billing": {"tools": ["get_invoice"]}, "slack": {"tools": ["search_messages"]}}
FINANCE = {"billing": {"tools": ["get_invoice", "refund_payment"]}}


def router():
    return ToolRouter(CATALOG, BM25(CATALOG))


def names(tools):
    return [t.name for t in tools]


def test_identifier_split_helps_keyword_search():
    assert split_identifier("rollbackDeploy") == "rollback deploy"
    assert split_identifier("get_invoice-pdf") == "get invoice pdf"
    assert "invoice" in tokenize(CATALOG[1].search_text)


def test_relevant_allowed_tool_ranks_first():
    top = router().search("give this customer their money back", allowed_for(FINANCE), k=3)
    assert names(top)[0] == "refund_payment"


def test_forbidden_tools_are_never_returned():
    top = router().search("refund the customer's payment", allowed_for(SUPPORT), k=5)
    assert "refund_payment" not in names(top)
    assert set(names(top)) <= {"get_invoice", "search_messages"}


def test_no_permissions_means_no_tools():
    """Fail closed: told to respect permissions but given none."""
    assert router().search("refund", None, k=5, mode="before") == []
    assert router().search("refund", allowed_for({}), k=5) == []


def test_a_broken_permission_check_denies():
    def boom(tool):
        raise RuntimeError("policy store down")
    assert router().search("refund", boom, k=5) == []


def test_filtering_after_ranking_returns_fewer_than_k():
    """The failure mode the router exists to avoid: forbidden tools use up
    the top-k slots, then get dropped."""
    r = router()
    after = r.search("refund payment amount", allowed_for(SUPPORT), k=2, mode="after")
    before = r.search("refund payment amount", allowed_for(SUPPORT), k=2, mode="before")
    assert len(after) < 2
    assert len(before) == 2


def test_unaware_search_shows_forbidden_tools():
    top = router().search("refund payment", None, k=3, mode="none")
    assert "refund_payment" in names(top)


def test_bm25_statistics_do_not_depend_on_the_caller():
    """Scores come from whole-catalog statistics, so a tool scores the same
    whichever other tools the caller is allowed."""
    b = BM25(CATALOG)
    full = b.scores("invoice", CATALOG)["get_invoice"]
    narrow = b.scores("invoice", [CATALOG[1]])["get_invoice"]
    assert full == narrow


def test_hybrid_fuses_by_rank():
    class Fixed:
        def __init__(self, order):
            self.order = order

        def scores(self, query, candidates):
            return {t.name: -self.order.index(t.name) for t in candidates}

    order_a = [t.name for t in CATALOG]
    order_b = list(reversed(order_a))
    h = Hybrid(Fixed(order_a), Fixed(order_b))
    s = h.scores("x", CATALOG)
    # Opposite rankings fuse to a near-tie: no single method dominates.
    assert max(s.values()) - min(s.values()) < 0.002


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        router().search("x", allowed_for(FINANCE), mode="sideways")


def test_results_are_deterministic():
    r = router()
    a = names(r.search("zzz nothing matches", allowed_for(FINANCE), k=2))
    b = names(r.search("zzz nothing matches", allowed_for(FINANCE), k=2))
    assert a == b
