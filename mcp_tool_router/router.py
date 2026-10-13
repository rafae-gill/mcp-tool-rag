"""Return the tools relevant to a request that the caller is allowed to use.

The order of the two steps is the design:

    filter by permission  ->  rank by relevance  ->  top k

Ranking first and filtering afterwards looks equivalent and isn't. The ranker
then spends its top-k slots on tools the caller can't use, and the filter
leaves fewer than k (sometimes none). And a permission-unaware search shows the
agent tools it may not call at all: their names and descriptions leak, and the
agent plans around tools that will be refused.

`mode` exists so the evaluation can measure exactly that:
    "before"  the router: only allowed tools are ever ranked
    "after"   rank everything, then drop forbidden tools from the top k
    "none"    no permissions at all (what a generic tool search does)
"""
from __future__ import annotations

import heapq
from typing import Callable

from .catalog import Tool

Allowed = Callable[[Tool], bool]


class ToolRouter:
    def __init__(self, catalog: list[Tool], retriever):
        self.catalog = catalog
        self.retriever = retriever

    def search(self, query: str, allowed: Allowed | None = None, k: int = 5,
               mode: str = "before") -> list[Tool]:
        if mode not in ("before", "after", "none"):
            raise ValueError("mode must be 'before', 'after' or 'none'")
        if mode == "before":
            if allowed is None:
                # Fail closed: a router told to respect permissions but given
                # none returns nothing, never "everything".
                return []
            candidates = [t for t in self.catalog if _safe(allowed, t)]
        else:
            candidates = list(self.catalog)
        if not candidates:
            return []

        scores = self.retriever.scores(query, candidates)
        by_name = {t.name: t for t in candidates}
        # Only the top k are needed, so no full sort. Ties broken by name so
        # results are deterministic run to run.
        ranked = heapq.nsmallest(k, scores.items(), key=lambda kv: (-kv[1], kv[0]))
        top = [by_name[n] for n, _ in ranked]
        if mode == "after":
            top = [t for t in top if allowed is not None and _safe(allowed, t)]
        return top


def _safe(allowed: Allowed, tool: Tool) -> bool:
    try:
        return bool(allowed(tool))
    except Exception:
        return False                          # a broken check denies


def allowed_for(permissions: dict) -> Allowed:
    """Permission check from a key's permissions, in mcp-gatekeeper's shape:

        {"warehouse": {"tools": ["run_query", ...]}, "docs": {...}}

    Uses mcp-gatekeeper's own policy parsing, so the router and the gate can't
    disagree about what a key allows (`"access": "none"`, `"*"`, legacy forms).
    """
    from mcp_gatekeeper import ServerPolicy

    policies = {server: ServerPolicy.from_dict(block)
                for server, block in (permissions or {}).items()}

    def allowed(tool: Tool) -> bool:
        p = policies.get(tool.server)
        return p is not None and p.grants(tool.name)

    return allowed
