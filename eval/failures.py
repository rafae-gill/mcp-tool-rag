"""Which requests does a retriever get wrong, and how?

    python eval/failures.py --retriever embeddings
    python eval/failures.py --retriever emb+ex --show 20

A miss is a request (best tool allowed) whose correct tool isn't in the top
k. Each miss is classed by what came first instead:
    wrong server     the top result is on a different server (e.g. Slack
                     search instead of docs search): a cross-service confusion
    right server     the right server, the wrong tool (e.g. list vs get)
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from mcp_tool_router import ToolRouter, allowed_for  # noqa: E402
from run_eval import build, load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", default="embeddings")
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--examples", default=str(ROOT / "data" / "enrichment" / "examples.json"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--show", type=int, default=12)
    a = ap.parse_args()

    import json
    catalog, roles, queries = load(Path(a.data))
    ex = Path(a.examples)
    examples = json.loads(ex.read_text(encoding="utf-8")) if ex.exists() else None
    router = ToolRouter(catalog, build([a.retriever], catalog, examples)[a.retriever])
    server = {t.name: t.server for t in catalog}

    misses, kinds, by_diff, by_srv = [], Counter(), Counter(), Counter()
    total = 0
    for q in queries:
        if q.get("forbidden_best"):
            continue
        total += 1
        top = [t.name for t in router.search(q["query"], allowed_for(roles[q["role"]]), k=a.k)]
        if set(q["gold"]) & set(top):
            continue
        gold_servers = {server[g] for g in q["gold"]}
        kind = "right server" if top and server[top[0]] in gold_servers else "wrong server"
        kinds[kind] += 1
        by_diff[q.get("difficulty", "?")] += 1
        for s in gold_servers:
            by_srv[s] += 1
        misses.append((q, top, kind))

    print("%s: %d of %d requests missed (top %d)\n" % (a.retriever, len(misses), total, a.k))
    print("how:        " + ", ".join("%s %d" % kv for kv in kinds.most_common()))
    print("difficulty: " + ", ".join("%s %d" % kv for kv in by_diff.most_common()))
    print("server:     " + ", ".join("%s %d" % kv for kv in by_srv.most_common()))
    print()
    for q, top, kind in misses[: a.show]:
        print("%s [%s, %s] %s" % (q["id"], q.get("difficulty"), kind, q["query"]))
        print("    wanted: %s" % ", ".join(q["gold"]))
        print("    got:    %s" % ", ".join(top[:3]))


if __name__ == "__main__":
    main()
