"""Score keyword, embedding and hybrid search on ToolRet, the way ToolRet does.

    python eval/fetch_external.py
    python eval/toolret_eval.py                  # -> results/toolret/

Protocol, following the benchmark's own code (github.com/mangopy/tool-
retrieval-benchmark, toolret/eval.py), so the numbers sit next to the
published ones:
  * every request searches the FULL merged corpus (~44k tools);
  * each tool is indexed by its `documentation` text, as-is;
  * NDCG@10, Recall@10 and C@10 (all of a request's tools in the top 10)
    are averaged over the requests of each category (web / code /
    customized), and "Avg" is the mean of the three categories;
  * two settings: the request alone ("w/o inst."), and the benchmark's task
    instruction prepended to it ("w/ inst.").

Hybrid fuses the top 1,000 of each method by reciprocal rank (k=60);
tools outside both top-1,000 lists would add < 1/1060 and can't reach a top 10.
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcp_tool_rag import BM25, Embeddings, Tool, tokenize  # noqa: E402

SRC = ROOT / "external" / "toolret"
OUT = ROOT / "results" / "toolret"
DEPTH, RRF_K, K = 1000, 60, 10


def load():
    import pyarrow.parquet as pq
    tools = []
    for f in sorted((SRC / "ToolRet-Tools").rglob("*.parquet")):
        for r in pq.read_table(f).to_pylist():
            tools.append(Tool(server=f.parent.name, name=r["id"], text=r["documentation"]))
    queries = []
    for f in sorted((SRC / "ToolRet-Queries").rglob("*.parquet")):
        for r in pq.read_table(f).to_pylist():
            rels = {str(x["id"]): int(x.get("relevance", 1)) for x in json.loads(r["labels"])}
            queries.append({"id": r["id"], "task": f.parent.name, "category": r["category"],
                            "query": r["query"], "instruction": r.get("instruction") or "",
                            "rels": rels})
    return tools, queries


def metrics(ranked: list[str], rels: dict[str, int]) -> tuple[float, float, float]:
    top = ranked[:K]
    dcg = sum(rels.get(t, 0) / math.log2(i + 2) for i, t in enumerate(top))
    ideal = sorted(rels.values(), reverse=True)[:K]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    hit = sum(1 for t in top if rels.get(t, 0) > 0)
    need = sum(1 for g in rels.values() if g > 0)
    recall = hit / need if need else 0.0
    return (dcg / idcg if idcg else 0.0), recall, float(recall == 1.0)


def bm25_rank(bm, names, text):
    acc = np.zeros(len(names))
    idx = bm._idx
    for w in tokenize(text):
        for name, s in bm.postings.get(w, ()):
            acc[idx[name]] += s
    top = np.argpartition(-acc, DEPTH)[:DEPTH]
    return top[np.argsort(-acc[top], kind="stable")]


def emb_rank(emb, text):
    sims = emb.matrix @ emb.query_vector(text)
    top = np.argpartition(-sims, DEPTH)[:DEPTH]
    return top[np.argsort(-sims[top], kind="stable")]


def rrf(*rankings):
    fused = defaultdict(float)
    for r in rankings:
        for rank, i in enumerate(r, 1):
            fused[int(i)] += 1.0 / (RRF_K + rank)
    return sorted(fused, key=lambda i: -fused[i])


def main():
    t0 = time.time()
    tools, queries = load()
    names = [t.name for t in tools]
    assert len(set(names)) == len(names), "tool ids must be unique"
    print("%d tools, %d requests (%.0fs)" % (len(tools), len(queries), time.time() - t0))

    bm = BM25(tools)
    bm._idx = {n: i for i, n in enumerate(names)}
    print("bm25 indexed (%.0fs)" % (time.time() - t0))

    cache = ROOT / "external" / "toolret_cache" / "tool_vectors_bge_small.npy"  # ~68 MB, git-ignored
    emb = Embeddings.__new__(Embeddings)          # build without re-embedding if cached
    from fastembed import TextEmbedding
    emb._np, emb.model_name = np, "BAAI/bge-small-en-v1.5"
    emb.model = TextEmbedding(model_name=emb.model_name)
    emb.names, emb.index, emb._qcache = names, {n: i for i, n in enumerate(names)}, {}
    if cache.exists():
        emb.matrix = np.load(cache)
    else:
        v = np.array(list(emb.model.passage_embed([t.text for t in tools], batch_size=64)))
        emb.matrix = v / np.linalg.norm(v, axis=1, keepdims=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, emb.matrix)
    print("embeddings ready (%.0fs)" % (time.time() - t0))

    scores = {}                       # (setting, method) -> category -> [ (ndcg, recall, comp) ]
    for setting in ("w/o inst.", "w/ inst."):
        for q in queries:
            text = (q["instruction"] + " " + q["query"]) if setting == "w/ inst." else q["query"]
            b = bm25_rank(bm, names, text)
            e = emb_rank(emb, text)
            h = rrf(b, e)
            for method, ranking in (("bm25", b), ("bge-small", e), ("hybrid", h)):
                ranked = [names[int(i)] for i in ranking[:K]]
                scores.setdefault((setting, method), defaultdict(list))[q["category"]].append(
                    metrics(ranked, q["rels"]))
        print("%s done (%.0fs)" % (setting, time.time() - t0))

    table, out = [], {}
    cats = ("web", "code", "customized")
    for (setting, method), by_cat in scores.items():
        per = {c: np.mean(by_cat[c], axis=0) * 100 for c in cats if by_cat.get(c)}
        avg = np.mean([per[c] for c in per], axis=0)
        out["%s | %s" % (setting, method)] = {
            **{c: {"N@10": round(v[0], 2), "R@10": round(v[1], 2), "C@10": round(v[2], 2)}
               for c, v in per.items()},
            "avg": {"N@10": round(avg[0], 2), "R@10": round(avg[1], 2), "C@10": round(avg[2], 2)}}
        table.append("| %s | %s | %.2f | %.2f | %s |" % (
            setting, method, avg[0], avg[2],
            " | ".join("%.2f / %.2f" % (per[c][0], per[c][2]) for c in cats)))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(out, indent=2))
    md = ["# ToolRet (full %d-tool corpus, %d requests)" % (len(tools), len(queries)), "",
          "| Setting | Method | Avg N@10 | Avg C@10 | Web N/C | Code N/C | Customized N/C |",
          "|---|---|---|---|---|---|---|"] + table
    (OUT / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
