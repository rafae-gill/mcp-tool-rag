"""Score keyword, embedding and hybrid tool search on a labelled dataset.

    python eval/run_eval.py                       # data/ -> results/
    python eval/run_eval.py --data data/real --out results/real
    python eval/run_eval.py --retrievers bm25     # no model download needed

A dataset directory holds:
    catalog.json   [{"server","name","description","params"}]
    roles.json     {"role": {"server": {"tools": [...]}}}    (mcp-gatekeeper shape)
    queries.jsonl  {"id","role","query","gold":[...],"difficulty",
                    "forbidden_best": bool, "allowed_fallback": [...]}

What is measured (k = 5 unless set):
    hit@1/3/5      a correct tool is in the top 1/3/5 the caller is allowed
    recall@5       share of ALL needed tools in the top 5 (matters for "multi")
    MRR            1 / rank of the first correct tool (top 10), averaged
    after-filter   the same search run permission-unaware, forbidden tools
                   dropped afterwards: what bolting permissions on costs
    leak@5         share of requests where a permission-UNAWARE top 5 shows
                   at least one tool the caller may not use
    fallback@5     for requests whose best tool is forbidden: an allowed tool
                   that still helps is in the top 5
    ms/query       median search time, permission filter included
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcp_tool_router import BM25, Embeddings, Hybrid, ToolRouter, allowed_for, load_catalog  # noqa: E402

DIFFICULTIES = ("easy", "paraphrase", "ambiguous", "multi", "real")


def load(data: Path):
    catalog = load_catalog(data / "catalog.json")
    if (data / "roles.json").exists():
        roles = json.loads((data / "roles.json").read_text(encoding="utf-8"))
    else:
        # No roles given: one role that may use every tool. Relevance is still
        # measured; the permission rows simply come out as "no leaks".
        roles = {"all": {s: {"tools": ["*"]} for s in {t.server for t in catalog}}}
    if (data / "queries.csv").exists():
        queries = _queries_from_csv(data / "queries.csv")
    else:
        queries = [json.loads(l) for l in
                   (data / "queries.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    # An independent review (labels_review.jsonl) may accept extra tools as
    # equally correct, or correct a wrong label. Scored alongside the
    # original labels, never instead of them.
    review_path = data / "labels_review.jsonl"
    review = {}
    if review_path.exists():
        review = {r["id"]: r for r in (json.loads(l) for l in
                  review_path.read_text(encoding="utf-8").splitlines() if l.strip())}
    for q in queries:
        r = review.get(q["id"])
        if r and r["verdict"] == "wrong_gold":
            q["gold_reviewed"] = list(r["suggested_gold"])
        elif r and r["verdict"] == "add_alternates":
            q["gold_reviewed"] = q["gold"] + [a for a in r["alternates"] if a not in q["gold"]]
        else:
            q["gold_reviewed"] = list(q["gold"])
    names = {t.name for t in catalog}
    for q in queries:
        for g in q["gold"] + q.get("allowed_fallback", []):
            if g not in names:
                raise SystemExit("query %s names unknown tool %r" % (q["id"], g))
        if q["role"] not in roles:
            raise SystemExit("query %s has unknown role %r" % (q["id"], q["role"]))
    return catalog, roles, queries


def _queries_from_csv(path: Path):
    """A spreadsheet is easier to fill in than JSON. Columns: query, gold
    (tool names separated by ';'), and optionally role, difficulty,
    forbidden_best (yes/no), allowed_fallback (';'-separated), id."""
    import csv
    split = lambda s: [x.strip() for x in (s or "").split(";") if x.strip()]
    out = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for i, row in enumerate(csv.DictReader(f), 1):
            if not (row.get("query") or "").strip():
                continue
            out.append({
                "id": (row.get("id") or "").strip() or "r%03d" % i,
                "role": (row.get("role") or "").strip() or "all",
                "query": row["query"].strip(),
                "gold": split(row.get("gold")),
                "difficulty": (row.get("difficulty") or "").strip() or "real",
                "forbidden_best": (row.get("forbidden_best") or "").strip().lower() in ("yes", "true", "1"),
                "allowed_fallback": split(row.get("allowed_fallback")),
            })
    return out


def build(names, catalog, examples=None):
    """Retrievers by name. The `+ex` variants also use example requests per
    tool (data/enrichment/examples.json), written without seeing the test set."""
    from mcp_tool_router import EnrichedEmbeddings, Tool
    made = {}

    def emb():
        if "embeddings" not in made:
            made["embeddings"] = Embeddings(catalog)
        return made["embeddings"]

    def bm25():
        if "bm25" not in made:
            made["bm25"] = BM25(catalog)
        return made["bm25"]

    def need_examples(n):
        if not examples:
            raise SystemExit("%s needs --examples (no examples file found)" % n)
        return examples

    def bm25_ex():
        if "bm25+ex" not in made:
            ex = need_examples("bm25+ex")
            made["bm25+ex"] = BM25([Tool(t.server, t.name, t.description + ". Example requests: "
                                         + " | ".join(ex.get(t.name, [])), t.params)
                                    for t in catalog])
        return made["bm25+ex"]

    def emb_ex():
        if "emb+ex" not in made:
            made["emb+ex"] = EnrichedEmbeddings(emb(), catalog, need_examples("emb+ex"), mode="max")
        return made["emb+ex"]

    for n in names:
        if n == "bm25":
            bm25()
        elif n == "embeddings":
            emb()
        elif n == "hybrid":
            made[n] = Hybrid(bm25(), emb())
        elif n == "bm25+ex":
            bm25_ex()
        elif n == "emb+ex":
            emb_ex()
        elif n == "emb+ex-concat":
            made[n] = EnrichedEmbeddings(emb(), catalog, need_examples(n), mode="concat")
        elif n == "hybrid+ex":
            made[n] = Hybrid(bm25_ex(), emb_ex())
        else:
            raise SystemExit("unknown retriever %r" % n)
    return {n: made[n] for n in names}


def score(router, catalog, roles, queries, k):
    allowed = {r: allowed_for(p) for r, p in roles.items()}
    by_name = {t.name: t for t in catalog}
    acc = defaultdict(list)
    by_diff = defaultdict(lambda: defaultdict(list))
    times = []

    for q in queries:
        ok = allowed[q["role"]]
        gold = set(q["gold"])

        t0 = time.perf_counter()
        top10 = [t.name for t in router.search(q["query"], ok, k=10, mode="before")]
        times.append((time.perf_counter() - t0) * 1000)

        # The router must never return a forbidden tool. Not a metric: a bug.
        assert all(ok(by_name[n]) for n in top10), q["id"]

        unaware = router.search(q["query"], None, k=k, mode="none")
        acc["leak"].append(any(not ok(t) for t in unaware))

        if q.get("forbidden_best"):
            acc["forbidden_gold_shown_unaware"].append(any(t.name in gold for t in unaware))
            fb = set(q.get("allowed_fallback") or [])
            if fb:
                acc["fallback"].append(bool(fb & set(top10[:k])))
            continue

        first = next((i for i, n in enumerate(top10, 1) if n in gold), None)
        hits = {kk: bool(gold & set(top10[:kk])) for kk in (1, 3, k, 10)}
        acc["hit1"].append(hits[1])
        acc["hit3"].append(hits[3])
        acc["hitk"].append(hits[k])
        acc["hit10"].append(hits[10])
        rev = set(q.get("gold_reviewed") or q["gold"])
        acc["rev_hit1"].append(bool(rev & set(top10[:1])))
        acc["rev_hitk"].append(bool(rev & set(top10[:k])))
        acc["recallk"].append(len(gold & set(top10[:k])) / len(gold))
        acc["mrr"].append(1.0 / first if first else 0.0)
        by_diff[q.get("difficulty", "?")]["hit3"].append(hits[3])
        by_diff[q.get("difficulty", "?")]["hitk"].append(hits[k])

        after = [t.name for t in router.search(q["query"], ok, k=k, mode="after")]
        acc["after_hitk"].append(bool(gold & set(after)))
        acc["after_returned"].append(len(after))

    mean = lambda xs: (sum(xs) / len(xs)) if xs else None
    return {
        "hit@1": mean(acc["hit1"]), "hit@3": mean(acc["hit3"]), "hit@%d" % k: mean(acc["hitk"]), "hit@10": mean(acc["hit10"]),
        "recall@%d" % k: mean(acc["recallk"]), "mrr@10": mean(acc["mrr"]),
        "reviewed_hit@1": mean(acc["rev_hit1"]), "reviewed_hit@%d" % k: mean(acc["rev_hitk"]),
        "after_filter_hit@%d" % k: mean(acc["after_hitk"]),
        "after_filter_avg_returned": mean(acc["after_returned"]),
        "leak@%d_unaware" % k: mean(acc["leak"]),
        "forbidden_best_shown_unaware": mean(acc["forbidden_gold_shown_unaware"]),
        "fallback@%d" % k: mean(acc["fallback"]),
        "ms_per_query_median": statistics.median(times) if times else None,
        "by_difficulty": {d: {"n": len(v["hit3"]), "hit@3": mean(v["hit3"]),
                              "hit@%d" % k: mean(v["hitk"])}
                          for d, v in sorted(by_diff.items())},
    }


def context_estimate(catalog, roles, k):
    """Rough size of tool definitions an agent would load: all allowed tools vs
    the top k. Estimated as JSON characters / 4 — a rule of thumb, not a
    tokenizer count."""
    size = {t.name: len(json.dumps({"name": t.name, "description": t.description,
                                    "params": t.params})) / 4 for t in catalog}
    out = {}
    for r, perms in roles.items():
        ok = allowed_for(perms)
        allowed = [t for t in catalog if ok(t)]
        out[r] = {"allowed_tools": len(allowed),
                  "all_allowed_tokens_est": round(sum(size[t.name] for t in allowed)),
                  "top_k_tokens_est": round(k * statistics.mean(size[t.name] for t in allowed))
                  if allowed else 0}
    return out


def markdown(dataset, catalog, queries, results, ctx, k):
    pct = lambda x: "–" if x is None else "%.0f%%" % (100 * x)
    n_ok = sum(1 for q in queries if not q.get("forbidden_best"))
    n_fb = len(queries) - n_ok
    lines = ["# Tool search results: `%s`" % dataset, "",
             "%d tools on %d servers · %d requests (%d where the best tool is forbidden for the "
             "caller) · k = %d" % (len(catalog), len({t.server for t in catalog}), len(queries), n_fb, k),
             "", "| | " + " | ".join(results) + " |", "|---|" + "---|" * len(results)]
    rows = [("Right tool ranked 1st", "hit@1"), ("Right tool in top 3", "hit@3"),
            ("Right tool in top %d" % k, "hit@%d" % k), ("Right tool in top 10", "hit@10"), ("All needed tools in top %d" % k, "recall@%d" % k),
            ("MRR (top 10)", "mrr@10"),
            ("Ranked 1st, reviewed labels", "reviewed_hit@1"),
            ("In top %d, reviewed labels" % k, "reviewed_hit@%d" % k),
            ("Top %d, permissions bolted on after" % k, "after_filter_hit@%d" % k),
            ("Permission-unaware top %d shows a forbidden tool" % k, "leak@%d_unaware" % k),
            ("…and it's the forbidden tool the request wanted", "forbidden_best_shown_unaware"),
            ("Best tool forbidden: a helpful allowed tool in top %d" % k, "fallback@%d" % k)]
    for label, key in rows:
        cells = []
        for r in results.values():
            v = r[key]
            cells.append("%.2f" % v if key == "mrr@10" and v is not None else pct(v))
        lines.append("| %s | %s |" % (label, " | ".join(cells)))
    lines.append("| Median ms per search | %s |" % " | ".join(
        "%.1f" % r["ms_per_query_median"] for r in results.values()))
    lines += ["", "**Right tool in top 3, by difficulty**", "",
              "| Difficulty | n | " + " | ".join(results) + " |", "|---|---|" + "---|" * len(results)]
    for d in DIFFICULTIES:
        first = next(iter(results.values()))["by_difficulty"].get(d)
        if not first:
            continue
        lines.append("| %s | %d | %s |" % (d, first["n"], " | ".join(
            pct(r["by_difficulty"][d]["hit@3"]) for r in results.values())))
    lines += ["", "**Tool definitions an agent loads (estimated tokens: JSON chars / 4)**", "",
              "| Role | Tools allowed | All allowed tools | Top %d |" % k, "|---|---|---|---|"]
    for role, c in ctx.items():
        lines.append("| %s | %d | ~%d | ~%d |" % (role, c["allowed_tools"],
                                                   c["all_allowed_tokens_est"], c["top_k_tokens_est"]))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--retrievers", default="bm25,embeddings,hybrid")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--examples", default=str(ROOT / "data" / "enrichment" / "examples.json"),
                    help="example requests per tool, for the +ex retrievers")
    a = ap.parse_args()

    data, out = Path(a.data), Path(a.out)
    catalog, roles, queries = load(data)
    ex_path = Path(a.examples)
    examples = json.loads(ex_path.read_text(encoding="utf-8")) if ex_path.exists() else None
    retrievers = build([r.strip() for r in a.retrievers.split(",") if r.strip()], catalog, examples)
    results = {}
    for name, r in retrievers.items():
        router = ToolRouter(catalog, r)
        router.search("warm up", lambda t: True)
        results[name] = score(router, catalog, roles, queries, a.k)
        print("%-11s hit@1 %.2f  hit@3 %.2f  hit@%d %.2f  mrr %.2f  leak(unaware) %.2f" % (
            name, results[name]["hit@1"], results[name]["hit@3"], a.k,
            results[name]["hit@%d" % a.k], results[name]["mrr@10"],
            results[name]["leak@%d_unaware" % a.k]))

    ctx = context_estimate(catalog, roles, a.k)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(
        {"dataset": str(data.name), "k": a.k, "results": results, "context": ctx}, indent=2))
    (out / "results.md").write_text(markdown(data.name, catalog, queries, results, ctx, a.k),
                                    encoding="utf-8")
    print("wrote", out / "results.md")


if __name__ == "__main__":
    main()
