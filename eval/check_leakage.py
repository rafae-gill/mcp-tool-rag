"""Make sure example requests didn't leak the test set.

    python eval/check_leakage.py
    python eval/check_leakage.py --data external/bfcl_search --examples external/bfcl_search/examples.json

Examples are written from the tool catalog only. This checks that none of
them is (nearly) a copy of a test request anyway: by shared words, and by
embedding similarity. Anything flagged is printed for a human to judge; a
real leak means rewriting that example before trusting any +ex score.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcp_tool_router import tokenize  # noqa: E402

WORD_OVERLAP = 0.6       # Jaccard on content words
COSINE = 0.92            # near-paraphrase in embedding space


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--examples", default=str(ROOT / "data" / "enrichment" / "examples.json"))
    ap.add_argument("--show", type=int, default=30)
    a = ap.parse_args()
    examples = json.loads(Path(a.examples).read_text(encoding="utf-8"))
    queries = [json.loads(l) for l in (Path(a.data) / "queries.jsonl").read_text(encoding="utf-8").splitlines()
               if l.strip()]
    flat = [(tool, e) for tool, es in examples.items() for e in es]

    flagged = []
    qsets = [(q, set(tokenize(q["query"]))) for q in queries]
    for tool, e in flat:
        es = set(tokenize(e))
        for q, qs in qsets:
            if es and qs:
                j = len(es & qs) / len(es | qs)
                if j >= WORD_OVERLAP:
                    flagged.append(("words %.2f" % j, tool, e, q["id"], q["query"]))

    try:
        import numpy as np
        from fastembed import TextEmbedding
        m = TextEmbedding("BAAI/bge-small-en-v1.5")
        unit = lambda rows: (lambda v: v / np.linalg.norm(v, axis=1, keepdims=True))(np.array(list(rows)))
        ev = unit(m.query_embed([e for _, e in flat]))
        qv = unit(m.query_embed([q["query"] for q in queries]))
        sims = ev @ qv.T
        for i, j in zip(*np.where(sims >= COSINE)):
            flagged.append(("cosine %.2f" % sims[i, j], flat[i][0], flat[i][1],
                            queries[j]["id"], queries[j]["query"]))
        print("max example-to-test similarity: %.3f (median of per-example max: %.3f)"
              % (sims.max(), float(np.median(sims.max(axis=1)))))
    except ImportError:
        print("(fastembed not installed: word-overlap check only)")

    print("%d examples checked against %d test requests; %d flagged"
          % (len(flat), len(queries), len(flagged)))
    for why, tool, e, qid, q in flagged[: a.show]:
        print("  [%s] %s: %r\n      ~ %s: %r" % (why, tool, e, qid, q))


if __name__ == "__main__":
    main()
