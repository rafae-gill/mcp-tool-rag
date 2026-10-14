"""One request, two people, three answers.

    python examples/demo.py
    python examples/demo.py "pause the webinar ads, budget's gone"

Runs the same request for two roles from the synthetic company, plus a
permission-unaware search for comparison. Uses the best method from the
evaluation: keyword + embeddings fused by rank, with example requests.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from mcp_tool_rag import (BM25, Embeddings, EnrichedEmbeddings, Hybrid, Tool,  # noqa: E402
                          ToolRouter, allowed_for, load_catalog)

REQUEST = (sys.argv[1] if len(sys.argv) > 1 else
           "this customer got charged twice, give them their money back")
ROLES = ("finance", "support")

catalog = load_catalog(ROOT / "data" / "catalog.json")
roles = json.loads((ROOT / "data" / "roles.json").read_text(encoding="utf-8"))
examples = json.loads((ROOT / "data" / "enrichment" / "examples.json").read_text(encoding="utf-8"))

bm25 = BM25([Tool(t.server, t.name, t.description + ". Example requests: "
                  + " | ".join(examples.get(t.name, [])), t.params) for t in catalog])
emb = EnrichedEmbeddings(Embeddings(catalog), catalog, examples)
router = ToolRouter(catalog, Hybrid(bm25, emb))

print('request: "%s"\n' % REQUEST)
for role in ROLES:
    top = router.search(REQUEST, allowed_for(roles[role]), k=3)
    print("as %-8s -> %s" % (role, ", ".join(t.name for t in top)))

support = allowed_for(roles["support"])
unaware = router.search(REQUEST, None, k=3, mode="none")
print("\nordinary search, no permissions (support's view):")
for t in unaware:
    print("   %-24s %s" % (t.name, "" if support(t) else "<- support may NOT use this"))
