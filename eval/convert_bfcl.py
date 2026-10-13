"""Turn BFCL "live" into a tool-SEARCH test.

    python eval/fetch_external.py
    python eval/convert_bfcl.py          # -> external/bfcl_search/

In BFCL each request comes with the 1–4 functions to choose from. Here every
function from every request goes into ONE catalog, so each request has to find
its tool among all of them — a retrieval problem, which is what a router
faces. The requests themselves are BFCL's user-contributed ones, unchanged.

Details that matter:
  * The same function often appears in several requests with slightly
    different definitions (`get_current_weather` in 46 versions). They are one
    function to a person asking for it, so they are merged into one catalog
    entry, keeping the most complete definition. Counting a different version
    of the same function as a miss would measure versioning, not search.
  * Gold = the function(s) called in BFCL's ground-truth answer.
  * There are no permissions in BFCL, so every tool is allowed (role "all").
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "external" / "bfcl"
OUT = ROOT / "external" / "bfcl_search"
SETS = ["simple", "multiple", "parallel", "parallel_multiple"]


def rows(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def user_text(question) -> str:
    """The user's request: the last user message of the first turn."""
    turn = question[0] if question and isinstance(question[0], list) else question
    users = [m.get("content", "") for m in turn if m.get("role") == "user"]
    return (users[-1] if users else "").strip()


def main():
    catalog: dict[str, dict] = {}                 # one entry per function name
    queries = []

    def richness(entry):
        return len(entry["description"]) + sum(len(k) + len(v) for k, v in entry["params"].items())

    for s in SETS:
        reqs = rows(SRC / ("BFCL_v4_live_%s.json" % s))
        answers = {r["id"]: r for r in rows(SRC / "possible_answer" / ("BFCL_v4_live_%s.json" % s))}
        for r in reqs:
            names = set()
            for fn in r["function"]:
                props = (fn.get("parameters") or {}).get("properties") or {}
                entry = {"server": "bfcl", "name": fn["name"],
                         "description": (fn.get("description") or "").strip(),
                         "params": {p: (v or {}).get("description", "") or "" for p, v in props.items()}}
                if fn["name"] not in catalog or richness(entry) > richness(catalog[fn["name"]]):
                    catalog[fn["name"]] = entry
                names.add(fn["name"])

            a = answers.get(r["id"])
            text = user_text(r.get("question"))
            if not a or not text:
                continue
            gold = []
            for call in a["ground_truth"]:
                for fname in call:
                    if fname in names and fname not in gold:
                        gold.append(fname)
            if not gold:
                continue
            queries.append({"id": r["id"], "role": "all", "query": text, "gold": gold,
                            "difficulty": s, "forbidden_best": False, "allowed_fallback": []})

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "catalog.json").write_text(json.dumps(list(catalog.values()), indent=1,
                                                 ensure_ascii=False), encoding="utf-8")
    (OUT / "queries.jsonl").write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in queries),
                                       encoding="utf-8")
    print("%d tools, %d requests -> %s" % (len(catalog), len(queries), OUT))


if __name__ == "__main__":
    main()
