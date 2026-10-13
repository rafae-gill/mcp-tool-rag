"""Sanity checks for catalog.json, roles.json and queries.jsonl.

    py check_dataset.py            # hard checks + counts + name-overlap list
    py check_dataset.py -v         # also list description-word overlaps (informational)

Exits non-zero if any hard check fails. The name-overlap list is a warning:
each entry should be rewritten or be easy to defend.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIFFICULTIES = {"easy", "paraphrase", "ambiguous", "multi"}
MIN_WORD = 5
STOP = {"about", "after", "again", "their", "there", "these", "those", "which", "while", "where",
        "would", "could", "should", "every", "other", "under", "until", "since", "before", "between",
        "optional", "default", "return", "returns", "including", "without"}


def split_identifier(s: str) -> list[str]:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    return [w for w in re.split(r"[_\-.\s]+", s.lower()) if w]


def words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def related(a: str, b: str) -> bool:
    """Same word or an inflection of it: equal, or a shared prefix of >= 5
    letters that covers all but at most 2 letters of the shorter word
    (deploy/deploys/deployment, refund/refunded, renew/renewals)."""
    if len(a) < MIN_WORD or len(b) < MIN_WORD:
        return False
    if a == b:
        return True
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n >= MIN_WORD and n >= min(len(a), len(b)) - 2


def main() -> int:
    verbose = "-v" in sys.argv
    errors: list[str] = []

    catalog = json.loads((HERE / "catalog.json").read_text(encoding="utf-8"))
    roles = json.loads((HERE / "roles.json").read_text(encoding="utf-8"))
    queries = [json.loads(l) for l in (HERE / "queries.jsonl").read_text(encoding="utf-8").splitlines()
               if l.strip()]

    # --- catalog -----------------------------------------------------------
    names = [t["name"] for t in catalog]
    for n, c in Counter(names).items():
        if c > 1:
            errors.append("duplicate tool name: %s" % n)
    for t in catalog:
        for k in ("server", "name", "description", "params"):
            if k not in t:
                errors.append("tool %s missing field %s" % (t.get("name"), k))
    server_of = {t["name"]: t["server"] for t in catalog}
    desc_of = {t["name"]: t["description"] for t in catalog}

    # --- roles -------------------------------------------------------------
    allowed: dict[str, set[str]] = {}
    for role, perms in roles.items():
        allowed[role] = set()
        for server, spec in perms.items():
            for tool in spec.get("tools", []):
                if tool == "*":
                    errors.append("role %s uses '*' on %s" % (role, server))
                elif tool not in server_of:
                    errors.append("role %s lists unknown tool %s" % (role, tool))
                elif server_of[tool] != server:
                    errors.append("role %s lists %s under %s but it belongs to %s"
                                  % (role, tool, server, server_of[tool]))
                else:
                    allowed[role].add(tool)

    # --- queries -----------------------------------------------------------
    ids = [q.get("id") for q in queries]
    for i, c in Counter(ids).items():
        if c > 1:
            errors.append("duplicate query id %s" % i)
    texts = Counter(q["query"].strip().lower() for q in queries)
    for t, c in texts.items():
        if c > 1:
            errors.append("duplicate query text: %r" % t)

    overlaps = []
    desc_overlaps = []
    for q in queries:
        qid, role = q["id"], q["role"]
        if role not in roles:
            errors.append("%s: unknown role %s" % (qid, role))
            continue
        if q.get("difficulty") not in DIFFICULTIES:
            errors.append("%s: bad difficulty %r" % (qid, q.get("difficulty")))
        gold = q.get("gold") or []
        if not gold:
            errors.append("%s: empty gold" % qid)
        if q["difficulty"] == "multi" and len(gold) < 2:
            errors.append("%s: multi query with <2 gold tools" % qid)
        for g in gold:
            if g not in server_of:
                errors.append("%s: gold tool %s not in catalog" % (qid, g))
        fb = q.get("forbidden_best")
        if not isinstance(fb, bool):
            errors.append("%s: forbidden_best must be a bool" % qid)
        if fb:
            if "allowed_fallback" not in q:
                errors.append("%s: forbidden_best query lacks allowed_fallback" % qid)
            for g in gold:
                if g in allowed[role]:
                    errors.append("%s: forbidden_best but gold %s IS allowed for %s" % (qid, g, role))
            for f in q.get("allowed_fallback", []):
                if f not in server_of:
                    errors.append("%s: fallback %s not in catalog" % (qid, f))
                elif f not in allowed[role]:
                    errors.append("%s: fallback %s not allowed for %s" % (qid, f, role))
        else:
            if q.get("allowed_fallback"):
                errors.append("%s: allowed_fallback on a non-forbidden query" % qid)
            for g in gold:
                if g in server_of and g not in allowed[role]:
                    errors.append("%s: gold %s not allowed for role %s" % (qid, g, role))

        qwords = set(words(q["query"]))
        for g in gold:
            hits = sorted({"%s~%s" % (w, n) for w in qwords for n in split_identifier(g) if related(w, n)})
            if hits:
                overlaps.append((qid, g, hits, q["query"]))
            dwords = {w for w in words(desc_of.get(g, "")) if len(w) >= MIN_WORD and w not in STOP}
            dh = sorted({w for w in qwords if w not in STOP and any(related(w, d) for d in dwords)})
            if dh:
                desc_overlaps.append((qid, g, dh))

    # --- report ------------------------------------------------------------
    print("tools: %d" % len(catalog))
    for s, c in sorted(Counter(server_of.values()).items()):
        print("  %-10s %d" % (s, c))
    print("\nallowed tools per role:")
    for r in roles:
        print("  %-10s %3d  servers: %s" % (r, len(allowed[r]), ", ".join(roles[r])))

    n = len(queries)
    print("\nqueries: %d" % n)
    print("by difficulty:")
    for d, c in sorted(Counter(q["difficulty"] for q in queries).items()):
        print("  %-11s %3d  (%.0f%%)" % (d, c, 100.0 * c / n))
    print("by role:")
    for r, c in sorted(Counter(q["role"] for q in queries).items()):
        f = sum(1 for q in queries if q["role"] == r and q.get("forbidden_best"))
        print("  %-10s %3d  (forbidden_best: %d)" % (r, c, f))
    print("by server (queries with at least one gold tool on it):")
    sc = Counter()
    for q in queries:
        for s in {server_of[g] for g in q["gold"] if g in server_of}:
            sc[s] += 1
    for s, c in sorted(sc.items()):
        print("  %-10s %3d" % (s, c))
    used = {g for q in queries for g in q["gold"]}
    print("distinct gold tools: %d of %d" % (len(used), len(catalog)))
    print("gold size: %s" % dict(sorted(Counter(len(q["gold"]) for q in queries).items())))
    print("forbidden_best: %d (with empty fallback: %d)" % (
        sum(1 for q in queries if q.get("forbidden_best")),
        sum(1 for q in queries if q.get("forbidden_best") and not q.get("allowed_fallback"))))

    print("\nname-word overlaps (query word ~ gold tool-name word, %d+ letters): %d" % (MIN_WORD, len(overlaps)))
    for qid, g, hits, text in overlaps:
        print("  %s  %-28s %-24s %s" % (qid, g, ",".join(hits), text))

    print("\ndescription-word overlaps (informational): %d query/tool pairs" % len(desc_overlaps))
    if verbose:
        for qid, g, dh in desc_overlaps:
            print("  %s  %-28s %s" % (qid, g, ",".join(dh)))

    if errors:
        print("\nERRORS (%d):" % len(errors))
        for e in errors:
            print("  " + e)
        return 1
    print("\nall hard checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
