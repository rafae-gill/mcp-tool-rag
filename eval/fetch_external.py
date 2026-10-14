"""Download the public benchmarks into external/ (git-ignored). No dependencies.

    python eval/fetch_external.py

The data is never re-published in this repo: this script fetches it from the
original sources, and only our scores on it are published.

    BFCL "live"  Berkeley Function Calling Leaderboard, user-contributed
                 requests. github.com/ShishirPatil/gorilla (Apache-2.0).
                 Patil et al., ICML 2025.
    ToolRet      Tool-retrieval benchmark. huggingface.co/datasets/mangopy
                 (code Apache-2.0; bundles ~30 earlier datasets, each with its
                 own terms). Shi et al., "Retrieval Models Aren't Tool-Savvy",
                 Findings of ACL 2025.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "external"

BFCL = ("https://raw.githubusercontent.com/ShishirPatil/gorilla/main/"
        "berkeley-function-call-leaderboard/bfcl_eval/data/")
BFCL_FILES = ["BFCL_v4_live_simple.json", "BFCL_v4_live_multiple.json",
              "BFCL_v4_live_parallel.json", "BFCL_v4_live_parallel_multiple.json",
              "BFCL_v4_live_relevance.json", "BFCL_v4_live_irrelevance.json",
              "possible_answer/BFCL_v4_live_simple.json",
              "possible_answer/BFCL_v4_live_multiple.json",
              "possible_answer/BFCL_v4_live_parallel.json",
              "possible_answer/BFCL_v4_live_parallel_multiple.json"]

HF = "https://huggingface.co/"
TOOLRET = ["mangopy/ToolRet-Queries", "mangopy/ToolRet-Tools"]


def get(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"user-agent": "mcp-tool-rag-eval"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())


def main():
    total = 0
    for name in BFCL_FILES:
        dest = OUT / "bfcl" / name
        get(BFCL + name, dest)
        total += dest.stat().st_size

    for repo in TOOLRET:
        with urllib.request.urlopen(HF + "api/datasets/" + repo, timeout=60) as r:
            files = [s["rfilename"] for s in json.load(r)["siblings"]
                     if s["rfilename"].endswith((".parquet", "README.md"))]
        for name in files:
            dest = OUT / "toolret" / repo.split("/")[1] / name
            get(HF + "datasets/%s/resolve/main/%s" % (repo, name), dest)
            total += dest.stat().st_size

    print("external/ ready: %.1f MB" % (total / 1e6))


if __name__ == "__main__":
    sys.exit(main())
