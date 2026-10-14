"""Three ways to rank tools against a request.

Every retriever takes the candidate tools at query time rather than indexing
once, because the candidate set is different for every caller: permission
filtering happens BEFORE ranking (see router.py). Expensive work that doesn't
depend on the caller — document statistics, embeddings — is computed once for
the whole catalog and reused.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from .catalog import Tool

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset("""a an and are as at be by can do does for from get give how i in is it
me my of on or our please show so that the their them this to us we what when where which
who with you your""".split())


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


class BM25:
    """Okapi BM25 over each tool's search text. The keyword baseline.

    Statistics (document frequency, average length) come from the WHOLE
    catalog, so a tool's score doesn't change depending on which other tools
    a caller happens to be allowed.
    """

    name = "bm25"

    def __init__(self, catalog: list[Tool], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = {t.name: Counter(tokenize(t.search_text)) for t in catalog}
        self.lens = {n: sum(c.values()) for n, c in self.docs.items()}
        self.avg = sum(self.lens.values()) / max(1, len(self.lens))
        df = Counter(w for c in self.docs.values() for w in c)
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        # Inverted index: term -> [(tool, its full BM25 contribution)]. Every
        # part of the formula is known per (term, tool) ahead of time, so a
        # query only touches tools that contain one of its words.
        self.postings: dict[str, list[tuple[str, float]]] = {}
        for name, tf in self.docs.items():
            norm = self.k1 * (1 - self.b + self.b * self.lens[name] / self.avg)
            for w, f in tf.items():
                self.postings.setdefault(w, []).append(
                    (name, self.idf[w] * f * (self.k1 + 1) / (f + norm)))

    def scores(self, query: str, candidates: list[Tool]) -> dict[str, float]:
        out = {t.name: 0.0 for t in candidates}
        for w in tokenize(query):
            for name, s in self.postings.get(w, ()):
                if name in out:
                    out[name] += s
        return out


class Embeddings:
    """Cosine similarity between a request and each tool, using a small local
    embedding model (fastembed + ONNX: no GPU, no API key, no per-call cost).

    Tool vectors are computed once for the whole catalog; each request costs
    one query embedding.
    """

    name = "embeddings"

    def __init__(self, catalog: list[Tool], model: str = "BAAI/bge-small-en-v1.5"):
        try:
            from fastembed import TextEmbedding
        except ImportError as e:                                   # pragma: no cover
            raise ImportError('pip install "mcp-tool-rag[embeddings]"') from e
        import numpy as np

        self._np = np
        self.model_name = model
        self.model = TextEmbedding(model_name=model)
        self.names = [t.name for t in catalog]
        self.index = {n: i for i, n in enumerate(self.names)}
        vecs = np.array(list(self.model.passage_embed([t.search_text for t in catalog])))
        self.matrix = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        self.vec = dict(zip(self.names, self.matrix))
        self._qcache: dict[str, object] = {}

    def query_vector(self, query: str):
        """Embed a request once; repeated searches for it reuse the vector."""
        v = self._qcache.get(query)
        if v is None:
            np = self._np
            v = np.array(next(iter(self.model.query_embed([query]))))
            v /= np.linalg.norm(v)
            if len(self._qcache) > 50_000:
                self._qcache.clear()
            self._qcache[query] = v
        return v

    def scores(self, query: str, candidates: list[Tool]) -> dict[str, float]:
        q = self.query_vector(query)
        idx = [self.index[t.name] for t in candidates]
        sims = self.matrix[idx] @ q
        return {t.name: float(s) for t, s in zip(candidates, sims)}


class EnrichedEmbeddings:
    """Embeddings that also match against example requests for each tool.

    A tool's description is written by whoever built the tool; a request is
    written by someone who never saw it. Example requests ("how people ask
    for this") bridge that gap. Two ways to use them:

        mode="max"     each example is its own vector; a tool scores as its
                       best-matching vector (description or any example)
        mode="concat"  examples are appended to the description and the
                       whole thing is one vector

    Reuses an `Embeddings` instance's model, so the model loads once.
    """

    def __init__(self, base: Embeddings, catalog: list[Tool], examples: dict[str, list[str]],
                 mode: str = "max"):
        if mode not in ("max", "concat"):
            raise ValueError("mode must be 'max' or 'concat'")
        np = base._np
        self._np, self.base, self.mode = np, base, mode
        self.name = "embeddings+examples" + ("" if mode == "max" else "(concat)")
        self.index = dict(base.index)

        def unit(rows):
            v = np.array(list(rows))
            return v / np.linalg.norm(v, axis=1, keepdims=True)

        if mode == "concat":
            texts = [t.search_text + ". Example requests: " + " | ".join(examples.get(t.name, []))
                     for t in catalog]
            self.vectors = unit(base.model.passage_embed(texts))
            self.owner = np.array([base.index[t.name] for t in catalog])
        else:
            ex_texts, owners = [], []
            for t in catalog:
                for e in examples.get(t.name, []):
                    ex_texts.append(e)
                    owners.append(base.index[t.name])
            # Examples are requests, so they're embedded the way requests are.
            parts = [base.matrix]
            if ex_texts:
                parts.append(unit(base.model.query_embed(ex_texts)))
            self.vectors = np.vstack(parts)
            self.owner = np.concatenate([np.arange(len(base.names)),
                                         np.array(owners, dtype=int)])
        self.n_tools = len(base.names)

    def scores(self, query: str, candidates: list[Tool]) -> dict[str, float]:
        np = self._np
        sims = self.vectors @ self.base.query_vector(query)
        best = np.full(self.n_tools, -np.inf)
        np.maximum.at(best, self.owner, sims)      # each tool: its best vector
        return {t.name: float(best[self.index[t.name]]) for t in candidates}


class Hybrid:
    """Reciprocal-rank fusion of keyword and embedding rankings.

    RRF uses only each method's RANK, not its raw score, so the two scales
    (BM25's unbounded sums, cosine's -1..1) never need calibrating against
    each other. k=60 is the constant from the original RRF paper.
    """

    name = "hybrid"

    def __init__(self, *retrievers, k: int = 60):
        self.parts, self.k = retrievers, k

    def scores(self, query: str, candidates: list[Tool]) -> dict[str, float]:
        fused = {t.name: 0.0 for t in candidates}
        for r in self.parts:
            ranked = sorted(r.scores(query, candidates).items(), key=lambda kv: -kv[1])
            for rank, (name, _) in enumerate(ranked, start=1):
                fused[name] += 1.0 / (self.k + rank)
        return fused
