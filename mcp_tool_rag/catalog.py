"""The tools a router can choose from, and the text each one is searched by."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Tool:
    server: str
    name: str
    description: str = ""
    params: dict = field(default_factory=dict, hash=False, compare=False)
    text: str = ""          # if set, searched as-is instead of the built text

    @property
    def search_text(self) -> str:
        """What retrieval sees: the name split into words, the server, the
        description, and every parameter with its description.

        Splitting the name matters for keyword search: `get_invoice_pdf` and
        `getInvoicePdf` should both match "invoice". `text`, when given,
        replaces all of this (benchmarks that define their own tool text).
        """
        if self.text:
            return self.text
        parts = [split_identifier(self.name), self.server, self.description]
        for p, desc in (self.params or {}).items():
            parts.append("%s: %s" % (split_identifier(p), desc))
        return ". ".join(x for x in parts if x)


def split_identifier(s: str) -> str:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    return re.sub(r"[_\-.]+", " ", s).strip().lower()


def load_catalog(path: str | Path) -> list[Tool]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    tools = [Tool(server=t["server"], name=t["name"],
                  description=t.get("description", ""), params=t.get("params") or {})
             for t in raw]
    names = [t.name for t in tools]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ValueError("tool names must be unique; duplicated: %s" % sorted(dupes))
    return tools
