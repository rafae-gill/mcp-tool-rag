# Real requests from a team (private)

Everything in this folder except this README and the template is **git-ignored**:
the scores get published, the data doesn't, unless you decide otherwise.

## 1. Export your servers' tool lists

Once per server, with a key that can see all its tools:

```bash
python eval/export_catalog.py --url https://<server>/mcp --key sk_... --server warehouse --out data/real/catalog.json
```

Then open `catalog.json` and anonymize anything that identifies the company: a
client or product name in a tool's name or description becomes something
generic. Keep the meaning, because the meaning is what's being searched.

## 2. Write down 30–50 real requests

Copy `queries_TEMPLATE.csv` to `queries.csv` and fill one row per request:

| column | what goes in it |
|---|---|
| `query` | what the person actually typed to their agent, as close to word-for-word as you can. Remove names, client names, numbers and anything private, but keep the phrasing: informal, jargon, typos and all. Don't tidy it up; the messiness is the test. |
| `gold` | the tool(s) that were the right first call, using the names from `catalog.json`; separate several with `;` |
| `role` | optional: who asked (analyst, finance, ...). Leave empty if you don't have roles. |
| `forbidden_best` | optional: `yes` if the best tool is one this person isn't allowed to use |
| `allowed_fallback` | optional: for those, a tool they ARE allowed that still helps, `;`-separated |

The best sources are, in order: requests you remember people making, requests
you can reconstruct from the audit log (it shows which tool was called, and
you'll remember roughly what was asked), and requests colleagues send you if
you ask them. **Don't write them by looking at the tool list.** That's exactly
the bias real requests are meant to remove.

## 3. Roles (optional)

If `queries.csv` uses roles, add `roles.json` in mcp-gatekeeper's permission
shape (`{"analyst": {"warehouse": {"tools": [...]}}}`). Without it, every
request is scored as if the caller may use every tool.

## 4. Run it

```bash
python eval/run_eval.py --data data/real --out results/real
```
