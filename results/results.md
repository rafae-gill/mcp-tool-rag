# Tool search results: `data`

158 tools on 8 servers · 200 requests (27 where the best tool is forbidden for the caller) · k = 5

| | bm25 | embeddings | hybrid | bm25+ex | emb+ex | emb+ex-concat | hybrid+ex |
|---|---|---|---|---|---|---|---|
| Right tool ranked 1st | 24% | 34% | 31% | 67% | 59% | 47% | 73% |
| Right tool in top 3 | 36% | 52% | 40% | 82% | 83% | 68% | 86% |
| Right tool in top 5 | 41% | 60% | 51% | 89% | 89% | 75% | 92% |
| Right tool in top 10 | 52% | 73% | 65% | 94% | 94% | 83% | 97% |
| All needed tools in top 5 | 36% | 52% | 43% | 80% | 81% | 68% | 84% |
| MRR (top 10) | 0.32 | 0.45 | 0.39 | 0.76 | 0.72 | 0.59 | 0.80 |
| Top 5, permissions bolted on after | 35% | 51% | 42% | 83% | 83% | 65% | 86% |
| Permission-unaware top 5 shows a forbidden tool | 88% | 76% | 79% | 76% | 64% | 74% | 63% |
| …and it's the forbidden tool the request wanted | 48% | 67% | 59% | 93% | 74% | 74% | 96% |
| Best tool forbidden: a helpful allowed tool in top 5 | 38% | 38% | 31% | 69% | 62% | 56% | 75% |
| Median ms per search | 0.1 | 5.2 | 5.2 | 0.1 | 6.3 | 5.2 | 6.3 |

**Right tool in top 3, by difficulty**

| Difficulty | n | bm25 | embeddings | hybrid | bm25+ex | emb+ex | emb+ex-concat | hybrid+ex |
|---|---|---|---|---|---|---|---|---|
| easy | 48 | 48% | 54% | 50% | 88% | 88% | 73% | 92% |
| paraphrase | 47 | 32% | 49% | 32% | 89% | 83% | 68% | 87% |
| ambiguous | 39 | 18% | 38% | 26% | 64% | 64% | 54% | 72% |
| multi | 39 | 46% | 67% | 54% | 82% | 97% | 74% | 90% |

**Tool definitions an agent loads (estimated tokens: JSON chars / 4)**

| Role | Tools allowed | All allowed tools | Top 5 |
|---|---|---|---|
| analyst | 93 | ~4634 | ~249 |
| finance | 80 | ~4042 | ~253 |
| marketing | 80 | ~3878 | ~242 |
| engineer | 93 | ~4492 | ~242 |
| support | 54 | ~2607 | ~241 |
| intern | 32 | ~1564 | ~244 |
