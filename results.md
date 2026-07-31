# Evaluation Results

Run: `uv run python eval/evaluate.py` — 50 ground truth pairs, 3 LLM test questions.

## Retrieval Metrics

| Method | Hit Rate | MRR |
|--------|----------|-----|
| keyword (default) | 0.600 | 0.431 |
| vector (all-MiniLM-L6-v2) | 0.420 | 0.314 |
| hybrid (RRF k=60) | 0.660 | 0.419 |
| keyword (optimized boost) | **0.820** | **0.525** |

Optimal boost weights found: `title=0.78`, `section=1.07`, `content=2.66`

## LLM Evaluation

| Model | % RELEVANT | % PARTLY | % NON | Cost per 1K queries |
|-------|-----------|----------|-------|-------------------|
| gpt-4o-mini | 100.00 | 0.00 | 0.00 | $0.39 |
| gpt-4o | 66.67 | 33.33 | 0.00 | $4.60 |

## Key Findings

1. **Boost optimization is the biggest win.** Optimized keyword (0.820 HR) beats default keyword (0.600) by +37%. Content field weight (2.66) matters ~3x more than title (0.78).

2. **Vector search underperforms keyword** (0.420 vs 0.600). `all-MiniLM-L6-v2` (384-dim) may be too small. A larger model like `all-mpnet-base-v2` (768d) might close the gap.

3. **Hybrid (0.660) beats both individual methods** but loses to keyword_opt (0.820). Embeddings aren't adding signal on top of optimized keyword — RRF fusion barely improves because vector results are noisy.

4. **gpt-4o-mini scoring 100% vs gpt-4o 66.67% is suspect at n=3.** Likely LLM-as-judge bias (judge favors its own outputs). Results are noise at this sample size — need 50+ questions and a held-out judge model.

## Recommendations

- **Minimum 50 ground truth pairs** before drawing conclusions
- **Stronger embedding model** (`all-mpnet-base-v2` or `BAAI/bge-small-en-v1.5`)
- **Use gpt-4o as judge** for both models to remove self-preference bias
