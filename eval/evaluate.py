"""Epic 4: Full evaluation pipeline — ground truth, retrieval eval, LLM eval."""

import json, os, sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
from minsearch import Index
from app.search import keyword_search, vector_search, hybrid_search
from app.evaluation import evaluate_retrieval, optimize_boosts, compare_models, comparison_summary
from notebooks.ground_truth import generate_ground_truth

CHUNKS_PATH = "data/processed/fastapi/chunks.json"
EMBEDDINGS_PATH = "data/processed/fastapi/embeddings.npy"
GT_PATH = "data/ground_truth.jsonl"

def step1_generate_ground_truth(num_chunks=50):
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks, generating ground truth for {num_chunks}...")
    gt = generate_ground_truth(chunks[:num_chunks])
    with open(GT_PATH, "w", encoding="utf-8") as f:
        for item in gt:
            f.write(json.dumps(item) + "\n")
    print(f"  -> {len(gt)} pairs saved to {GT_PATH}")
    return gt

def step2_retrieval_evaluation():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    embeddings = np.load(EMBEDDINGS_PATH)
    with open(GT_PATH, encoding="utf-8") as f:
        gt = [json.loads(l) for l in f]
    print(f"Loaded {len(gt)} ground truth pairs")

    index = Index(text_fields=["title", "section", "content"], keyword_fields=["id", "doc_library"])
    index.fit(chunks)
    print("Index built")

    print("\n--- Retrieval Metrics ---")
    methods = {
        "keyword": lambda q: keyword_search(q, index=index),
        "vector": lambda q: vector_search(q, embeddings=embeddings, chunks=chunks),
        "hybrid": lambda q: hybrid_search(q, method="hybrid", index=index, embeddings=embeddings, chunks=chunks),
    }
    results = {}
    for name, fn in methods.items():
        m = evaluate_retrieval(gt, fn)
        results[name] = m
        print(f"  {name:>10}:  HR={m['hit_rate']:.3f}  MRR={m['mrr']:.3f}")

    best = optimize_boosts(index, gt, iterations=30)
    opt = evaluate_retrieval(gt, lambda q: index.search(q, boost_dict=best))
    results["keyword_opt"] = opt
    print(f"  keyword_opt:  HR={opt['hit_rate']:.3f}  MRR={opt['mrr']:.3f}")
    print(f"  best boosts: {best}")

    return results

def step3_llm_evaluation():
    questions = [
        "How do I create a path operation?",
        "What is a dependency in FastAPI?",
        "How do I handle errors?",
    ]
    print("\n--- LLM Evaluation ---")
    results = compare_models(questions, models=["gpt-4o-mini", "gpt-4o"])
    summary = comparison_summary(results)
    print(summary.to_string(float_format="%.2f"))
    return summary

if __name__ == "__main__":
    if not os.path.exists(GT_PATH):
        if not os.path.exists(CHUNKS_PATH):
            print("Run ingestion first: uv run python -m ingest.run --library fastapi")
            sys.exit(1)
        step1_generate_ground_truth()
    else:
        print(f"Ground truth exists at {GT_PATH}, skipping generation")

    step2_retrieval_evaluation()
    step3_llm_evaluation()
