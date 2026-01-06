"""
Late interaction with two-stage retrieval benchmark on SciFact dataset.
Uses Jina ColBERT v2 for multi-vector embeddings and turbopuffer for storage/search.
Reuses existing index from late.py.

Two-stage approach:
1. First stage: Get candidate doc_ids via ANN search (union of top-k per query token)
2. Second stage: Re-rank candidates using filtered search + MaxSim scoring
"""

import os
import time
from collections import defaultdict
from typing import Any

from dotenv import load_dotenv

load_dotenv()

import httpx
import numpy as np
from tqdm import tqdm
import turbopuffer
from beir import util
from beir.datasets.data_loader import GenericDataLoader


# Configuration
DATASET_NAME = "scifact"
NAMESPACE = "scifact-late-bench"  # Reuse existing index
JINA_MODEL = "jina-colbert-v2"
EMBEDDING_DIM = 128
TOP_K = 10
MAX_CONCURRENT_QUERIES = 16  # Turbopuffer limit

# Two-stage parameters
FIRST_STAGE_TOP_K = 10  # Top-k per query token in first stage
SECOND_STAGE_TOP_K = 256  # Top-k per query token in second stage (within filtered set)


def get_jina_client() -> httpx.Client:
    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        raise ValueError("JINA_API_KEY environment variable not set")
    return httpx.Client(
        base_url="https://api.jina.ai",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=120.0,
    )


def get_turbopuffer_client() -> turbopuffer.Turbopuffer:
    api_key = os.getenv("TURBOPUFFER_API_KEY")
    if not api_key:
        raise ValueError("TURBOPUFFER_API_KEY environment variable not set")
    return turbopuffer.Turbopuffer(api_key=api_key, region="gcp-us-central1")


def get_multi_vector_embeddings(
    client: httpx.Client,
    texts: list[str],
    input_type: str = "document",
) -> list[list[list[float]]]:
    """
    Get multi-vector embeddings from Jina ColBERT v2.
    Returns list of documents, each containing list of token embeddings.
    """
    response = client.post(
        "/v1/multi-vector",
        json={
            "model": JINA_MODEL,
            "dimensions": EMBEDDING_DIM,
            "input_type": input_type,
            "embedding_type": "float",
            "input": texts,
        },
    )
    response.raise_for_status()
    data = response.json()
    return [item["embeddings"] for item in data["data"]]


def load_scifact_dataset() -> tuple[dict, dict, dict]:
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{DATASET_NAME}.zip"
    data_path = util.download_and_unzip(url, "datasets")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    return corpus, queries, qrels


def first_stage_retrieval(
    ns: Any,
    query_embeddings: list[list[float]],
) -> set[str]:
    """
    First stage: Get candidate doc_ids via ANN search.
    Returns union of top-k docs per query token.
    """
    candidate_docs: set[str] = set()

    # Process query tokens in batches of MAX_CONCURRENT_QUERIES
    for i in range(0, len(query_embeddings), MAX_CONCURRENT_QUERIES):
        batch_embeddings = query_embeddings[i : i + MAX_CONCURRENT_QUERIES]

        queries = [
            {
                "rank_by": ["vector", "ANN", emb],
                "top_k": FIRST_STAGE_TOP_K,
                "include_attributes": ["doc_id"],
            }
            for emb in batch_embeddings
        ]

        response = ns.multi_query(queries=queries)

        for result in response.results:
            for row in result.rows:
                doc_id = row.model_extra.get("doc_id")
                if doc_id:
                    candidate_docs.add(doc_id)

    return candidate_docs


def second_stage_rerank(
    ns: Any,
    query_embeddings: list[list[float]],
    candidate_docs: set[str],
) -> list[str]:
    """
    Second stage: Re-rank candidate docs using filtered search + MaxSim.
    Returns top-k doc_ids sorted by MaxSim score.
    """
    if not candidate_docs:
        return []

    doc_scores: dict[str, float] = defaultdict(float)
    candidate_list = list(candidate_docs)

    # Process query tokens in batches of MAX_CONCURRENT_QUERIES
    for i in range(0, len(query_embeddings), MAX_CONCURRENT_QUERIES):
        batch_embeddings = query_embeddings[i : i + MAX_CONCURRENT_QUERIES]

        queries = [
            {
                "rank_by": ["vector", "ANN", emb],
                "top_k": SECOND_STAGE_TOP_K,
                "include_attributes": ["doc_id"],
                "filters": ["doc_id", "In", candidate_list],
            }
            for emb in batch_embeddings
        ]

        response = ns.multi_query(queries=queries)

        # Process results for each query token
        for result in response.results:
            # Track max similarity per doc for this query token
            seen_docs: set[str] = set()
            for row in result.rows:
                doc_id = row.model_extra.get("doc_id")
                if doc_id and doc_id not in seen_docs:
                    seen_docs.add(doc_id)
                    # Convert distance to similarity and add to MaxSim score
                    similarity = 1.0 - row["$dist"]
                    doc_scores[doc_id] += similarity

    # Sort by score descending and return top-k
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_docs[:TOP_K]]


def search_query(
    ns: Any,
    query_embeddings: list[list[float]],
) -> tuple[list[str], int]:
    """
    Two-stage search for a single query.
    Returns (top-k doc_ids, num_candidates from first stage).
    """
    # Stage 1: Get candidates
    candidate_docs = first_stage_retrieval(ns, query_embeddings)

    # Stage 2: Re-rank candidates
    results = second_stage_rerank(ns, query_embeddings, candidate_docs)

    return results, len(candidate_docs)


def search_queries(
    ns: Any,
    queries: dict,
    jina_client: httpx.Client,
) -> tuple[dict[str, list[str]], list[float], list[int]]:
    """Search all queries. Returns (results, latencies, candidate_counts)."""
    print(f"Searching {len(queries)} queries (two-stage late interaction)...")

    results: dict[str, list[str]] = {}
    latencies: list[float] = []
    candidate_counts: list[int] = []

    for qid, query_text in tqdm(queries.items(), desc="Searching"):
        # Get query multi-vector embeddings
        query_embeddings = get_multi_vector_embeddings(
            jina_client, [query_text], input_type="query"
        )[0]

        start = time.perf_counter()
        doc_ids, num_candidates = search_query(ns, query_embeddings)
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)
        candidate_counts.append(num_candidates)

        results[qid] = doc_ids

    return results, latencies, candidate_counts


def calculate_ndcg(
    results: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    k: int = 10,
) -> float:
    """Calculate NDCG@k."""
    ndcg_scores = []

    for qid, doc_ids in results.items():
        if qid not in qrels:
            continue

        relevances = qrels[qid]

        # DCG
        dcg = 0.0
        for i, doc_id in enumerate(doc_ids[:k]):
            rel = relevances.get(doc_id, 0)
            dcg += rel / np.log2(i + 2)

        # Ideal DCG
        ideal_rels = sorted(relevances.values(), reverse=True)[:k]
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))

        if idcg > 0:
            ndcg_scores.append(dcg / idcg)
        else:
            ndcg_scores.append(0.0)

    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def calculate_recall(
    results: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    k: int = 10,
) -> float:
    """Calculate Recall@k."""
    recall_scores = []

    for qid, doc_ids in results.items():
        if qid not in qrels:
            continue

        relevances = qrels[qid]
        relevant_docs = {doc_id for doc_id, rel in relevances.items() if rel > 0}

        if not relevant_docs:
            continue

        retrieved_relevant = len(set(doc_ids[:k]) & relevant_docs)
        recall_scores.append(retrieved_relevant / len(relevant_docs))

    return np.mean(recall_scores) if recall_scores else 0.0


def main():
    print("=" * 60)
    print("Late Interaction (Two-Stage) Benchmark - SciFact Dataset")
    print("=" * 60)

    # Initialize clients
    jina_client = get_jina_client()
    tpuf_client = get_turbopuffer_client()
    ns = tpuf_client.namespace(NAMESPACE)

    # Load dataset
    print("\nLoading SciFact dataset...")
    corpus, queries, qrels = load_scifact_dataset()
    print(f"  Corpus:  {len(corpus)} documents")
    print(f"  Queries: {len(queries)}")
    print(f"  QRels:   {len(qrels)} query-relevance pairs")

    print(f"\nUsing existing index in namespace: {NAMESPACE}")
    print(f"  First stage top-k per token: {FIRST_STAGE_TOP_K}")
    print(f"  Second stage top-k per token: {SECOND_STAGE_TOP_K}")

    # Search queries
    print("\n" + "-" * 40)
    results, latencies, candidate_counts = search_queries(ns, queries, jina_client)

    # Latency stats
    print(f"\nLatency Statistics (search only, excludes embedding):")
    print(f"  Average: {np.mean(latencies):.2f} ms")
    print(f"  P50:     {np.percentile(latencies, 50):.2f} ms")
    print(f"  P95:     {np.percentile(latencies, 95):.2f} ms")
    print(f"  P99:     {np.percentile(latencies, 99):.2f} ms")

    print(f"\nCandidate Statistics (first stage):")
    print(f"  Average: {np.mean(candidate_counts):.1f} docs")
    print(f"  Min:     {np.min(candidate_counts)} docs")
    print(f"  Max:     {np.max(candidate_counts)} docs")

    # Calculate metrics
    print("\n" + "-" * 40)
    print("Evaluation Metrics:")

    ndcg_10 = calculate_ndcg(results, qrels, k=10)
    recall_10 = calculate_recall(results, qrels, k=10)

    print(f"  NDCG@10:   {ndcg_10:.4f}")
    print(f"  Recall@10: {recall_10:.4f}")

    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)

    return {
        "ndcg@10": ndcg_10,
        "recall@10": recall_10,
        "latency_avg_ms": np.mean(latencies),
        "latency_p50_ms": np.percentile(latencies, 50),
        "latency_p95_ms": np.percentile(latencies, 95),
        "latency_p99_ms": np.percentile(latencies, 99),
        "avg_candidates": np.mean(candidate_counts),
    }


if __name__ == "__main__":
    main()
