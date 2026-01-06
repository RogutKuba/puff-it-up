"""
Late interaction (ColBERT-style) benchmark on SciFact dataset.
Uses Jina ColBERT v2 for multi-vector embeddings and turbopuffer for storage/search.
Implements client-side MaxSim scoring.
"""

import json
import os
import time
from collections import defaultdict
from pathlib import Path
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
NAMESPACE = "scifact-late-bench"
JINA_MODEL = "jina-colbert-v2"
EMBEDDING_DIM = 128
BATCH_SIZE = 8  # Smaller batch for multi-vector (more data per doc)
TOP_K = 10
MAX_CONCURRENT_QUERIES = 16  # Turbopuffer limit

# Checkpoint file for resumable indexing
CHECKPOINT_FILE = Path(__file__).parent / ".late_index_checkpoint.json"


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


def load_checkpoint() -> set[str]:
    """Load set of already-indexed doc IDs from checkpoint file."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            data = json.load(f)
            return set(data.get("indexed_docs", []))
    return set()


def save_checkpoint(indexed_docs: set[str]) -> None:
    """Save indexed doc IDs to checkpoint file."""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"indexed_docs": list(indexed_docs)}, f)


def clear_checkpoint() -> None:
    """Remove checkpoint file after successful completion."""
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()


def index_corpus(ns: Any, corpus: dict, jina_client: httpx.Client) -> float:
    """
    Index corpus with multi-vector embeddings.
    Each token vector is stored as a separate row with doc_id reference.
    Supports resuming from checkpoint if interrupted.
    """
    # Load checkpoint to skip already-indexed docs
    indexed_docs = load_checkpoint()
    if indexed_docs:
        print(f"Resuming from checkpoint: {len(indexed_docs)} docs already indexed")

    doc_ids = [d for d in corpus.keys() if d not in indexed_docs]
    print(f"Indexing {len(doc_ids)} documents (multi-vector)...")

    if not doc_ids:
        print("  All documents already indexed!")
        return 0.0

    total_time = 0.0
    total_vectors = 0

    for i in tqdm(range(0, len(doc_ids), BATCH_SIZE), desc="Indexing"):
        batch_ids = doc_ids[i : i + BATCH_SIZE]
        batch_texts = [
            f"{corpus[doc_id].get('title', '')} {corpus[doc_id]['text']}".strip()
            for doc_id in batch_ids
        ]

        # Get multi-vector embeddings
        multi_embeddings = get_multi_vector_embeddings(
            jina_client, batch_texts, input_type="document"
        )

        # Flatten: create one row per token vector
        rows = []
        for doc_id, token_embeddings in zip(batch_ids, multi_embeddings):
            for token_idx, embedding in enumerate(token_embeddings):
                rows.append({
                    "id": f"{doc_id}_{token_idx}",
                    "vector": embedding,
                    "doc_id": doc_id,
                    "token_idx": token_idx,
                })
            total_vectors += len(token_embeddings)

        # Upsert to turbopuffer
        start = time.perf_counter()
        ns.write(upsert_rows=rows, distance_metric="cosine_distance")
        total_time += time.perf_counter() - start

        # Save checkpoint after each batch
        indexed_docs.update(batch_ids)
        save_checkpoint(indexed_docs)

    print(f"  Total token vectors indexed: {total_vectors}")
    return total_time


def maxsim_score(
    query_token_results: list[dict[str, float]],
) -> dict[str, float]:
    """
    Compute MaxSim scores for documents.
    For each document, sum the max similarity across all query tokens.

    query_token_results: list of dicts, one per query token
        Each dict maps doc_id -> similarity score
    """
    doc_scores: dict[str, float] = defaultdict(float)

    for token_results in query_token_results:
        # Group by doc_id and take max per doc for this query token
        doc_max: dict[str, float] = defaultdict(lambda: float("-inf"))
        for doc_id, score in token_results.items():
            doc_max[doc_id] = max(doc_max[doc_id], score)

        # Add max scores to total
        for doc_id, max_score in doc_max.items():
            doc_scores[doc_id] += max_score

    return dict(doc_scores)


def search_query(
    ns: Any,
    query_embeddings: list[list[float]],
    top_k_per_token: int = 100,
) -> list[str]:
    """
    Search using multi-query for all query tokens, then compute MaxSim.
    Returns top-k document IDs sorted by MaxSim score.
    """
    query_token_results: list[dict[str, float]] = []

    # Process query tokens in batches of MAX_CONCURRENT_QUERIES
    for i in range(0, len(query_embeddings), MAX_CONCURRENT_QUERIES):
        batch_embeddings = query_embeddings[i : i + MAX_CONCURRENT_QUERIES]

        # Build multi-query
        queries = [
            {
                "rank_by": ("vector", "ANN", emb),
                "top_k": top_k_per_token,
                "include_attributes": ["doc_id"],
            }
            for emb in batch_embeddings
        ]

        # Execute multi-query
        response = ns.multi_query(queries=queries)

        # Process results for each query token
        for result in response.results:
            token_docs: dict[str, float] = {}
            for row in result.rows:
                doc_id = row.model_extra.get("doc_id")
                # Convert distance to similarity (cosine_distance = 1 - similarity)
                similarity = 1.0 - row["$dist"]
                if doc_id:
                    token_docs[doc_id] = max(token_docs.get(doc_id, float("-inf")), similarity)
            query_token_results.append(token_docs)

    # Compute MaxSim scores
    doc_scores = maxsim_score(query_token_results)

    # Sort by score descending and return top-k doc IDs
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_docs[:TOP_K]]


def search_queries(
    ns: Any,
    queries: dict,
    jina_client: httpx.Client,
) -> tuple[dict[str, list[str]], list[float]]:
    """Search all queries. Returns (results, latencies)."""
    print(f"Searching {len(queries)} queries (late interaction)...")

    results: dict[str, list[str]] = {}
    latencies: list[float] = []

    for qid, query_text in tqdm(queries.items(), desc="Searching"):
        # Get query multi-vector embeddings
        query_embeddings = get_multi_vector_embeddings(
            jina_client, [query_text], input_type="query"
        )[0]

        start = time.perf_counter()
        doc_ids = search_query(ns, query_embeddings)
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)

        results[qid] = doc_ids

    return results, latencies


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
    print("Late Interaction Benchmark - SciFact Dataset")
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

    # Index corpus
    print("\n" + "-" * 40)
    index_time = index_corpus(ns, corpus, jina_client)
    print(f"Total indexing time: {index_time:.2f}s")

    # Search queries
    print("\n" + "-" * 40)
    results, latencies = search_queries(ns, queries, jina_client)

    # Latency stats
    print(f"\nLatency Statistics (search only, excludes embedding):")
    print(f"  Average: {np.mean(latencies):.2f} ms")
    print(f"  P50:     {np.percentile(latencies, 50):.2f} ms")
    print(f"  P95:     {np.percentile(latencies, 95):.2f} ms")
    print(f"  P99:     {np.percentile(latencies, 99):.2f} ms")

    # Calculate metrics
    print("\n" + "-" * 40)
    print("Evaluation Metrics:")

    ndcg_10 = calculate_ndcg(results, qrels, k=10)
    recall_10 = calculate_recall(results, qrels, k=10)

    print(f"  NDCG@10:   {ndcg_10:.4f}")
    print(f"  Recall@10: {recall_10:.4f}")

    # Clear checkpoint on successful completion
    clear_checkpoint()

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
        "index_time_s": index_time,
    }


if __name__ == "__main__":
    main()
