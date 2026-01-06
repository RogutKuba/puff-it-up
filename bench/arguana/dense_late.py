"""
Hybrid retrieval benchmark: Dense first-stage + Late interaction reranking.
Uses OpenAI text-embedding-3-small for candidate retrieval, Jina ColBERT v2 for MaxSim reranking.
Reuses existing indices from dense.py and late.py.
"""

import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from io import StringIO
from typing import Any

from dotenv import load_dotenv

load_dotenv()

import httpx
import numpy as np
from openai import OpenAI
from tqdm import tqdm
import turbopuffer
from beir import util
from beir.datasets.data_loader import GenericDataLoader


# Configuration
DATASET_NAME = "arguana"
DENSE_NAMESPACE = "arguana-dense-bench-openai"
LATE_NAMESPACE = "arguana-late-bench"

# Models
DENSE_MODEL = "openai/text-embedding-3-small"
COLBERT_MODEL = "jina-colbert-v2"
COLBERT_DIM = 128

# Retrieval parameters
FIRST_STAGE_TOP_K = 100  # Candidates from dense retrieval
FINAL_TOP_K = 10
MAX_CONCURRENT_QUERIES = 16

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


class TeeOutput:
    """Capture output while still printing to console."""

    def __init__(self):
        self.buffer = StringIO()
        self.console = sys.stdout

    def write(self, text):
        self.buffer.write(text)
        self.console.write(text)

    def flush(self):
        self.console.flush()

    def getvalue(self):
        return self.buffer.getvalue()


def get_next_result_filename() -> str:
    """Get the next available result filename (arguana_dense_late_N.txt)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    existing = os.listdir(RESULTS_DIR)
    max_num = 0
    for fname in existing:
        match = re.match(r"arguana_dense_late_(\d+)\.txt", fname)
        if match:
            max_num = max(max_num, int(match.group(1)))
    return os.path.join(RESULTS_DIR, f"arguana_dense_late_{max_num + 1}.txt")


def get_openrouter_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")


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


def embed_dense(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Get dense embeddings from OpenAI."""
    response = client.embeddings.create(model=DENSE_MODEL, input=texts)
    return [item.embedding for item in response.data]


def embed_colbert(
    client: httpx.Client,
    texts: list[str],
    input_type: str = "query",
) -> list[list[list[float]]]:
    """Get multi-vector embeddings from Jina ColBERT v2."""
    response = client.post(
        "/v1/multi-vector",
        json={
            "model": COLBERT_MODEL,
            "dimensions": COLBERT_DIM,
            "input_type": input_type,
            "embedding_type": "float",
            "input": texts,
        },
    )
    response.raise_for_status()
    data = response.json()
    return [item["embeddings"] for item in data["data"]]


def load_arguana_dataset() -> tuple[dict, dict, dict]:
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{DATASET_NAME}.zip"
    data_path = util.download_and_unzip(url, "datasets")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    return corpus, queries, qrels


def first_stage_dense(
    ns: Any,
    query_embedding: list[float],
) -> list[str]:
    """First stage: Get candidate doc_ids via dense ANN search."""
    response = ns.query(
        rank_by=("vector", "ANN", query_embedding),
        top_k=FIRST_STAGE_TOP_K,
    )
    return [row.id for row in response.rows]


def second_stage_rerank(
    ns: Any,
    query_embeddings: list[list[float]],
    candidate_docs: list[str],
) -> list[str]:
    """
    Second stage: Re-rank candidates using ColBERT MaxSim.
    Queries the late interaction index filtered to candidate docs.
    """
    if not candidate_docs:
        return []

    doc_scores: dict[str, float] = defaultdict(float)

    # Process query tokens in batches
    for i in range(0, len(query_embeddings), MAX_CONCURRENT_QUERIES):
        batch_embeddings = query_embeddings[i : i + MAX_CONCURRENT_QUERIES]

        queries = [
            {
                "rank_by": ["vector", "ANN", emb],
                "top_k": 256,  # Get enough tokens per candidate
                "include_attributes": ["doc_id"],
                "filters": ["doc_id", "In", candidate_docs],
            }
            for emb in batch_embeddings
        ]

        response = ns.multi_query(queries=queries)

        # Compute MaxSim: for each query token, find max similarity per doc
        for result in response.results:
            doc_max: dict[str, float] = {}
            for row in result.rows:
                doc_id = row.model_extra.get("doc_id") or row.get("doc_id")
                if doc_id:
                    similarity = 1.0 - row["$dist"]
                    if doc_id not in doc_max or similarity > doc_max[doc_id]:
                        doc_max[doc_id] = similarity

            # Add max scores to total
            for doc_id, max_score in doc_max.items():
                doc_scores[doc_id] += max_score

    # Sort by score descending and return top-k
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_docs[:FINAL_TOP_K]]


def search_query(
    dense_ns: Any,
    late_ns: Any,
    dense_embedding: list[float],
    colbert_embeddings: list[list[float]],
) -> tuple[list[str], float, float]:
    """
    Two-stage hybrid search for a single query.
    Returns (top-k doc_ids, first_stage_latency_ms, second_stage_latency_ms).
    """
    # Stage 1: Dense retrieval
    start = time.perf_counter()
    candidates = first_stage_dense(dense_ns, dense_embedding)
    first_stage_ms = (time.perf_counter() - start) * 1000

    # Stage 2: ColBERT rerank
    start = time.perf_counter()
    results = second_stage_rerank(late_ns, colbert_embeddings, candidates)
    second_stage_ms = (time.perf_counter() - start) * 1000

    return results, first_stage_ms, second_stage_ms


def search_queries(
    dense_ns: Any,
    late_ns: Any,
    queries: dict,
    openrouter_client: OpenAI,
    jina_client: httpx.Client,
) -> tuple[dict[str, list[str]], list[float], list[float]]:
    """Search all queries. Returns (results, first_stage_latencies, second_stage_latencies)."""
    print(f"Searching {len(queries)} queries (dense + late interaction rerank)...")

    results: dict[str, list[str]] = {}
    first_latencies: list[float] = []
    second_latencies: list[float] = []

    for qid, query_text in tqdm(queries.items(), desc="Searching"):
        # Get embeddings from both models
        dense_embedding = embed_dense(openrouter_client, [query_text])[0]
        colbert_embeddings = embed_colbert(jina_client, [query_text], input_type="query")[0]

        # Two-stage search
        doc_ids, first_ms, second_ms = search_query(
            dense_ns, late_ns, dense_embedding, colbert_embeddings
        )

        results[qid] = doc_ids
        first_latencies.append(first_ms)
        second_latencies.append(second_ms)

    return results, first_latencies, second_latencies


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
    # Set up output capture
    tee = TeeOutput()
    sys.stdout = tee

    print("=" * 60)
    print("Hybrid Benchmark: Dense + Late Interaction Rerank")
    print("ArguAna Dataset")
    print("=" * 60)

    # Print configuration
    print("\nConfiguration:")
    print(f"  Dataset:              {DATASET_NAME}")
    print(f"  Dense Namespace:      {DENSE_NAMESPACE}")
    print(f"  Late Namespace:       {LATE_NAMESPACE}")
    print(f"  Dense Model:          {DENSE_MODEL}")
    print(f"  ColBERT Model:        {COLBERT_MODEL}")
    print(f"  ColBERT Dim:          {COLBERT_DIM}")
    print(f"  First Stage Top K:    {FIRST_STAGE_TOP_K}")
    print(f"  Final Top K:          {FINAL_TOP_K}")
    print(f"  Timestamp:            {datetime.now().isoformat()}")

    # Initialize clients
    openrouter_client = get_openrouter_client()
    jina_client = get_jina_client()
    tpuf_client = get_turbopuffer_client()

    dense_ns = tpuf_client.namespace(DENSE_NAMESPACE)
    late_ns = tpuf_client.namespace(LATE_NAMESPACE)

    # Load dataset
    print("\nLoading ArguAna dataset...")
    corpus, queries, qrels = load_arguana_dataset()
    print(f"  Corpus:  {len(corpus)} documents")
    print(f"  Queries: {len(queries)}")
    print(f"  QRels:   {len(qrels)} query-relevance pairs")

    print(f"\nUsing existing indices:")
    print(f"  Dense namespace: {DENSE_NAMESPACE}")
    print(f"  Late namespace:  {LATE_NAMESPACE}")

    # Search queries
    print("\n" + "-" * 40)
    results, first_latencies, second_latencies = search_queries(
        dense_ns, late_ns, queries, openrouter_client, jina_client
    )

    total_latencies = [f + s for f, s in zip(first_latencies, second_latencies)]

    # Latency stats
    print(f"\nLatency Statistics (search only, excludes embedding):")
    print(f"\n  First Stage (Dense ANN):")
    print(f"    Average: {np.mean(first_latencies):.2f} ms")
    print(f"    P50:     {np.percentile(first_latencies, 50):.2f} ms")
    print(f"    P95:     {np.percentile(first_latencies, 95):.2f} ms")

    print(f"\n  Second Stage (ColBERT Rerank):")
    print(f"    Average: {np.mean(second_latencies):.2f} ms")
    print(f"    P50:     {np.percentile(second_latencies, 50):.2f} ms")
    print(f"    P95:     {np.percentile(second_latencies, 95):.2f} ms")

    print(f"\n  Total:")
    print(f"    Average: {np.mean(total_latencies):.2f} ms")
    print(f"    P50:     {np.percentile(total_latencies, 50):.2f} ms")
    print(f"    P95:     {np.percentile(total_latencies, 95):.2f} ms")
    print(f"    P99:     {np.percentile(total_latencies, 99):.2f} ms")

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

    # Save results to file
    sys.stdout = tee.console  # Restore stdout before saving
    result_file = get_next_result_filename()
    with open(result_file, "w") as f:
        f.write(tee.buffer.getvalue())
    print(f"\nResults saved to: {result_file}")

    return {
        "ndcg@10": ndcg_10,
        "recall@10": recall_10,
        "latency_first_avg_ms": np.mean(first_latencies),
        "latency_second_avg_ms": np.mean(second_latencies),
        "latency_total_avg_ms": np.mean(total_latencies),
        "latency_total_p50_ms": np.percentile(total_latencies, 50),
        "latency_total_p95_ms": np.percentile(total_latencies, 95),
        "latency_total_p99_ms": np.percentile(total_latencies, 99),
    }


if __name__ == "__main__":
    main()
