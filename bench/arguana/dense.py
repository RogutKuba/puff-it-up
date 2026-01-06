"""
Dense vector search benchmark on ArguAna dataset using turbopuffer and OpenAI embeddings.
Tracks NDCG@10, Recall@10, and latency metrics.
"""

import os
import re
import sys
import time
from datetime import datetime
from io import StringIO
from typing import Any

from dotenv import load_dotenv

load_dotenv()

import numpy as np
from openai import OpenAI
from tqdm import tqdm
import turbopuffer
from beir import util
from beir.datasets.data_loader import GenericDataLoader


# Configuration
DATASET_NAME = "arguana"
NAMESPACE = "arguana-dense-bench-openai"
EMBEDDING_MODEL = "openai/text-embedding-3-small"
BATCH_SIZE = 32
TOP_K = 10

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
    """Get the next available result filename (arguana_dense_N.txt)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    existing = os.listdir(RESULTS_DIR)
    max_num = 0
    for fname in existing:
        match = re.match(r"arguana_dense_(\d+)\.txt", fname)
        if match:
            max_num = max(max_num, int(match.group(1)))
    return os.path.join(RESULTS_DIR, f"arguana_dense_{max_num + 1}.txt")


def get_openrouter_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")


def get_turbopuffer_client() -> turbopuffer.Turbopuffer:
    api_key = os.getenv("TURBOPUFFER_API_KEY")
    if not api_key:
        raise ValueError("TURBOPUFFER_API_KEY environment variable not set")
    return turbopuffer.Turbopuffer(api_key=api_key, region="gcp-us-central1")


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def load_arguana_dataset() -> tuple[dict, dict, dict]:
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{DATASET_NAME}.zip"
    data_path = util.download_and_unzip(url, "datasets")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    return corpus, queries, qrels


def index_corpus(ns: Any, corpus: dict, embed_client: OpenAI) -> float:
    """Index corpus documents into turbopuffer. Returns total indexing time."""
    print(f"Indexing {len(corpus)} documents...")

    doc_ids = list(corpus.keys())
    total_time = 0.0

    for i in tqdm(range(0, len(doc_ids), BATCH_SIZE), desc="Indexing"):
        batch_ids = doc_ids[i : i + BATCH_SIZE]
        batch_texts = [
            f"{corpus[doc_id].get('title', '')} {corpus[doc_id]['text']}".strip()
            for doc_id in batch_ids
        ]

        embeddings = embed_texts(embed_client, batch_texts)

        rows = [
            {
                "id": doc_id,
                "vector": embedding,
            }
            for doc_id, embedding in zip(batch_ids, embeddings)
        ]

        start = time.perf_counter()
        ns.write(upsert_rows=rows, distance_metric="cosine_distance")
        total_time += time.perf_counter() - start

    return total_time


def search_queries(
    ns: Any,
    queries: dict,
    embed_client: OpenAI,
) -> tuple[dict[str, list[str]], list[float]]:
    """Search all queries. Returns (results, latencies)."""
    print(f"Searching {len(queries)} queries...")

    results: dict[str, list[str]] = {}
    latencies: list[float] = []

    query_ids = list(queries.keys())

    for i in tqdm(range(0, len(query_ids), BATCH_SIZE), desc="Searching"):
        batch_ids = query_ids[i : i + BATCH_SIZE]
        batch_texts = [queries[qid] for qid in batch_ids]

        embeddings = embed_texts(embed_client, batch_texts)

        for qid, embedding in zip(batch_ids, embeddings):
            start = time.perf_counter()
            response = ns.query(
                rank_by=("vector", "ANN", embedding),
                top_k=TOP_K,
            )
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)

            results[qid] = [row.id for row in response.rows]

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
    # Set up output capture
    tee = TeeOutput()
    sys.stdout = tee

    print("=" * 60)
    print("Dense Vector Search Benchmark - ArguAna Dataset")
    print("=" * 60)

    # Print configuration
    print("\nConfiguration:")
    print(f"  Dataset:         {DATASET_NAME}")
    print(f"  Namespace:       {NAMESPACE}")
    print(f"  Embedding Model: {EMBEDDING_MODEL}")
    print(f"  Batch Size:      {BATCH_SIZE}")
    print(f"  Top K:           {TOP_K}")
    print(f"  Timestamp:       {datetime.now().isoformat()}")

    # Initialize clients
    embed_client = get_openrouter_client()
    tpuf_client = get_turbopuffer_client()
    ns = tpuf_client.namespace(NAMESPACE)

    # Load dataset
    print("\nLoading ArguAna dataset...")
    corpus, queries, qrels = load_arguana_dataset()
    print(f"  Corpus:  {len(corpus)} documents")
    print(f"  Queries: {len(queries)}")
    print(f"  QRels:   {len(qrels)} query-relevance pairs")

    # Index corpus
    print("\n" + "-" * 40)
    index_time = index_corpus(ns, corpus, embed_client)
    print(f"Total indexing time: {index_time:.2f}s")

    # Search queries
    print("\n" + "-" * 40)
    results, latencies = search_queries(ns, queries, embed_client)

    # Latency stats
    print(f"\nLatency Statistics:")
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
        "latency_avg_ms": np.mean(latencies),
        "latency_p50_ms": np.percentile(latencies, 50),
        "latency_p95_ms": np.percentile(latencies, 95),
        "latency_p99_ms": np.percentile(latencies, 99),
        "index_time_s": index_time,
    }


if __name__ == "__main__":
    main()
