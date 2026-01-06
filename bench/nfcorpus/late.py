"""
Late interaction (ColBERT-style) benchmark on NFCorpus dataset.
Uses Jina ColBERT v2 for multi-vector embeddings and turbopuffer for storage/search.
Implements client-side MaxSim scoring.
"""

import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from io import StringIO
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
DATASET_NAME = "nfcorpus"
NAMESPACE = "nfcorpus-late-bench"
JINA_MODEL = "jina-colbert-v2"
EMBEDDING_DIM = 128
BATCH_SIZE = 8  # Smaller batch for multi-vector (more data per doc)
TOP_K = 10
MAX_CONCURRENT_QUERIES = 16  # Turbopuffer limit

# Two-stage retrieval parameters
FIRST_STAGE_TOP_K = 5  # Per-token candidates in first stage
SECOND_STAGE_TOP_K = 256  # Vectors to retrieve per token in second stage

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Checkpoint file for resumable indexing
CHECKPOINT_FILE = Path(__file__).parent / ".late_index_checkpoint.json"


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
    """Get the next available result filename (nfcorpus_late_N.txt)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    existing = os.listdir(RESULTS_DIR)
    max_num = 0
    for fname in existing:
        match = re.match(r"nfcorpus_late_(\d+)\.txt", fname)
        if match:
            max_num = max(max_num, int(match.group(1)))
    return os.path.join(RESULTS_DIR, f"nfcorpus_late_{max_num + 1}.txt")


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


def load_nfcorpus_dataset() -> tuple[dict, dict, dict]:
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


def first_stage_retrieval(
    ns: Any,
    query_embeddings: list[list[float]],
) -> set[str]:
    """
    First stage: ANN search for each query token to get candidate document IDs.
    """
    doc_ids: set[str] = set()

    # Process query tokens in batches of MAX_CONCURRENT_QUERIES
    for i in range(0, len(query_embeddings), MAX_CONCURRENT_QUERIES):
        batch_embeddings = query_embeddings[i : i + MAX_CONCURRENT_QUERIES]

        multi_query_payload = [
            {
                "rank_by": ["vector", "ANN", emb],
                "include_attributes": ["doc_id"],
                "top_k": FIRST_STAGE_TOP_K,
            }
            for emb in batch_embeddings
        ]
        results = ns.multi_query(queries=multi_query_payload)

        for result_set in results.results:
            for result in result_set.rows:
                doc_id = result.model_extra.get("doc_id") or result.get("doc_id")
                if doc_id:
                    doc_ids.add(doc_id)

    return doc_ids


def second_stage_scoring(
    ns: Any,
    query_embeddings: list[list[float]],
    candidate_doc_ids: set[str],
) -> dict[str, float]:
    """
    Second stage: Filter to candidate docs and compute MaxSim scores.
    """
    doc_scores: dict[str, float] = defaultdict(float)

    # Process query tokens in batches of MAX_CONCURRENT_QUERIES
    for i in range(0, len(query_embeddings), MAX_CONCURRENT_QUERIES):
        batch_embeddings = query_embeddings[i : i + MAX_CONCURRENT_QUERIES]

        multi_query_payload = [
            {
                "rank_by": ["vector", "ANN", emb],
                "include_attributes": ["doc_id"],
                "consistency": {"level": "eventual"},
                "filters": ["doc_id", "In", list(candidate_doc_ids)],
                "top_k": SECOND_STAGE_TOP_K,
            }
            for emb in batch_embeddings
        ]
        results = ns.multi_query(queries=multi_query_payload)

        # Compute MaxSim: for each query token, take max similarity per doc, then sum
        for result_set in results.results:
            seen_doc_ids: set[str] = set()
            for chunk in result_set.rows:
                doc_id = chunk.model_extra.get("doc_id") or chunk.get("doc_id")
                if doc_id and doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    # Convert distance to similarity
                    similarity = 1 - chunk["$dist"]
                    doc_scores[doc_id] += similarity

    return dict(doc_scores)


def search_query(
    ns: Any,
    query_embeddings: list[list[float]],
) -> list[str]:
    """
    Two-stage late interaction search:
    1. First stage: Get candidate docs via ANN search per query token
    2. Second stage: Compute MaxSim scores within filtered candidates
    Returns top-k document IDs sorted by MaxSim score.
    """
    # First stage: get candidate documents
    candidate_docs = first_stage_retrieval(ns, query_embeddings)

    if not candidate_docs:
        return []

    # Second stage: compute MaxSim scores within candidates
    doc_scores = second_stage_scoring(ns, query_embeddings, candidate_docs)

    # Sort by score descending and return top-k doc IDs
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_docs[:TOP_K]]


def search_queries(
    ns: Any,
    queries: dict,
    jina_client: httpx.Client,
) -> tuple[dict[str, list[str]], list[float]]:
    """Search all queries using two-stage late interaction. Returns (results, latencies)."""
    print(f"Searching {len(queries)} queries (two-stage late interaction)...")

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
    # Set up output capture
    tee = TeeOutput()
    sys.stdout = tee

    print("=" * 60)
    print("Two-Stage Late Interaction Benchmark - NFCorpus Dataset")
    print("=" * 60)

    # Print configuration
    print("\nConfiguration:")
    print(f"  Dataset:            {DATASET_NAME}")
    print(f"  Namespace:          {NAMESPACE}")
    print(f"  Jina Model:         {JINA_MODEL}")
    print(f"  Embedding Dim:      {EMBEDDING_DIM}")
    print(f"  Batch Size:         {BATCH_SIZE}")
    print(f"  Top K:              {TOP_K}")
    print(f"  First Stage Top K:  {FIRST_STAGE_TOP_K}")
    print(f"  Second Stage Top K: {SECOND_STAGE_TOP_K}")
    print(f"  Timestamp:          {datetime.now().isoformat()}")

    # Initialize clients
    jina_client = get_jina_client()
    tpuf_client = get_turbopuffer_client()
    ns = tpuf_client.namespace(NAMESPACE)

    # Load dataset
    print("\nLoading NFCorpus dataset...")
    corpus, queries, qrels = load_nfcorpus_dataset()
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
