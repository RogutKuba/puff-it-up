"""
Late interaction (ColBERT-style) search on Amazon Reviews dataset.
Uses Jina ColBERT v2 for multi-vector embeddings.
Outputs actual results for article comparison with dense search.
"""

import gzip
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
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

from queries import QUERIES


# Configuration
DATASET_NAME = "amazon-reviews"
NAMESPACE = "amazon-reviews-late-bench"
JINA_MODEL = "jina-colbert-v2"
EMBEDDING_DIM = 128
BATCH_SIZE = 8
TOP_K = 5
MAX_CONCURRENT_QUERIES = 16

# Two-stage retrieval parameters
FIRST_STAGE_TOP_K = 50
SECOND_STAGE_TOP_K = 256

# Subset size
NUM_REVIEWS = 15000

# Skip indexing if namespace already has data
SKIP_INDEXING = True

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHECKPOINT_FILE = Path(__file__).parent / ".late_index_checkpoint.json"


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query: str
    latency_ms: float  # Client-side measured latency
    server_total_ms: int = 0  # Sum of all tpuf server_total_ms
    query_execution_ms: int = 0  # Sum of all tpuf query_execution_ms
    billable_logical_bytes_queried: int = 0
    billable_logical_bytes_returned: int = 0
    tpuf_queries_made: int = 0  # Number of multi_query batches * queries per batch
    num_query_tokens: int = 0  # Number of token embeddings in query
    num_candidates: int = 0  # Number of candidate docs from first stage
    first_stage_server_ms: int = 0
    second_stage_server_ms: int = 0


@dataclass
class BenchmarkStats:
    """Aggregated benchmark statistics."""
    query_metrics: list[QueryMetrics] = field(default_factory=list)

    @property
    def latencies(self) -> list[float]:
        return [m.latency_ms for m in self.query_metrics]

    @property
    def server_times(self) -> list[int]:
        return [m.server_total_ms for m in self.query_metrics]

    @property
    def total_tpuf_queries(self) -> int:
        return sum(m.tpuf_queries_made for m in self.query_metrics)

    @property
    def total_bytes_queried(self) -> int:
        return sum(m.billable_logical_bytes_queried for m in self.query_metrics)

    @property
    def total_bytes_returned(self) -> int:
        return sum(m.billable_logical_bytes_returned for m in self.query_metrics)

    @property
    def avg_query_tokens(self) -> float:
        return np.mean([m.num_query_tokens for m in self.query_metrics])

    @property
    def avg_candidates(self) -> float:
        return np.mean([m.num_candidates for m in self.query_metrics])

    def print_summary(self):
        latencies = self.latencies
        server_times = self.server_times

        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        print("\nQuery Statistics:")
        print(f"  Total user queries:     {len(self.query_metrics)}")
        print(f"  Total tpuf API queries: {self.total_tpuf_queries}")
        print(f"  Queries per user query: {self.total_tpuf_queries / len(self.query_metrics):.1f}")
        print(f"  Avg query tokens:       {self.avg_query_tokens:.1f}")
        print(f"  Avg candidates (stage1): {self.avg_candidates:.1f}")

        print("\nClient-Side Latency (includes network + embedding):")
        print(f"  P50: {np.percentile(latencies, 50):.2f} ms")
        print(f"  P90: {np.percentile(latencies, 90):.2f} ms")
        print(f"  P99: {np.percentile(latencies, 99):.2f} ms")
        print(f"  Avg: {np.mean(latencies):.2f} ms")

        print("\nServer-Side Latency (sum of tpuf server_total_ms across stages):")
        print(f"  P50: {np.percentile(server_times, 50):.0f} ms")
        print(f"  P90: {np.percentile(server_times, 90):.0f} ms")
        print(f"  P99: {np.percentile(server_times, 99):.0f} ms")
        print(f"  Avg: {np.mean(server_times):.0f} ms")

        # Stage breakdown
        first_stage_times = [m.first_stage_server_ms for m in self.query_metrics]
        second_stage_times = [m.second_stage_server_ms for m in self.query_metrics]
        print(f"\n  First stage avg:  {np.mean(first_stage_times):.0f} ms")
        print(f"  Second stage avg: {np.mean(second_stage_times):.0f} ms")

        print("\nBilling:")
        print(f"  Total bytes queried:  {self.total_bytes_queried:,}")
        print(f"  Total bytes returned: {self.total_bytes_returned:,}")


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
    """Get the next available result filename."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    existing = os.listdir(RESULTS_DIR)
    max_num = 0
    for fname in existing:
        match = re.match(r"amazon_late_(\d+)\.txt", fname)
        if match:
            max_num = max(max_num, int(match.group(1)))
    return os.path.join(RESULTS_DIR, f"amazon_late_{max_num + 1}.txt")


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
    """Get multi-vector embeddings from Jina ColBERT v2."""
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


def load_amazon_reviews() -> list[dict]:
    """Load Amazon Reviews dataset."""
    reviews = []

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    data_files = [f for f in os.listdir(DATA_DIR) if f.endswith(('.jsonl', '.jsonl.gz'))]

    if not data_files:
        print(f"\nNo data files found in {DATA_DIR}")
        print("Please download review data from:")
        print("  https://amazon-reviews-2023.github.io/")
        print("\nDownload one or more category files (e.g., Electronics.jsonl.gz)")
        print(f"and save them to: {DATA_DIR}")
        sys.exit(1)

    print(f"Found {len(data_files)} data file(s)")

    for filename in data_files:
        filepath = os.path.join(DATA_DIR, filename)
        print(f"  Loading {filename}...")

        open_fn = gzip.open if filename.endswith('.gz') else open
        mode = 'rt' if filename.endswith('.gz') else 'r'

        with open_fn(filepath, mode, encoding='utf-8') as f:
            for line in f:
                if len(reviews) >= NUM_REVIEWS:
                    break
                try:
                    review = json.loads(line.strip())
                    text = review.get('text', '')
                    if text and 50 < len(text) < 2000:
                        reviews.append({
                            'id': str(len(reviews)),
                            'text': text,
                            'title': review.get('title', ''),
                            'rating': review.get('rating', 0),
                            'verified': review.get('verified_purchase', False),
                        })
                except json.JSONDecodeError:
                    continue

        if len(reviews) >= NUM_REVIEWS:
            break

    print(f"  Loaded {len(reviews)} reviews")
    return reviews


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


def index_reviews(ns: Any, reviews: list[dict], jina_client: httpx.Client) -> float:
    """Index reviews with multi-vector embeddings."""
    if SKIP_INDEXING:
        print("Skipping indexing (SKIP_INDEXING=True)")
        return 0.0

    indexed_docs = load_checkpoint()
    if indexed_docs:
        print(f"Resuming from checkpoint: {len(indexed_docs)} docs already indexed")

    # Filter to unindexed docs
    reviews_to_index = [r for r in reviews if r['id'] not in indexed_docs]
    print(f"Indexing {len(reviews_to_index)} reviews (multi-vector)...")

    if len(reviews_to_index) == 0:
        print("  All documents already indexed!")
        return 0.0

    total_time = 0.0
    total_vectors = 0

    for i in tqdm(range(0, len(reviews_to_index), BATCH_SIZE), desc="Indexing"):
        batch = reviews_to_index[i : i + BATCH_SIZE]

        batch_texts = [
            f"{r['title']} {r['text']}".strip() if r['title'] else r['text']
            for r in batch
        ]

        # Get multi-vector embeddings
        multi_embeddings = get_multi_vector_embeddings(
            jina_client, batch_texts, input_type="document"
        )

        # Flatten: create one row per token vector
        rows = []
        for review, token_embeddings in zip(batch, multi_embeddings):
            for token_idx, embedding in enumerate(token_embeddings):
                rows.append({
                    "id": f"{review['id']}_{token_idx}",
                    "vector": embedding,
                    "doc_id": review['id'],
                    "token_idx": token_idx,
                    "rating": int(review['rating']),
                    "verified": review['verified'],
                })
            total_vectors += len(token_embeddings)

        # Upsert to turbopuffer
        start = time.perf_counter()
        ns.write(upsert_rows=rows, distance_metric="cosine_distance")
        total_time += time.perf_counter() - start

        # Save checkpoint
        indexed_docs.update(r['id'] for r in batch)
        save_checkpoint(indexed_docs)

    print(f"  Total token vectors indexed: {total_vectors}")
    return total_time


@dataclass
class StageMetrics:
    """Metrics accumulated during a retrieval stage."""
    server_total_ms: int = 0
    query_execution_ms: int = 0
    bytes_queried: int = 0
    bytes_returned: int = 0
    queries_made: int = 0


def first_stage_retrieval(
    ns: Any,
    query_embeddings: list[list[float]],
) -> tuple[set[str], StageMetrics]:
    """First stage: ANN search for each query token to get candidate doc IDs."""
    doc_ids: set[str] = set()
    metrics = StageMetrics()

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

        # Extract performance/billing from multi_query response (Pydantic models)
        perf = getattr(results, 'performance', None)
        billing = getattr(results, 'billing', None)

        metrics.server_total_ms += getattr(perf, 'server_total_ms', 0) if perf else 0
        metrics.query_execution_ms += getattr(perf, 'query_execution_ms', 0) if perf else 0
        metrics.bytes_queried += getattr(billing, 'billable_logical_bytes_queried', 0) if billing else 0
        metrics.bytes_returned += getattr(billing, 'billable_logical_bytes_returned', 0) if billing else 0
        metrics.queries_made += len(batch_embeddings)

        for result_set in results.results:
            for result in result_set.rows:
                doc_id = getattr(result, 'doc_id', None) or result.model_extra.get("doc_id")
                if doc_id:
                    doc_ids.add(doc_id)

    return doc_ids, metrics


def second_stage_scoring(
    ns: Any,
    query_embeddings: list[list[float]],
    candidate_doc_ids: set[str],
) -> tuple[dict[str, float], StageMetrics]:
    """Second stage: Filter to candidates and compute MaxSim scores."""
    doc_scores: dict[str, float] = defaultdict(float)
    metrics = StageMetrics()

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

        # Extract performance/billing from multi_query response (Pydantic models)
        perf = getattr(results, 'performance', None)
        billing = getattr(results, 'billing', None)

        metrics.server_total_ms += getattr(perf, 'server_total_ms', 0) if perf else 0
        metrics.query_execution_ms += getattr(perf, 'query_execution_ms', 0) if perf else 0
        metrics.bytes_queried += getattr(billing, 'billable_logical_bytes_queried', 0) if billing else 0
        metrics.bytes_returned += getattr(billing, 'billable_logical_bytes_returned', 0) if billing else 0
        metrics.queries_made += len(batch_embeddings)

        for result_set in results.results:
            seen_doc_ids: set[str] = set()
            for chunk in result_set.rows:
                doc_id = getattr(chunk, 'doc_id', None) or chunk.model_extra.get("doc_id")
                if doc_id and doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    similarity = 1 - chunk["$dist"]
                    doc_scores[doc_id] += similarity

    return dict(doc_scores), metrics


def search_query(
    ns: Any,
    query_embeddings: list[list[float]],
) -> tuple[list[str], int, StageMetrics, StageMetrics]:
    """Two-stage search. Returns (doc_ids, num_candidates, first_stage_metrics, second_stage_metrics)."""
    candidate_docs, first_metrics = first_stage_retrieval(ns, query_embeddings)

    if not candidate_docs:
        return [], 0, first_metrics, StageMetrics()

    doc_scores, second_metrics = second_stage_scoring(ns, query_embeddings, candidate_docs)

    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_docs[:TOP_K]], len(candidate_docs), first_metrics, second_metrics


def search_and_display(
    ns: Any,
    reviews: list[dict],
    queries: list[str],
    jina_client: httpx.Client,
) -> BenchmarkStats:
    """Search queries and display results. Returns benchmark stats."""
    print(f"\nSearching {len(queries)} queries...\n")

    reviews_by_id = {r['id']: r for r in reviews}
    stats = BenchmarkStats()

    for query in queries:
        print("=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        # Get query embeddings
        query_embeddings = get_multi_vector_embeddings(
            jina_client, [query], input_type="query"
        )[0]

        # Search
        start = time.perf_counter()
        doc_ids, num_candidates, first_metrics, second_metrics = search_query(ns, query_embeddings)
        latency = (time.perf_counter() - start) * 1000

        # Aggregate metrics from both stages
        metrics = QueryMetrics(
            query=query,
            latency_ms=latency,
            server_total_ms=first_metrics.server_total_ms + second_metrics.server_total_ms,
            query_execution_ms=first_metrics.query_execution_ms + second_metrics.query_execution_ms,
            billable_logical_bytes_queried=first_metrics.bytes_queried + second_metrics.bytes_queried,
            billable_logical_bytes_returned=first_metrics.bytes_returned + second_metrics.bytes_returned,
            tpuf_queries_made=first_metrics.queries_made + second_metrics.queries_made,
            num_query_tokens=len(query_embeddings),
            num_candidates=num_candidates,
            first_stage_server_ms=first_metrics.server_total_ms,
            second_stage_server_ms=second_metrics.server_total_ms,
        )
        stats.query_metrics.append(metrics)

        print(f"[Client: {latency:.1f}ms | Server: {metrics.server_total_ms}ms | Tokens: {len(query_embeddings)} | Candidates: {num_candidates}]\n")

        # Display results
        for rank, doc_id in enumerate(doc_ids, 1):
            review = reviews_by_id.get(doc_id, {})
            title = review.get('title', '')
            text = review.get('text', '[Review not found]')
            rating = review.get('rating', 0)
            verified = review.get('verified', False)

            stars = "★" * int(rating) + "☆" * (5 - int(rating))
            verified_str = "✓ Verified" if verified else ""

            print(f"#{rank} [{stars}] {verified_str}")
            if title:
                print(f"   \"{title}\"")
            print(f"   {text}")
            print()

    return stats


def main():
    # Set up output capture
    tee = TeeOutput()
    sys.stdout = tee

    print("=" * 60)
    print("Late Interaction Search - Amazon Reviews Dataset")
    print("=" * 60)

    # Print configuration
    print("\nConfiguration:")
    print(f"  Dataset:            {DATASET_NAME}")
    print(f"  Namespace:          {NAMESPACE}")
    print(f"  Jina Model:         {JINA_MODEL}")
    print(f"  Embedding Dim:      {EMBEDDING_DIM}")
    print(f"  Num Reviews:        {NUM_REVIEWS}")
    print(f"  Batch Size:         {BATCH_SIZE}")
    print(f"  Top K:              {TOP_K}")
    print(f"  First Stage Top K:  {FIRST_STAGE_TOP_K}")
    print(f"  Second Stage Top K: {SECOND_STAGE_TOP_K}")
    print(f"  Num Queries:        {len(QUERIES)}")
    print(f"  Timestamp:          {datetime.now().isoformat()}")

    # Initialize clients
    jina_client = get_jina_client()
    tpuf_client = get_turbopuffer_client()
    ns = tpuf_client.namespace(NAMESPACE)

    # Load dataset
    print("\n" + "-" * 40)
    reviews = load_amazon_reviews()

    # Index reviews
    print("\n" + "-" * 40)
    index_time = index_reviews(ns, reviews, jina_client)
    print(f"Total indexing time: {index_time:.2f}s")

    # Search and display results
    print("\n" + "-" * 40)
    stats = search_and_display(ns, reviews, QUERIES, jina_client)

    # Print summary
    stats.print_summary()

    # Clear checkpoint on success
    clear_checkpoint()

    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)

    # Save results to file
    sys.stdout = tee.console
    result_file = get_next_result_filename()
    with open(result_file, "w") as f:
        f.write(tee.buffer.getvalue())
    print(f"\nResults saved to: {result_file}")


if __name__ == "__main__":
    main()
