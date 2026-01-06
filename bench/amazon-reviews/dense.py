"""
Dense vector search on Amazon Reviews dataset.
Uses a subset of reviews for qualitative comparison with late interaction.
Outputs actual results for article inclusion.
"""

import gzip
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from io import StringIO
from typing import Any

from dotenv import load_dotenv

load_dotenv()

import numpy as np
from openai import OpenAI
from tqdm import tqdm
import turbopuffer

from queries import QUERIES


# Configuration
DATASET_NAME = "amazon-reviews"
NAMESPACE = "amazon-reviews-dense-bench"
EMBEDDING_MODEL = "openai/text-embedding-3-small"
BATCH_SIZE = 32
TOP_K = 5

# Subset size - enough for variety, manageable for cost
NUM_REVIEWS = 15000

# Skip indexing if namespace already has data
SKIP_INDEXING = True

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query: str
    latency_ms: float  # Client-side measured latency
    server_total_ms: int = 0
    query_execution_ms: int = 0
    cache_hit_ratio: float = 0.0
    cache_temperature: str = ""
    exhaustive_search_count: int = 0
    billable_logical_bytes_queried: int = 0
    billable_logical_bytes_returned: int = 0
    tpuf_queries_made: int = 1  # Dense search makes 1 query per user query


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

        print("\nClient-Side Latency (includes network + embedding):")
        print(f"  P50: {np.percentile(latencies, 50):.2f} ms")
        print(f"  P90: {np.percentile(latencies, 90):.2f} ms")
        print(f"  P99: {np.percentile(latencies, 99):.2f} ms")
        print(f"  Avg: {np.mean(latencies):.2f} ms")

        print("\nServer-Side Latency (tpuf server_total_ms):")
        print(f"  P50: {np.percentile(server_times, 50):.0f} ms")
        print(f"  P90: {np.percentile(server_times, 90):.0f} ms")
        print(f"  P99: {np.percentile(server_times, 99):.0f} ms")
        print(f"  Avg: {np.mean(server_times):.0f} ms")

        print("\nBilling:")
        print(f"  Total bytes queried:  {self.total_bytes_queried:,}")
        print(f"  Total bytes returned: {self.total_bytes_returned:,}")

        # Cache stats
        cache_temps = [m.cache_temperature for m in self.query_metrics if m.cache_temperature]
        if cache_temps:
            hot = sum(1 for t in cache_temps if t == "hot")
            warm = sum(1 for t in cache_temps if t == "warm")
            cold = sum(1 for t in cache_temps if t == "cold")
            print(f"\nCache Temperature: hot={hot}, warm={warm}, cold={cold}")


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
        match = re.match(r"amazon_dense_(\d+)\.txt", fname)
        if match:
            max_num = max(max_num, int(match.group(1)))
    return os.path.join(RESULTS_DIR, f"amazon_dense_{max_num + 1}.txt")


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


def load_amazon_reviews() -> list[dict]:
    """
    Load Amazon Reviews dataset.
    Expects JSONL or JSONL.GZ files in data/ directory.
    Download from: https://amazon-reviews-2023.github.io/
    """
    reviews = []

    # Look for jsonl or jsonl.gz files
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
                    # Filter: need text and reasonable length
                    text = review.get('text', '')
                    if text and 50 < len(text) < 2000:
                        reviews.append({
                            'id': str(len(reviews)),
                            'text': text,
                            'title': review.get('title', ''),
                            'rating': review.get('rating', 0),
                            'product_title': review.get('parent_asin', 'Unknown'),
                            'verified': review.get('verified_purchase', False),
                        })
                except json.JSONDecodeError:
                    continue

        if len(reviews) >= NUM_REVIEWS:
            break

    print(f"  Loaded {len(reviews)} reviews")
    return reviews


def index_reviews(ns: Any, reviews: list[dict], embed_client: OpenAI) -> float:
    """Index reviews into turbopuffer. Returns total indexing time."""
    if SKIP_INDEXING:
        print("Skipping indexing (SKIP_INDEXING=True)")
        return 0.0

    print(f"Indexing {len(reviews)} reviews...")

    total_time = 0.0

    for i in tqdm(range(0, len(reviews), BATCH_SIZE), desc="Indexing"):
        batch = reviews[i : i + BATCH_SIZE]

        # Combine title and text for embedding
        batch_texts = [
            f"{r['title']} {r['text']}".strip() if r['title'] else r['text']
            for r in batch
        ]

        embeddings = embed_texts(embed_client, batch_texts)

        rows = [
            {
                "id": r['id'],
                "vector": embedding,
                "rating": int(r['rating']),
                "verified": r['verified'],
            }
            for r, embedding in zip(batch, embeddings)
        ]

        start = time.perf_counter()
        ns.write(upsert_rows=rows, distance_metric="cosine_distance")
        total_time += time.perf_counter() - start

    return total_time


def search_and_display(
    ns: Any,
    reviews: list[dict],
    queries: list[str],
    embed_client: OpenAI,
) -> BenchmarkStats:
    """Search queries and display results. Returns benchmark stats."""
    print(f"\nSearching {len(queries)} queries...\n")

    # Build lookup dict
    reviews_by_id = {r['id']: r for r in reviews}
    stats = BenchmarkStats()

    for query in queries:
        print("=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        # Get query embedding
        query_embedding = embed_texts(embed_client, [query])[0]

        # Search
        start = time.perf_counter()
        response = ns.query(
            rank_by=("vector", "ANN", query_embedding),
            top_k=TOP_K,
            include_attributes=["rating", "verified"],
        )
        latency = (time.perf_counter() - start) * 1000

        # Extract performance metrics from response (Pydantic models)
        perf = getattr(response, 'performance', None)
        billing = getattr(response, 'billing', None)

        metrics = QueryMetrics(
            query=query,
            latency_ms=latency,
            server_total_ms=getattr(perf, 'server_total_ms', 0) if perf else 0,
            query_execution_ms=getattr(perf, 'query_execution_ms', 0) if perf else 0,
            cache_hit_ratio=getattr(perf, 'cache_hit_ratio', 0.0) if perf else 0.0,
            cache_temperature=getattr(perf, 'cache_temperature', '') if perf else '',
            exhaustive_search_count=getattr(perf, 'exhaustive_search_count', 0) if perf else 0,
            billable_logical_bytes_queried=getattr(billing, 'billable_logical_bytes_queried', 0) if billing else 0,
            billable_logical_bytes_returned=getattr(billing, 'billable_logical_bytes_returned', 0) if billing else 0,
            tpuf_queries_made=1,
        )
        stats.query_metrics.append(metrics)

        print(f"[Client: {latency:.1f}ms | Server: {metrics.server_total_ms}ms | Cache: {metrics.cache_temperature}]\n")

        # Display results
        for rank, row in enumerate(response.rows, 1):
            doc_id = row.id
            rating = row.model_extra.get("rating", 0)
            verified = row.model_extra.get("verified", False)
            distance = row["$dist"]
            similarity = 1 - distance

            review = reviews_by_id.get(doc_id, {})
            title = review.get('title', '')
            text = review.get('text', '[Review not found]')

            stars = "★" * int(rating) + "☆" * (5 - int(rating))
            verified_str = "✓ Verified" if verified else ""

            print(f"#{rank} [{stars}] {verified_str} (sim: {similarity:.3f})")
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
    print("Dense Vector Search - Amazon Reviews Dataset")
    print("=" * 60)

    # Print configuration
    print("\nConfiguration:")
    print(f"  Dataset:         {DATASET_NAME}")
    print(f"  Namespace:       {NAMESPACE}")
    print(f"  Embedding Model: {EMBEDDING_MODEL}")
    print(f"  Num Reviews:     {NUM_REVIEWS}")
    print(f"  Batch Size:      {BATCH_SIZE}")
    print(f"  Top K:           {TOP_K}")
    print(f"  Num Queries:     {len(QUERIES)}")
    print(f"  Timestamp:       {datetime.now().isoformat()}")

    # Initialize clients
    embed_client = get_openrouter_client()
    tpuf_client = get_turbopuffer_client()
    ns = tpuf_client.namespace(NAMESPACE)

    # Load dataset
    print("\n" + "-" * 40)
    reviews = load_amazon_reviews()

    # Index reviews
    print("\n" + "-" * 40)
    index_time = index_reviews(ns, reviews, embed_client)
    print(f"Total indexing time: {index_time:.2f}s")

    # Search and display results
    print("\n" + "-" * 40)
    stats = search_and_display(ns, reviews, QUERIES, embed_client)

    # Print summary
    stats.print_summary()

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
