"""
Hybrid search: Dense first-stage + Late interaction reranking on Amazon Reviews.
Uses OpenAI for high-recall candidate retrieval, Jina ColBERT v2 for precision reranking.
"""

import gzip
import json
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


# Configuration
DATASET_NAME = "amazon-reviews"
DENSE_NAMESPACE = "amazon-reviews-dense-bench"
LATE_NAMESPACE = "amazon-reviews-late-bench"

# Models
DENSE_MODEL = "openai/text-embedding-3-small"
COLBERT_MODEL = "jina-colbert-v2"
COLBERT_DIM = 128

# Retrieval parameters
FIRST_STAGE_TOP_K = 100
FINAL_TOP_K = 5
MAX_CONCURRENT_QUERIES = 16

NUM_REVIEWS = 15000

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Queries designed for appliances - multi-attribute queries benefit late interaction
# These queries have multiple specific requirements that need fine-grained token matching
QUERIES = [
    # Multi-attribute queries (late interaction should excel here)
    "quiet dishwasher that actually cleans dishes well",
    "energy efficient refrigerator that keeps food fresh longer",
    "blender powerful enough to crush ice but easy to clean",
    "slow cooker with programmable timer and keep warm function",
    "vacuum cleaner lightweight but strong suction for pet hair",
    # Specific problem/solution queries
    "air fryer that cooks evenly without preheating",
    "coffee maker that keeps coffee hot without burning it",
    "toaster with wide slots for thick bagels and frozen waffles",
    "instant pot replacement for multiple kitchen appliances",
    "water filter that removes chlorine taste and is easy to install",
]


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
        match = re.match(r"amazon_dense_late_(\d+)\.txt", fname)
        if match:
            max_num = max(max_num, int(match.group(1)))
    return os.path.join(RESULTS_DIR, f"amazon_dense_late_{max_num + 1}.txt")


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
    """Second stage: Re-rank candidates using ColBERT MaxSim."""
    if not candidate_docs:
        return []

    doc_scores: dict[str, float] = defaultdict(float)

    for i in range(0, len(query_embeddings), MAX_CONCURRENT_QUERIES):
        batch_embeddings = query_embeddings[i : i + MAX_CONCURRENT_QUERIES]

        queries = [
            {
                "rank_by": ["vector", "ANN", emb],
                "top_k": 256,
                "include_attributes": ["doc_id"],
                "filters": ["doc_id", "In", candidate_docs],
            }
            for emb in batch_embeddings
        ]

        response = ns.multi_query(queries=queries)

        for result in response.results:
            doc_max: dict[str, float] = {}
            for row in result.rows:
                doc_id = row.model_extra.get("doc_id") or row.get("doc_id")
                if doc_id:
                    similarity = 1.0 - row["$dist"]
                    if doc_id not in doc_max or similarity > doc_max[doc_id]:
                        doc_max[doc_id] = similarity

            for doc_id, max_score in doc_max.items():
                doc_scores[doc_id] += max_score

    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_docs[:FINAL_TOP_K]]


def search_and_display(
    dense_ns: Any,
    late_ns: Any,
    reviews: list[dict],
    queries: list[str],
    openrouter_client: OpenAI,
    jina_client: httpx.Client,
) -> tuple[list[float], list[float]]:
    """Search queries and display results. Returns (first_latencies, second_latencies)."""
    print(f"\nSearching {len(queries)} queries (dense + late rerank)...\n")

    reviews_by_id = {r['id']: r for r in reviews}
    first_latencies: list[float] = []
    second_latencies: list[float] = []

    for query in queries:
        print("=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        # Get embeddings
        dense_embedding = embed_dense(openrouter_client, [query])[0]
        colbert_embeddings = embed_colbert(jina_client, [query], input_type="query")[0]

        # First stage: Dense retrieval
        start = time.perf_counter()
        candidates = first_stage_dense(dense_ns, dense_embedding)
        first_ms = (time.perf_counter() - start) * 1000
        first_latencies.append(first_ms)

        # Second stage: ColBERT rerank
        start = time.perf_counter()
        doc_ids = second_stage_rerank(late_ns, colbert_embeddings, candidates)
        second_ms = (time.perf_counter() - start) * 1000
        second_latencies.append(second_ms)

        print(f"[Dense: {first_ms:.1f}ms, Rerank: {second_ms:.1f}ms, Total: {first_ms + second_ms:.1f}ms]\n")

        # Display results
        for rank, doc_id in enumerate(doc_ids, 1):
            review = reviews_by_id.get(doc_id, {})
            title = review.get('title', '')
            text = review.get('text', '[Review not found]')
            rating = review.get('rating', 0)
            verified = review.get('verified', False)

            if len(text) > 250:
                text = text[:250] + "..."

            stars = "★" * int(rating) + "☆" * (5 - int(rating))
            verified_str = "✓ Verified" if verified else ""

            print(f"#{rank} [{stars}] {verified_str}")
            if title:
                print(f"   \"{title}\"")
            print(f"   {text}")
            print()

    return first_latencies, second_latencies


def main():
    # Set up output capture
    tee = TeeOutput()
    sys.stdout = tee

    print("=" * 60)
    print("Hybrid Search: Dense + Late Interaction Rerank")
    print("Amazon Reviews Dataset")
    print("=" * 60)

    # Print configuration
    print("\nConfiguration:")
    print(f"  Dataset:           {DATASET_NAME}")
    print(f"  Dense Namespace:   {DENSE_NAMESPACE}")
    print(f"  Late Namespace:    {LATE_NAMESPACE}")
    print(f"  Dense Model:       {DENSE_MODEL}")
    print(f"  ColBERT Model:     {COLBERT_MODEL}")
    print(f"  First Stage Top K: {FIRST_STAGE_TOP_K}")
    print(f"  Final Top K:       {FINAL_TOP_K}")
    print(f"  Num Queries:       {len(QUERIES)}")
    print(f"  Timestamp:         {datetime.now().isoformat()}")

    # Initialize clients
    openrouter_client = get_openrouter_client()
    jina_client = get_jina_client()
    tpuf_client = get_turbopuffer_client()

    dense_ns = tpuf_client.namespace(DENSE_NAMESPACE)
    late_ns = tpuf_client.namespace(LATE_NAMESPACE)

    # Load dataset (for displaying results)
    print("\n" + "-" * 40)
    reviews = load_amazon_reviews()

    print("\nUsing existing indices:")
    print(f"  Dense namespace: {DENSE_NAMESPACE}")
    print(f"  Late namespace:  {LATE_NAMESPACE}")

    # Search and display results
    print("\n" + "-" * 40)
    first_latencies, second_latencies = search_and_display(
        dense_ns, late_ns, reviews, QUERIES, openrouter_client, jina_client
    )

    total_latencies = [f + s for f, s in zip(first_latencies, second_latencies)]

    # Latency stats
    print("\n" + "=" * 60)
    print("Latency Statistics (search only, excludes embedding):")
    print(f"\n  First Stage (Dense):")
    print(f"    Average: {np.mean(first_latencies):.2f} ms")
    print(f"    P50:     {np.percentile(first_latencies, 50):.2f} ms")

    print(f"\n  Second Stage (ColBERT Rerank):")
    print(f"    Average: {np.mean(second_latencies):.2f} ms")
    print(f"    P50:     {np.percentile(second_latencies, 50):.2f} ms")

    print(f"\n  Total:")
    print(f"    Average: {np.mean(total_latencies):.2f} ms")
    print(f"    P50:     {np.percentile(total_latencies, 50):.2f} ms")
    print(f"    P95:     {np.percentile(total_latencies, 95):.2f} ms")

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
