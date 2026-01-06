"""
Hybrid search: Dense first-stage + Late interaction reranking on Reddit comments.
Uses OpenAI for candidate retrieval, Jina ColBERT v2 for MaxSim reranking.
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
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import turbopuffer


# Configuration
DATASET_NAME = "reddit-comments"
DENSE_NAMESPACE = "reddit-dense-bench-openai"
LATE_NAMESPACE = "reddit-late-bench"

# Models
DENSE_MODEL = "openai/text-embedding-3-small"
COLBERT_MODEL = "jina-colbert-v2"
COLBERT_DIM = 128

# Retrieval parameters
FIRST_STAGE_TOP_K = 100
FINAL_TOP_K = 5
MAX_CONCURRENT_QUERIES = 16

NUM_COMMENTS = 20000

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Same queries for comparison
QUERIES = [
    "What programming language should I learn first as a beginner?",
    "How do I deal with anxiety and depression?",
    "What are the best books everyone should read?",
    "Tips for losing weight and getting in shape",
    "Why is my code not working and how do I debug it?",
    "What are the most underrated video games?",
    "How do I improve my relationship with my parents?",
    "What's the best way to learn a new skill quickly?",
    "Unpopular opinions that are actually true",
    "What life advice would you give to your younger self?",
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
        match = re.match(r"reddit_dense_late_(\d+)\.txt", fname)
        if match:
            max_num = max(max_num, int(match.group(1)))
    return os.path.join(RESULTS_DIR, f"reddit_dense_late_{max_num + 1}.txt")


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


def load_reddit_dataset() -> pd.DataFrame:
    """Load Reddit comments dataset."""
    csv_path = os.path.join(DATA_DIR, "reddit_comments.csv")

    if not os.path.exists(csv_path):
        print(f"\nDataset not found at {csv_path}")
        print("Please download the dataset from:")
        print("  https://www.kaggle.com/datasets/smagnan/1-million-reddit-comments-from-40-subreddits")
        print(f"\nAnd save the CSV file to: {csv_path}")
        sys.exit(1)

    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Clean and filter
    df = df.dropna(subset=["body"])
    df = df[df["body"].str.len() > 50]
    df = df[df["body"].str.len() < 2000]
    df = df[~df["body"].str.contains(r"^\[deleted\]$|^\[removed\]$", regex=True)]

    # Sample subset
    if len(df) > NUM_COMMENTS:
        df = df.sample(n=NUM_COMMENTS, random_state=42)

    df = df.reset_index(drop=True)
    df["id"] = df.index.astype(str)

    return df


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
    df: pd.DataFrame,
    queries: list[str],
    openrouter_client: OpenAI,
    jina_client: httpx.Client,
) -> tuple[list[float], list[float]]:
    """Search queries and display results. Returns (first_latencies, second_latencies)."""
    print(f"\nSearching {len(queries)} queries (dense + late rerank)...\n")

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
            comment_row = df.loc[df["id"] == doc_id]
            if len(comment_row) > 0:
                row = comment_row.iloc[0]
                text = row["body"]
                subreddit = row.get("subreddit", "unknown")
                score = row.get("score", 0)

                if len(text) > 300:
                    text = text[:300] + "..."
            else:
                text = "[Comment not found]"
                subreddit = "unknown"
                score = 0

            print(f"#{rank} [r/{subreddit}] (score: {score})")
            print(f"   {text}")
            print()

    return first_latencies, second_latencies


def main():
    # Set up output capture
    tee = TeeOutput()
    sys.stdout = tee

    print("=" * 60)
    print("Hybrid Search: Dense + Late Interaction Rerank")
    print("Reddit Comments Dataset")
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
    df = load_reddit_dataset()
    print(f"  Loaded {len(df)} comments")

    print("\nUsing existing indices:")
    print(f"  Dense namespace: {DENSE_NAMESPACE}")
    print(f"  Late namespace:  {LATE_NAMESPACE}")

    # Search and display results
    print("\n" + "-" * 40)
    first_latencies, second_latencies = search_and_display(
        dense_ns, late_ns, df, QUERIES, openrouter_client, jina_client
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
