"""
Late interaction (ColBERT-style) search on Reddit comments dataset.
Uses Jina ColBERT v2 for multi-vector embeddings.
Since there are no relevance labels, outputs actual results for comparison.
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
import pandas as pd
from tqdm import tqdm
import turbopuffer


# Configuration
DATASET_NAME = "reddit-comments"
NAMESPACE = "reddit-late-bench"
JINA_MODEL = "jina-colbert-v2"
EMBEDDING_DIM = 128
BATCH_SIZE = 8
TOP_K = 5
MAX_CONCURRENT_QUERIES = 16

# Two-stage retrieval parameters
FIRST_STAGE_TOP_K = 50
SECOND_STAGE_TOP_K = 256

# Subset size
NUM_COMMENTS = 20000

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHECKPOINT_FILE = Path(__file__).parent / ".late_index_checkpoint.json"

# Same queries as dense.py for comparison
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
        match = re.match(r"reddit_late_(\d+)\.txt", fname)
        if match:
            max_num = max(max_num, int(match.group(1)))
    return os.path.join(RESULTS_DIR, f"reddit_late_{max_num + 1}.txt")


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


def index_comments(ns: Any, df: pd.DataFrame, jina_client: httpx.Client) -> float:
    """Index comments with multi-vector embeddings."""
    indexed_docs = load_checkpoint()
    if indexed_docs:
        print(f"Resuming from checkpoint: {len(indexed_docs)} docs already indexed")

    # Filter to unindexed docs
    df_to_index = df[~df["id"].isin(indexed_docs)]
    print(f"Indexing {len(df_to_index)} comments (multi-vector)...")

    if len(df_to_index) == 0:
        print("  All documents already indexed!")
        return 0.0

    total_time = 0.0
    total_vectors = 0

    for i in tqdm(range(0, len(df_to_index), BATCH_SIZE), desc="Indexing"):
        batch = df_to_index.iloc[i : i + BATCH_SIZE]
        batch_texts = batch["body"].tolist()
        batch_ids = batch["id"].tolist()

        # Get multi-vector embeddings
        multi_embeddings = get_multi_vector_embeddings(
            jina_client, batch_texts, input_type="document"
        )

        # Flatten: create one row per token vector
        rows = []
        for doc_id, token_embeddings, (_, row) in zip(batch_ids, multi_embeddings, batch.iterrows()):
            subreddit = row["subreddit"] if "subreddit" in row else "unknown"
            score = int(row["score"]) if "score" in row and pd.notna(row["score"]) else 0

            for token_idx, embedding in enumerate(token_embeddings):
                rows.append({
                    "id": f"{doc_id}_{token_idx}",
                    "vector": embedding,
                    "doc_id": doc_id,
                    "token_idx": token_idx,
                    "subreddit": subreddit,
                    "score": score,
                })
            total_vectors += len(token_embeddings)

        # Upsert to turbopuffer
        start = time.perf_counter()
        ns.write(upsert_rows=rows, distance_metric="cosine_distance")
        total_time += time.perf_counter() - start

        # Save checkpoint
        indexed_docs.update(batch_ids)
        save_checkpoint(indexed_docs)

    print(f"  Total token vectors indexed: {total_vectors}")
    return total_time


def first_stage_retrieval(
    ns: Any,
    query_embeddings: list[list[float]],
) -> set[str]:
    """First stage: ANN search for each query token to get candidate doc IDs."""
    doc_ids: set[str] = set()

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
    """Second stage: Filter to candidates and compute MaxSim scores."""
    doc_scores: dict[str, float] = defaultdict(float)

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

        for result_set in results.results:
            seen_doc_ids: set[str] = set()
            for chunk in result_set.rows:
                doc_id = chunk.model_extra.get("doc_id") or chunk.get("doc_id")
                if doc_id and doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    similarity = 1 - chunk["$dist"]
                    doc_scores[doc_id] += similarity

    return dict(doc_scores)


def search_query(
    ns: Any,
    query_embeddings: list[list[float]],
) -> tuple[list[str], int]:
    """Two-stage search. Returns (doc_ids, num_candidates)."""
    candidate_docs = first_stage_retrieval(ns, query_embeddings)

    if not candidate_docs:
        return [], 0

    doc_scores = second_stage_scoring(ns, query_embeddings, candidate_docs)

    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_docs[:TOP_K]], len(candidate_docs)


def search_and_display(
    ns: Any,
    df: pd.DataFrame,
    queries: list[str],
    jina_client: httpx.Client,
) -> list[float]:
    """Search queries and display results. Returns latencies."""
    print(f"\nSearching {len(queries)} queries...\n")

    latencies: list[float] = []

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
        doc_ids, num_candidates = search_query(ns, query_embeddings)
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)

        print(f"[Latency: {latency:.1f}ms, Candidates: {num_candidates}]\n")

        # Display results
        for rank, doc_id in enumerate(doc_ids, 1):
            # Get comment details from dataframe
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

    return latencies


def main():
    # Set up output capture
    tee = TeeOutput()
    sys.stdout = tee

    print("=" * 60)
    print("Late Interaction Search - Reddit Comments Dataset")
    print("=" * 60)

    # Print configuration
    print("\nConfiguration:")
    print(f"  Dataset:            {DATASET_NAME}")
    print(f"  Namespace:          {NAMESPACE}")
    print(f"  Jina Model:         {JINA_MODEL}")
    print(f"  Embedding Dim:      {EMBEDDING_DIM}")
    print(f"  Num Comments:       {NUM_COMMENTS}")
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
    df = load_reddit_dataset()
    print(f"  Loaded {len(df)} comments")
    if "subreddit" in df.columns:
        print(f"  Subreddits: {df['subreddit'].nunique()}")

    # Index comments
    print("\n" + "-" * 40)
    index_time = index_comments(ns, df, jina_client)
    print(f"Total indexing time: {index_time:.2f}s")

    # Search and display results
    print("\n" + "-" * 40)
    latencies = search_and_display(ns, df, QUERIES, jina_client)

    # Latency stats
    print("\n" + "=" * 60)
    print("Latency Statistics (search only, excludes embedding):")
    print(f"  Average: {np.mean(latencies):.2f} ms")
    print(f"  P50:     {np.percentile(latencies, 50):.2f} ms")
    print(f"  P95:     {np.percentile(latencies, 95):.2f} ms")

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
