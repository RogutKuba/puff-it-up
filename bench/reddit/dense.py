"""
Dense vector search on Reddit comments dataset.
Downloads from Kaggle, indexes a subset, and runs qualitative queries.
Since there are no relevance labels, outputs actual results for comparison.
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
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import turbopuffer


# Configuration
DATASET_NAME = "reddit-comments"
NAMESPACE = "reddit-dense-bench-openai"
EMBEDDING_MODEL = "openai/text-embedding-3-small"
BATCH_SIZE = 32
TOP_K = 5

# Subset size - balance between variety and cost
NUM_COMMENTS = 20000

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Interesting queries to test semantic search
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
        match = re.match(r"reddit_dense_(\d+)\.txt", fname)
        if match:
            max_num = max(max_num, int(match.group(1)))
    return os.path.join(RESULTS_DIR, f"reddit_dense_{max_num + 1}.txt")


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


def load_reddit_dataset() -> pd.DataFrame:
    """
    Load Reddit comments dataset.
    Expects CSV file at data/reddit_comments.csv with columns: body, subreddit, score, etc.
    """
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
    df = df[df["body"].str.len() > 50]  # Filter out very short comments
    df = df[df["body"].str.len() < 2000]  # Filter out very long comments
    df = df[~df["body"].str.contains(r"^\[deleted\]$|^\[removed\]$", regex=True)]

    # Sample subset
    if len(df) > NUM_COMMENTS:
        df = df.sample(n=NUM_COMMENTS, random_state=42)

    df = df.reset_index(drop=True)
    df["id"] = df.index.astype(str)

    return df


def index_comments(ns: Any, df: pd.DataFrame, embed_client: OpenAI) -> float:
    """Index comments into turbopuffer. Returns total indexing time."""
    print(f"Indexing {len(df)} comments...")

    total_time = 0.0

    for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Indexing"):
        batch = df.iloc[i : i + BATCH_SIZE]
        batch_texts = batch["body"].tolist()
        batch_ids = batch["id"].tolist()

        embeddings = embed_texts(embed_client, batch_texts)

        rows = [
            {
                "id": doc_id,
                "vector": embedding,
                "subreddit": row["subreddit"] if "subreddit" in row else "unknown",
                "score": int(row["score"]) if "score" in row and pd.notna(row["score"]) else 0,
            }
            for doc_id, embedding, (_, row) in zip(batch_ids, embeddings, batch.iterrows())
        ]

        start = time.perf_counter()
        ns.write(upsert_rows=rows, distance_metric="cosine_distance")
        total_time += time.perf_counter() - start

    return total_time


def search_and_display(
    ns: Any,
    df: pd.DataFrame,
    queries: list[str],
    embed_client: OpenAI,
) -> list[float]:
    """Search queries and display results. Returns latencies."""
    print(f"\nSearching {len(queries)} queries...\n")

    latencies: list[float] = []

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
            include_attributes=["subreddit", "score"],
        )
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)

        print(f"[Latency: {latency:.1f}ms]\n")

        # Display results
        for rank, row in enumerate(response.rows, 1):
            doc_id = row.id
            subreddit = row.model_extra.get("subreddit", "unknown")
            score = row.model_extra.get("score", 0)
            distance = row["$dist"]
            similarity = 1 - distance

            # Get the actual comment text
            comment_text = df.loc[df["id"] == doc_id, "body"].values
            if len(comment_text) > 0:
                text = comment_text[0]
                # Truncate for display
                if len(text) > 300:
                    text = text[:300] + "..."
            else:
                text = "[Comment not found]"

            print(f"#{rank} [r/{subreddit}] (score: {score}, sim: {similarity:.3f})")
            print(f"   {text}")
            print()

    return latencies


def main():
    # Set up output capture
    tee = TeeOutput()
    sys.stdout = tee

    print("=" * 60)
    print("Dense Vector Search - Reddit Comments Dataset")
    print("=" * 60)

    # Print configuration
    print("\nConfiguration:")
    print(f"  Dataset:         {DATASET_NAME}")
    print(f"  Namespace:       {NAMESPACE}")
    print(f"  Embedding Model: {EMBEDDING_MODEL}")
    print(f"  Num Comments:    {NUM_COMMENTS}")
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
    df = load_reddit_dataset()
    print(f"  Loaded {len(df)} comments")
    if "subreddit" in df.columns:
        print(f"  Subreddits: {df['subreddit'].nunique()}")

    # Index comments
    print("\n" + "-" * 40)
    index_time = index_comments(ns, df, embed_client)
    print(f"Total indexing time: {index_time:.2f}s")

    # Search and display results
    print("\n" + "-" * 40)
    latencies = search_and_display(ns, df, QUERIES, embed_client)

    # Latency stats
    print("\n" + "=" * 60)
    print("Latency Statistics:")
    print(f"  Average: {np.mean(latencies):.2f} ms")
    print(f"  P50:     {np.percentile(latencies, 50):.2f} ms")
    print(f"  P95:     {np.percentile(latencies, 95):.2f} ms")

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
