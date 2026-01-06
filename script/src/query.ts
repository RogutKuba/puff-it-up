import {
  QUERY,
  tpuf,
  DENSE_NS,
  LATE_INTERACTION_NS,
  getDenseEmbedding,
  getColBERTEmbeddings,
} from "./shared";

// ============================================================================
// Dense Vector Query (Traditional)
// ============================================================================

/**
 * Query documents using traditional dense vector search.
 *
 * The query is converted to a single vector, then compared against
 * all document vectors using cosine similarity.
 */
export async function queryDense() {
  const ns = tpuf.namespace(DENSE_NS);

  const queryEmbedding = await getDenseEmbedding(QUERY);

  const results = await ns.query({
    rank_by: ["vector", "ANN", queryEmbedding],
    distance_metric: "cosine_distance",
    top_k: 10,
    include_attributes: ["text"],
  });

  return results;
}

// ============================================================================
// Late Interaction Query (ColBERT MaxSim)
// ============================================================================

/**
 * Query documents using late interaction with MaxSim scoring.
 *
 * Algorithm:
 * 1. Convert query to multiple token embeddings
 * 2. For each query token, find similar document tokens
 * 3. For each document, compute MaxSim:
 *    Score(Q, D) = Σ max(sim(q_i, d_j)) for all query tokens
 * 4. Rank documents by MaxSim score
 */
export async function queryLateInteraction() {
  const ns = tpuf.namespace(LATE_INTERACTION_NS);

  // Step 1: Get token-level embeddings for the query
  const [queryTokenEmbeddings] = await getColBERTEmbeddings([QUERY], "query");
  console.log(`Query tokenized into ${queryTokenEmbeddings.length} vectors`);

  // Step 2: Search for similar document tokens
  // turbopuffer supports up to 16 sub-queries at a time
  const BATCH_SIZE = 16;
  const TOP_K_PER_TOKEN = 50;

  const allResults: Array<{ doc_id: string; doc_text: string; dist: number }> =
    [];

  for (let i = 0; i < queryTokenEmbeddings.length; i += BATCH_SIZE) {
    const batch = queryTokenEmbeddings.slice(i, i + BATCH_SIZE);

    const response = await ns.multiQuery({
      queries: batch.map((embedding) => ({
        rank_by: ["vector", "ANN", embedding],
        top_k: TOP_K_PER_TOKEN,
        include_attributes: ["doc_id", "doc_text"],
      })),
    });

    for (const result of response.results) {
      if (result.rows) {
        for (const row of result.rows) {
          allResults.push({
            doc_id: row.doc_id as string,
            doc_text: row.doc_text as string,
            dist: row.$dist ?? 1,
          });
        }
      }
    }
  }

  // Step 3: Compute MaxSim scores per document
  // For each query token, find max similarity with any token from each doc
  const docScores = new Map<
    string,
    { text: string; maxSims: Map<number, number> }
  >();

  for (let qIdx = 0; qIdx < queryTokenEmbeddings.length; qIdx++) {
    // Get results for this query token (results are in order)
    const startIdx = qIdx * TOP_K_PER_TOKEN;
    const tokenResults = allResults.slice(startIdx, startIdx + TOP_K_PER_TOKEN);

    for (const result of tokenResults) {
      const similarity = 1 - result.dist; // Convert distance to similarity

      if (!docScores.has(result.doc_id)) {
        docScores.set(result.doc_id, {
          text: result.doc_text,
          maxSims: new Map(),
        });
      }

      const doc = docScores.get(result.doc_id)!;
      const currentMax = doc.maxSims.get(qIdx) ?? -Infinity;
      if (similarity > currentMax) {
        doc.maxSims.set(qIdx, similarity);
      }
    }
  }

  // Step 4: Sum max similarities to get final score
  const rankedDocs = Array.from(docScores.entries())
    .map(([docId, data]) => {
      const maxSimScore = Array.from(data.maxSims.values()).reduce(
        (sum, s) => sum + s,
        0
      );
      return { docId, text: data.text, maxSimScore };
    })
    .sort((a, b) => b.maxSimScore - a.maxSimScore);

  return rankedDocs;
}

// ============================================================================
// Main - Compare Both Methods
// ============================================================================

async function main() {
  console.log(`Query: "${QUERY}"\n`);

  console.log("=== Dense Vector Results ===");
  const denseResults = await queryDense();
  for (const row of denseResults.rows ?? []) {
    const similarity = 1 - (row.$dist ?? 1);
    console.log(`[${similarity.toFixed(4)}] ${row.text}`);
  }

  console.log("\n=== Late Interaction Results ===");
  const lateResults = await queryLateInteraction();
  for (const doc of lateResults) {
    console.log(`[${doc.maxSimScore.toFixed(4)}] ${doc.text}`);
  }
}

main().catch(console.error);
