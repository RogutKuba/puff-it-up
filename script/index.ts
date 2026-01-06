import Turbopuffer from '@turbopuffer/turbopuffer';
import OpenAI from 'openai';
import { env } from './env';

const DOC1 =
  'Thousands of movies and shows, new releases weekly. Download anything for offline viewing. Plan includes ads every 15 minutes.';
const DOC2 =
  'Huge library of classic and original content. Some titles available for download. Completely ad-free streaming experience.';

const DOCUMENTS = [
  { id: 'doc1', text: DOC1 },
  { id: 'doc2', text: DOC2 },
];

const QUERY = 'streaming service without ads';

const tpuf = new Turbopuffer({
  apiKey: env.TURBOPUFFER_API_KEY,
  region: 'gcp-northamerica-northeast2',
});

const NS = 'basic-example-default-2';
const LATE_INTERACTION_NS = 'colbert-late-interaction-2';

const openai = new OpenAI({
  apiKey: env.OPENAI_API_KEY,
});

// ============================================================================
// Jina ColBERT v2 API Client
// ============================================================================

interface JinaMultiVectorResponse {
  model: string;
  object: string;
  usage: { total_tokens: number };
  data: Array<{
    object: string;
    index: number;
    embeddings: number[][];
  }>;
}

/**
 * Get multi-vector embeddings from Jina ColBERT v2
 * Each input text returns multiple token-level embeddings (contextualized)
 */
async function getColBERTEmbeddings(
  texts: string[],
  inputType: 'document' | 'query'
): Promise<number[][][]> {
  const response = await fetch('https://api.jina.ai/v1/multi-vector', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${env.JINA_API_KEY}`,
    },
    body: JSON.stringify({
      model: 'jina-colbert-v2',
      dimensions: 128,
      input_type: inputType,
      embedding_type: 'float',
      input: texts,
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Jina API error: ${response.status} - ${error}`);
  }

  const data = (await response.json()) as JinaMultiVectorResponse;
  // Each text returns an array of token embeddings
  return data.data.map((item) => item.embeddings);
}

// ============================================================================
// Dense Retrieval (Single Vector)
// ============================================================================

const createEmbeddings = async (text: string) => {
  const response = await openai.embeddings.create({
    model: 'text-embedding-3-small',
    input: text,
  });
  return response.data[0].embedding;
};

async function insertDocuments() {
  // create namespace if it doesn't exist
  const ns = tpuf.namespace(NS);

  console.log('Inserting DOC1');
  const doc1Embedding = await createEmbeddings(DOC1);

  console.log('Inserting DOC2');
  const doc2Embedding = await createEmbeddings(DOC2);

  await ns.write({
    upsert_rows: [
      { id: 'doc1', text: DOC1, vector: doc1Embedding },
      { id: 'doc2', text: DOC2, vector: doc2Embedding },
    ],
    distance_metric: 'cosine_distance',
    schema: {
      text: {
        type: 'string',
      },
    },
  });
}

async function queryDocuments() {
  const ns = tpuf.namespace(NS);

  const queryEmbedding = await createEmbeddings(QUERY);
  const results = await ns.query({
    rank_by: ['vector', 'ANN', queryEmbedding],
    distance_metric: 'cosine_distance',
    top_k: 10,
    include_attributes: ['text'],
  });
  console.log(results);
}

// ============================================================================
// Late Interaction (ColBERT) - Real Implementation
// ============================================================================

/**
 * Insert documents with token-level embeddings for late interaction.
 *
 * Each document is embedded using ColBERT which produces multiple token embeddings.
 * We store each token embedding as a separate row, linked back to its source document.
 */
async function insertDocumentsLateInteraction() {
  console.log('Getting ColBERT embeddings for documents...');

  // Get multi-vector embeddings for all documents
  const docTexts = DOCUMENTS.map((d) => d.text);
  const allDocEmbeddings = await getColBERTEmbeddings(docTexts, 'document');

  console.log(
    `Received embeddings: ${allDocEmbeddings.map(
      (e) => e.length
    )} tokens per document`
  );

  // Build rows: one row per token embedding, with doc_id reference
  const rows: Array<{
    id: string;
    doc_id: string;
    doc_text: string;
    token_index: number;
    vector: number[];
  }> = [];

  for (let docIdx = 0; docIdx < DOCUMENTS.length; docIdx++) {
    const doc = DOCUMENTS[docIdx];
    const tokenEmbeddings = allDocEmbeddings[docIdx];

    for (let tokenIdx = 0; tokenIdx < tokenEmbeddings.length; tokenIdx++) {
      rows.push({
        id: `${doc.id}_token_${tokenIdx}`,
        doc_id: doc.id,
        doc_text: doc.text,
        token_index: tokenIdx,
        vector: tokenEmbeddings[tokenIdx],
      });
    }
  }

  console.log(`Inserting ${rows.length} token embeddings...`);

  const ns = tpuf.namespace(LATE_INTERACTION_NS);
  await ns.write({
    upsert_rows: rows,
    distance_metric: 'cosine_distance',
    schema: {
      doc_id: { type: 'string' },
      doc_text: { type: 'string' },
      token_index: { type: 'uint' },
    },
  });

  console.log('Done inserting late interaction embeddings!');
}

/**
 * Query documents using late interaction (MaxSim).
 *
 * Algorithm:
 * 1. Get multi-vector embeddings for the query (one per query token)
 * 2. For each query token, find the most similar document tokens
 * 3. Group results by document
 * 4. For each document, compute MaxSim score:
 *    Score(Q, D) = Σ max_j(sim(q_i, d_j)) for all query tokens q_i
 * 5. Rank documents by their MaxSim scores
 */
async function queryDocumentsLateInteraction() {
  console.log(`\nQuery: "${QUERY}"\n`);
  console.log('Getting ColBERT embeddings for query...');

  // Get multi-vector embeddings for the query
  const [queryTokenEmbeddings] = await getColBERTEmbeddings([QUERY], 'query');
  console.log(`Query has ${queryTokenEmbeddings.length} token embeddings`);

  const ns = tpuf.namespace(LATE_INTERACTION_NS);

  // For each query token, find the top-k most similar document tokens
  // We need enough results to cover all documents
  const TOP_K_PER_QUERY_TOKEN = 50;

  console.log('Searching for similar document tokens...');

  // Turbopuffer multiQuery API returns a list of results for each query token
  // only max 16 sub-queries are allowed at a time

  const batchSize = 16;
  const batches = queryTokenEmbeddings.reduce((acc, embedding, index) => {
    const batchIndex = Math.floor(index / batchSize);
    if (!acc[batchIndex]) {
      acc[batchIndex] = [];
    }
    acc[batchIndex].push(embedding);
    return acc;
  }, [] as number[][][]);

  const totalResults: Turbopuffer.Row[] = [];
  let batchIndex = 0;

  let startTime = Date.now();
  for (const batch of batches) {
    console.log(`Processing batch ${++batchIndex} of ${batches.length}...`);
    const results = await ns.multiQuery({
      queries: batch.map((embedding) => ({
        rank_by: ['vector', 'ANN', embedding],
        top_k: TOP_K_PER_QUERY_TOKEN,
        include_attributes: ['doc_id', 'doc_text'],
      })),
    });
    console.log(`Batch ${batchIndex} results: ${results.results.length}`);

    for (const r of results.results) {
      if (r.rows) {
        totalResults.push(...r.rows);
      }
    }
  }
  const endTime = Date.now();
  console.log(`Total time: ${endTime - startTime}ms`);

  // Compute MaxSim scores for each document
  // For each query token, we need the MAX similarity with any token from each document
  // Turbopuffer returns cosine_distance, so similarity = 1 - distance
  const docScores = new Map<string, { maxSims: number[]; text: string }>();

  for (
    let queryTokenIdx = 0;
    queryTokenIdx < queryTokenEmbeddings.length;
    queryTokenIdx++
  ) {
    // For this query token, find max similarity per document
    const docMaxSimForThisQueryToken = new Map<string, number>();

    for (const row of totalResults) {
      const docId = row?.doc_id as string;
      const docText = row?.doc_text as string;

      // Convert cosine_distance to similarity (Turbopuffer handles the vector math)
      const similarity = 1 - (row.$dist ?? 1);

      // Track the max similarity for this document for this query token
      const currentMax = docMaxSimForThisQueryToken.get(docId) ?? -Infinity;
      if (similarity > currentMax) {
        docMaxSimForThisQueryToken.set(docId, similarity);
      }

      // Initialize document score tracking if needed
      if (!docScores.has(docId)) {
        docScores.set(docId, {
          maxSims: new Array(queryTokenEmbeddings.length).fill(0),
          text: docText,
        });
      }
    }

    // Store the max similarity for each document for this query token
    for (const [docId, maxSim] of docMaxSimForThisQueryToken) {
      const docData = docScores.get(docId)!;
      docData.maxSims[queryTokenIdx] = maxSim;
    }
  }

  console.log(`Doc scores`, docScores);

  // Compute final MaxSim score for each document (sum of all query token max similarities)
  const rankedDocs = Array.from(docScores.entries())
    .map(([docId, data]) => ({
      docId,
      text: data.text,
      maxSimScore: data.maxSims.reduce((sum, s) => sum + s, 0),
      tokenScores: data.maxSims,
    }))
    .sort((a, b) => b.maxSimScore - a.maxSimScore);

  // Display results
  console.log('\n=== Late Interaction (ColBERT MaxSim) Results ===\n');
  for (const doc of rankedDocs) {
    console.log(`Document: ${doc.docId}`);
    console.log(`MaxSim Score: ${doc.maxSimScore.toFixed(4)}`);
    console.log(`Text: ${doc.text}`);
    console.log('---');
  }
}

// ============================================================================
// Main
// ============================================================================

// Uncomment the function you want to run:

// Dense retrieval
async function main() {
  await insertDocuments()
    .then(() => {
      console.log('Documents inserted');
      queryDocuments().catch(console.error);
    })
    .catch(console.error);

  console.log('--------------------------------\n');

  // Late interaction (ColBERT)
  insertDocumentsLateInteraction()
    .then(() => {
      console.log('Documents inserted');
      queryDocumentsLateInteraction().catch(console.error);
    })
    .catch(console.error);
}

main().catch(console.error);
