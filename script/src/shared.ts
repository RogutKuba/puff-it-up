import Turbopuffer from "@turbopuffer/turbopuffer";
import OpenAI from "openai";
import { env } from "../env";

// ============================================================================
// Sample Data
// ============================================================================

export const DOCUMENTS = [
  {
    id: "doc1",
    text: "Thousands of movies and shows, new releases weekly. Download anything for offline viewing. Plan includes ads every 15 minutes.",
  },
  {
    id: "doc2",
    text: "Huge library of classic and original content. Some titles available for download. Completely ad-free streaming experience.",
  },
];

export const QUERY = "streaming service without ads";

// ============================================================================
// Turbopuffer Client
// ============================================================================

export const tpuf = new Turbopuffer({
  apiKey: env.TURBOPUFFER_API_KEY,
  region: "gcp-northamerica-northeast2",
});

// Namespace for traditional dense vector search
export const DENSE_NS = "late-interaction-dense";

// Namespace for late interaction (ColBERT) search
export const LATE_INTERACTION_NS = "late-interaction-colbert";

// ============================================================================
// OpenAI Client (for dense embeddings)
// ============================================================================

const openai = new OpenAI({
  apiKey: env.OPENAI_API_KEY,
});

/**
 * Generate a single dense embedding for text using OpenAI.
 * Returns a single vector representing the entire input.
 */
export async function getDenseEmbedding(text: string): Promise<number[]> {
  const response = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: text,
  });
  return response.data[0].embedding;
}

// ============================================================================
// Jina ColBERT Client (for multi-vector embeddings)
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
 * Generate multi-vector (token-level) embeddings using Jina ColBERT v2.
 *
 * Unlike dense embeddings which produce ONE vector per text,
 * ColBERT produces MANY vectors - one for each token.
 *
 * @param texts - Array of texts to embed
 * @param inputType - "document" for indexing, "query" for searching
 * @returns Array of token embeddings for each input text
 */
export async function getColBERTEmbeddings(
  texts: string[],
  inputType: "document" | "query"
): Promise<number[][][]> {
  const response = await fetch("https://api.jina.ai/v1/multi-vector", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${env.JINA_API_KEY}`,
    },
    body: JSON.stringify({
      model: "jina-colbert-v2",
      dimensions: 128,
      input_type: inputType,
      embedding_type: "float",
      input: texts,
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Jina API error: ${response.status} - ${error}`);
  }

  const data = (await response.json()) as JinaMultiVectorResponse;
  return data.data.map((item) => item.embeddings);
}
