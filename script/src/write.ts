import {
  DOCUMENTS,
  tpuf,
  DENSE_NS,
  LATE_INTERACTION_NS,
  getDenseEmbedding,
  getColBERTEmbeddings,
} from "./shared";

// ============================================================================
// Dense Vector Indexing (Traditional)
// ============================================================================

/**
 * Index documents using traditional dense vectors.
 *
 * Each document is converted to a SINGLE vector that represents
 * the entire document's meaning.
 */
export async function indexDocumentsDense() {
  const ns = tpuf.namespace(DENSE_NS);

  // Generate one embedding per document
  const rows = await Promise.all(
    DOCUMENTS.map(async (doc) => ({
      id: doc.id,
      vector: await getDenseEmbedding(doc.text),
      text: doc.text,
    }))
  );

  await ns.write({
    upsert_rows: rows,
    distance_metric: "cosine_distance",
    schema: {
      text: { type: "string" },
    },
  });

  console.log(`Indexed ${rows.length} documents with dense vectors`);
}

// ============================================================================
// Late Interaction Indexing (ColBERT)
// ============================================================================

/**
 * Index documents using late interaction (ColBERT multi-vectors).
 *
 * Each document is converted to MULTIPLE vectors - one per token.
 * This preserves fine-grained semantic information that would be
 * lost when compressing to a single vector.
 *
 * Storage schema:
 * - Each token embedding is stored as a separate row
 * - Rows are linked back to their source document via `doc_id`
 */
export async function indexDocumentsLateInteraction() {
  const ns = tpuf.namespace(LATE_INTERACTION_NS);

  // Get multi-vector embeddings for all documents
  const docTexts = DOCUMENTS.map((d) => d.text);
  const allDocEmbeddings = await getColBERTEmbeddings(docTexts, "document");

  console.log(
    `Token counts per document: ${allDocEmbeddings.map((e) => e.length)}`
  );

  // Flatten: one row per token embedding, linked to source document
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

  await ns.write({
    upsert_rows: rows,
    distance_metric: "cosine_distance",
    schema: {
      doc_id: { type: "string" },
      doc_text: { type: "string" },
      token_index: { type: "uint" },
    },
  });

  console.log(
    `Indexed ${DOCUMENTS.length} documents as ${rows.length} token embeddings`
  );
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  console.log("=== Indexing with Dense Vectors ===");
  await indexDocumentsDense();

  console.log("\n=== Indexing with Late Interaction ===");
  await indexDocumentsLateInteraction();
}

main().catch(console.error);
