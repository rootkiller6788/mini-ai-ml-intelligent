#include "embedding_retrieve.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(void) {
    printf("=== Basic RAG Pipeline Demo ===\n\n");

    RagConfig cfg = {
        .embedding_dim       = 128,
        .chunk_size          = 256,
        .chunk_overlap       = 32,
        .top_k               = 3,
        .similarity_threshold = 0.5f,
        .use_metadata_filter  = false,
        .normalize_embeddings = true,
    };

    RagContext *rag = rag_create(&cfg);

    const char *doc1 =
        "Retrieval-Augmented Generation (RAG) combines information retrieval "
        "with language models to produce accurate and context-aware responses. "
        "The process involves chunking documents into smaller pieces, embedding "
        "them into vector representations, and indexing them for similarity search. "
        "When a user asks a question, the query is embedded and compared against "
        "the index to retrieve the most relevant chunks. These chunks are then "
        "inserted into a prompt template and sent to the language model.";

    const char *doc2 =
        "Vector embeddings are dense numerical representations of text that "
        "capture semantic meaning. Models like BERT and Sentence-Transformers "
        "are commonly used to generate embeddings. The cosine similarity metric "
        "is often used to measure the similarity between two embeddings. "
        "A vector index like FAISS or Annoy enables fast approximate nearest "
        "neighbor search over millions of vectors.";

    const char *doc3 =
        "Document chunking strategies include fixed-size character splitting, "
        "recursive splitting with separator priority (paragraph, sentence, word), "
        "and semantic splitting based on embedding similarity thresholds. "
        "Overlap between chunks helps maintain context across chunk boundaries. "
        "The typical chunk size ranges from 512 to 2048 tokens with 10-20% overlap.";

    rag_add_document(rag, doc1, strlen(doc1), "docs/rag_intro.txt", "RAG Introduction", 1);
    rag_add_document(rag, doc2, strlen(doc2), "docs/embeddings.txt", "Vector Embeddings", 1);
    rag_add_document(rag, doc3, strlen(doc3), "docs/chunking.txt", "Chunking Strategies", 2);

    printf("Added %zu documents\n", rag->num_docs);

    rag_chunk_documents(rag);
    printf("Created %zu chunks\n", rag->num_chunks);

    rag_embed_chunks(rag);
    printf("Embedded %zu chunks (dim=%zu)\n", rag->num_chunks, rag->model->dim);

    rag_build_index(rag);
    printf("Built vector index with %zu vectors\n\n", rag->index->count);

    const char *queries[] = {
        "How does RAG combine retrieval with generation?",
        "What is cosine similarity used for?",
        "What are the different chunking methods?",
    };

    for (int q = 0; q < 3; q++) {
        printf("Query: %s\n", queries[q]);
        RetrieveResult *res = rag_retrieve(rag, queries[q], strlen(queries[q]));
        printf("Top-%zu results:\n", res->count);
        for (size_t i = 0; i < res->count; i++) {
            size_t ci = res->indices[i];
            printf("  [%zu] score=%.4f | %s...\n",
                   i + 1, res->scores[i],
                   rag->chunks[ci].text_len > 60
                       ? (char[61]){0} : rag->chunks[ci].text);
            if (rag->chunks[ci].text_len > 60) {
                char preview[61];
                memcpy(preview, rag->chunks[ci].text, 60);
                preview[60] = '\0';
                printf("    \"%s...\"\n", preview);
            } else {
                printf("    \"%s\"\n", rag->chunks[ci].text);
            }
        }
        printf("\n");
        retrieve_destroy(res);
    }

    rag_destroy(rag);
    printf("Done.\n");
    return 0;
}
