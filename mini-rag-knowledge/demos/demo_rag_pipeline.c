#include "embedding_retrieve.h"
#include "chunking_strategy.h"
#include "reranking_model.h"
#include "hybrid_search.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#define DEMO_EMBED_DIM   128
#define DEMO_NUM_DOCS     10

/* ──────────────────────────────────────────────
   Sample Knowledge Base
   ────────────────────────────────────────────── */
static const char *knowledge_docs[] = {
    "Quantum computing leverages quantum mechanical phenomena such as "
    "superposition and entanglement to perform computations. Unlike classical "
    "bits that are either 0 or 1, quantum bits (qubits) can exist in multiple "
    "states simultaneously. This property enables quantum computers to solve "
    "certain problems exponentially faster than classical computers. Shor's "
    "algorithm for integer factorization and Grover's search algorithm are "
    "two famous quantum algorithms that demonstrate this advantage.",

    "Machine learning is a field of artificial intelligence that focuses on "
    "building systems that learn from data. Supervised learning uses labeled "
    "training data, while unsupervised learning finds patterns in unlabeled "
    "data. Deep learning, a subset of machine learning, uses multi-layered "
    "neural networks to learn hierarchical representations. Backpropagation "
    "and gradient descent are the core optimization algorithms used to train "
    "neural networks by minimizing a loss function.",

    "Large language models (LLMs) like GPT, BERT, and LLaMA are transformer-"
    "based neural networks trained on massive text corpora. The transformer "
    "architecture introduced the self-attention mechanism, which allows the "
    "model to weigh the importance of different words in a sequence. LLMs "
    "exhibit emergent abilities such as few-shot learning, chain-of-thought "
    "reasoning, and instruction following. However, they can hallucinate "
    "facts and require alignment techniques like RLHF to be safe and helpful.",

    "Retrieval-Augmented Generation (RAG) combines information retrieval with "
    "text generation to produce factually grounded responses. The pipeline "
    "involves chunking documents, embedding them into vectors, indexing them, "
    "retrieving relevant chunks for a query, and conditioning a language model "
    "on the retrieved context. RAG reduces hallucination by grounding the "
    "model's output in retrieved evidence. Advanced RAG systems use reranking, "
    "hybrid search, and iterative retrieval for better results.",

    "Vector databases are specialized database systems designed to store and "
    "query high-dimensional vector embeddings. They support similarity search "
    "operations like k-nearest neighbors (k-NN) using distance metrics such "
    "as cosine similarity, Euclidean distance, and dot product. Popular vector "
    "databases include Pinecone, Weaviate, Milvus, Qdrant, and Chroma. They "
    "are essential for semantic search, recommendation systems, and RAG.",

    "The attention mechanism computes a weighted sum of values where the "
    "weights are determined by compatibility between queries and keys. The "
    "scaled dot-product attention is defined as Attention(Q,K,V) = "
    "softmax(QK^T/sqrt(d_k))V. Multi-head attention runs multiple attention "
    "operations in parallel, allowing the model to attend to different "
    "representation subspaces. This is the core innovation behind transformers.",

    "Python has become the dominant programming language for machine learning "
    "and data science, with libraries such as NumPy for numerical computing, "
    "Pandas for data manipulation, Scikit-learn for classical ML algorithms, "
    "PyTorch and TensorFlow for deep learning, and Hugging Face Transformers "
    "for state-of-the-art NLP models.",

    "BM25 (Best Matching 25) is a probabilistic relevance scoring function "
    "used in information retrieval. It ranks documents based on term frequency "
    "(TF), inverse document frequency (IDF), and document length normalization. "
    "The formula is: BM25(q,d) = Σ IDF(q_i) * (tf(k1+1)) / (tf + k1*(1-b+b*dl/avgdl)). "
    "BM25 is widely used as a strong lexical baseline in search systems.",

    "Prompt engineering is the practice of designing effective prompts for "
    "language models. Techniques include few-shot prompting (providing examples), "
    "chain-of-thought (encouraging step-by-step reasoning), role prompting "
    "(assigning a persona to the model), and structured output formatting. "
    "Good prompts are clear, specific, and provide relevant context.",

    "Embeddings are dense vector representations that capture semantic meaning. "
    "Word2Vec and GloVe were early embedding models that learned word-level "
    "representations. BERT and Sentence-BERT produce contextual embeddings "
    "where the same word can have different vectors depending on context. "
    "Embeddings enable semantic similarity computation and are fundamental "
    "to modern NLP systems.",
};

/* ──────────────────────────────────────────────
   Helper: print chunk with truncation
   ────────────────────────────────────────────── */
static void print_preview(const char *text, size_t len, size_t max_chars) {
    size_t n = len < max_chars ? len : max_chars;
    for (size_t i = 0; i < n; i++) {
        putchar(text[i]);
        if (text[i] == '\n') break;
    }
    if (n < len) printf("...");
}

/* ──────────────────────────────────────────────
   Display chunking results
   ────────────────────────────────────────────── */
static void demo_chunking(void) {
    printf("═══════════════════════════════════════\n");
    printf("  PART 1: Document Chunking Strategies\n");
    printf("═══════════════════════════════════════\n\n");

    const char *text = knowledge_docs[0];

    printf("─── Character Split (size=120, overlap=20) ───\n");
    ChunkList *cl = chunk_split_characters(text, strlen(text), 120, 20);
    printf("  Created %zu chunks\n", cl->count);
    for (size_t i = 0; i < cl->count && i < 3; i++) {
        printf("  Chunk %zu [%zu-%zu]: \"", i, cl->chunks[i].meta.start_char,
               cl->chunks[i].meta.end_char);
        print_preview(cl->chunks[i].text, cl->chunks[i].text_len, 80);
        printf("\"\n");
    }
    chunklist_destroy(cl);

    printf("\n─── Recursive Character Split (size=150, overlap=25) ───\n");
    cl = chunk_split_recursive(text, strlen(text), 150, 25);
    printf("  Created %zu chunks\n", cl->count);
    for (size_t i = 0; i < cl->count && i < 3; i++) {
        printf("  Chunk %zu [%zu-%zu]: \"", i, cl->chunks[i].meta.start_char,
               cl->chunks[i].meta.end_char);
        print_preview(cl->chunks[i].text, cl->chunks[i].text_len, 80);
        printf("\"\n");
    }
    chunklist_destroy(cl);

    printf("\n─── Token Split (tokens=30, overlap=5) ───\n");
    cl = chunk_split_tokens(text, strlen(text), 30, 5);
    printf("  Created %zu chunks\n", cl->count);
    for (size_t i = 0; i < cl->count && i < 3; i++) {
        printf("  Chunk %zu [%zu-%zu]: \"", i, cl->chunks[i].meta.start_char,
               cl->chunks[i].meta.end_char);
        print_preview(cl->chunks[i].text, cl->chunks[i].text_len, 80);
        printf("\"\n");
    }
    chunklist_destroy(cl);

    printf("\n─── Small-to-Big Retrieval Demo ───\n");
    TextChunk big_chunk;
    big_chunk.text = (char*)text;
    big_chunk.text_len = strlen(text);
    SmallBigIndex *sbi = smallbig_create(&big_chunk, 1, 80, 15);
    printf("  Big chunks: %zu, Small chunks: %zu\n", sbi->num_big, sbi->num_small);
    printf("  Mappings: %zu\n", sbi->num_mappings);
    for (size_t i = 0; i < sbi->num_mappings && i < 3; i++) {
        printf("  Map[%zu]: small_id=%zu → big_id=%zu, offset=%zu\n",
               i, sbi->mappings[i].small_id, sbi->mappings[i].big_id,
               sbi->mappings[i].offset_in_big);
    }
    TextChunk *expanded = smallbig_expand(sbi, 0);
    if (expanded) {
        printf("  Expand small[0] → big chunk: \"");
        print_preview(expanded->text, expanded->text_len, 60);
        printf("\"\n");
    }
    smallbig_destroy(sbi);
}

/* ──────────────────────────────────────────────
   Display embedding & indexing
   ────────────────────────────────────────────── */
static void demo_embedding(void) {
    printf("\n═══════════════════════════════════════\n");
    printf("  PART 2: Embedding & Indexing\n");
    printf("═══════════════════════════════════════\n\n");

    EmbeddingModel *model = embed_model_create(DEMO_EMBED_DIM, EMBED_SIM_SIMPLE_PROJ);
    printf("  Model: dim=%zu, sim_type=SIMPLE_PROJ\n", model->dim);

    const char *texts[] = {
        "quantum computing qubits superposition",
        "machine learning neural networks",
        "large language models transformers",
    };
    for (int i = 0; i < 3; i++) {
        float *emb = malloc(model->dim * sizeof(float));
        embed_encode(model, texts[i], strlen(texts[i]), emb);
        float norm = 0.0f;
        for (size_t j = 0; j < model->dim; j++) norm += emb[j] * emb[j];
        float s01 = vec_cosine_sim(emb, emb, model->dim);
        printf("  Text %d: \"%s\" → norm=%.4f, self_sim=%.4f\n",
               i, texts[i], sqrtf(norm), s01);
        free(emb);
    }

    printf("\n  Building vector index...\n");
    VectorIndex *idx = index_create(100, model->dim, INDEX_COSINE);
    for (int i = 0; i < DEMO_NUM_DOCS; i++) {
        float *emb = malloc(model->dim * sizeof(float));
        embed_encode(model, knowledge_docs[i], strlen(knowledge_docs[i]), emb);
        index_add(idx, emb, i);
        free(emb);
    }
    index_build(idx);
    printf("  Index contains %zu vectors\n", idx->count);

    float *q_emb = malloc(model->dim * sizeof(float));
    embed_encode(model, "how does transformer attention work",
                 strlen("how does transformer attention work"), q_emb);
    RetrieveResult *res = index_search(idx, q_emb, 5);
    printf("  Query: \"how does transformer attention work\"\n");
    printf("  Top-5 results:\n");
    for (size_t i = 0; i < res->count; i++) {
        size_t di = res->indices[i];
        printf("    [%zu] score=%.4f → doc[%zu]: \"", i + 1, res->scores[i], di);
        print_preview(knowledge_docs[di], strlen(knowledge_docs[di]), 60);
        printf("\"\n");
    }
    retrieve_destroy(res);
    free(q_emb);
    index_destroy(idx);
    embed_model_destroy(model);
}

/* ──────────────────────────────────────────────
   Display reranking
   ────────────────────────────────────────────── */
static void demo_reranking(void) {
    printf("\n═══════════════════════════════════════\n");
    printf("  PART 3: Reranking & MMR Diversity\n");
    printf("═══════════════════════════════════════\n\n");

    size_t dim = DEMO_EMBED_DIM;
    size_t num_candidates = 8;
    float **doc_embs = malloc(num_candidates * sizeof(float*));
    const char *labels[] = {
        "quantum mechanics intro",    "quantum gates circuits",
        "ml basics tutorial",         "deep learning overview",
        "transformer architecture",   "attention mechanism detail",
        "python programming intro",   "numpy pandas tutorial",
    };
    for (size_t i = 0; i < num_candidates; i++) {
        doc_embs[i] = malloc(dim * sizeof(float));
        for (size_t j = 0; j < dim; j++) {
            doc_embs[i][j] = sinf((float)(i * dim + j + 1) * 0.05f) * 0.5f +
                             cosf((float)((i * 13 + j * 7) % dim) * 0.03f) * 0.3f;
        }
        vec_normalize(doc_embs[i], dim);
    }
    float *query_emb = malloc(dim * sizeof(float));
    for (size_t j = 0; j < dim; j++) {
        query_emb[j] = cosf((float)(j * 7 + 3) * 0.04f) * 0.6f +
                       sinf((float)(j * 11 + 5) * 0.02f) * 0.4f;
    }
    vec_normalize(query_emb, dim);

    printf("─── Initial Retrieval (top-%zu) ───\n", num_candidates);
    CandidatePool *pool = candidate_pool_create(num_candidates);
    for (size_t i = 0; i < num_candidates; i++) {
        float vs = vec_cosine_sim(query_emb, doc_embs[i], dim);
        candidate_pool_add(pool, labels[i], i, 0.0f, vs);
        printf("  [%zu] %-25s vector_score=%.4f\n", i + 1, labels[i], vs);
    }

    printf("\n─── MMR Reranking (λ=0.7, k=5) ───\n");
    mmr_rerank(pool, query_emb, (const float**)doc_embs, dim, 0.7f, 5);
    for (size_t i = 0; i < pool->count && i < 5; i++) {
        printf("  [%zu] %-25s selected=%s\n", i + 1,
               pool->candidates[i].doc_text,
               pool->candidates[i].selected ? "yes" : "no");
    }
    candidate_pool_destroy(pool);

    printf("\n─── Cross-Encoder Scoring ───\n");
    CrossEncoder *ce = cross_encoder_create(dim, 32);
    for (size_t i = 0; i < num_candidates; i++) {
        float score = cross_encoder_score(ce, query_emb, doc_embs[i]);
        printf("  [%zu] %-25s CE_score=%.4f\n", i + 1, labels[i], score);
    }
    cross_encoder_destroy(ce);

    printf("\n─── RRF Fusion Demo ───\n");
    CandidatePool *rrf_pool = candidate_pool_create(num_candidates);
    for (size_t i = 0; i < num_candidates; i++) {
        candidate_pool_add(rrf_pool, labels[i], i, (float)(num_candidates - i), 0.0f);
    }
    size_t *dr = malloc(num_candidates * sizeof(size_t));
    size_t *sr = malloc(num_candidates * sizeof(size_t));
    for (size_t i = 0; i < num_candidates; i++) { dr[i] = i; sr[i] = num_candidates - 1 - i; }
    rrf_fuse(rrf_pool, dr, sr, 60.0f);
    printf("  RRF fusion (dense_rank, sparse_rank reversed, k=60):\n");
    for (size_t i = 0; i < rrf_pool->count && i < 5; i++) {
        printf("    [%zu] %-25s rrf=%.4f\n", i + 1,
               rrf_pool->candidates[i].doc_text,
               rrf_pool->candidates[i].final_score);
    }
    free(dr); free(sr);
    candidate_pool_destroy(rrf_pool);

    printf("\n─── Lost-in-the-Middle Reordering ───\n");
    CandidatePool *litm_pool = candidate_pool_create(10);
    for (size_t i = 0; i < 10; i++) {
        char buf[32];
        snprintf(buf, sizeof(buf), "Document-%02zu", i);
        candidate_pool_add(litm_pool, buf, i, 10.0f - (float)i, 0.0f);
        litm_pool->candidates[i].final_score = 10.0f - (float)i;
    }
    printf("  Before: ");
    for (size_t i = 0; i < 5; i++) printf("%s ", litm_pool->candidates[i].doc_text);
    printf("\n");
    lost_in_middle_reorder(litm_pool, 5);
    printf("  After:  ");
    for (size_t i = 0; i < 5; i++) printf("%s ", litm_pool->candidates[i].doc_text);
    printf("\n");
    candidate_pool_destroy(litm_pool);

    free(query_emb);
    for (size_t i = 0; i < num_candidates; i++) free(doc_embs[i]);
    free(doc_embs);
}

/* ──────────────────────────────────────────────
   Display full RAG pipeline
   ────────────────────────────────────────────── */
static void demo_full_rag(void) {
    printf("\n═══════════════════════════════════════\n");
    printf("  PART 4: Full RAG Pipeline\n");
    printf("═══════════════════════════════════════\n\n");

    RagConfig cfg = {
        .embedding_dim       = DEMO_EMBED_DIM,
        .chunk_size          = 150,
        .chunk_overlap       = 20,
        .top_k               = 4,
        .similarity_threshold = 0.3f,
        .use_metadata_filter  = false,
        .normalize_embeddings = true,
    };

    RagContext *rag = rag_create(&cfg);
    for (int i = 0; i < DEMO_NUM_DOCS; i++) {
        char src[32];
        snprintf(src, sizeof(src), "doc_%02d.txt", i);
        rag_add_document(rag, knowledge_docs[i], strlen(knowledge_docs[i]),
                         src, "Knowledge Base", i + 1);
    }
    printf("  Added %zu documents\n", rag->num_docs);

    rag_chunk_documents(rag);
    printf("  Chunked into %zu chunks (size=%zu, overlap=%zu)\n",
           rag->num_chunks, cfg.chunk_size, cfg.chunk_overlap);

    rag_embed_chunks(rag);
    printf("  Embedded %zu chunks (%zu-dim vectors)\n", rag->num_chunks, rag->model->dim);

    rag_build_index(rag);
    printf("  Built index with %zu vectors\n", rag->index->count);

    const char *test_queries[] = {
        "What are qubits and how do they work?",
        "Explain the attention mechanism in transformers",
        "How does BM25 ranking work?",
        "What is the purpose of vector databases?",
    };

    for (int q = 0; q < 4; q++) {
        printf("\n  Query %d: \"%s\"\n", q + 1, test_queries[q]);
        RetrieveResult *res = rag_retrieve(rag, test_queries[q],
                                           strlen(test_queries[q]));
        for (size_t i = 0; i < res->count; i++) {
            size_t ci = res->indices[i];
            Chunk *ch = &rag->chunks[ci];
            printf("    [%zu] score=%.4f doc[%zu] \"", i + 1, res->scores[i],
                   ch->doc_index);
            print_preview(ch->text, ch->text_len, 80);
            printf("\"\n");
        }
        retrieve_destroy(res);
    }

    /* Show prompt construction for first query */
    printf("\n─── Generated Prompt (first query) ───\n");
    RetrieveResult *res = rag_retrieve(rag, test_queries[0], strlen(test_queries[0]));
    char *prompt = rag_build_prompt(rag, res, test_queries[0],
        "You are a helpful AI assistant. Answer based on the provided context only.");
    printf("%s", prompt);
    free(prompt);
    retrieve_destroy(res);

    rag_destroy(rag);
}

/* ──────────────────────────────────────────────
   Main
   ────────────────────────────────────────────── */
int main(void) {
    printf("╔══════════════════════════════════════════╗\n");
    printf("║   mini-rag-knowledge: RAG Pipeline Demo  ║\n");
    printf("╚══════════════════════════════════════════╝\n\n");

    demo_chunking();
    demo_embedding();
    demo_reranking();
    demo_full_rag();

    printf("\n═══════════════════════════════════════\n");
    printf("  All pipeline components demonstrated.\n");
    printf("═══════════════════════════════════════\n");
    return 0;
}
