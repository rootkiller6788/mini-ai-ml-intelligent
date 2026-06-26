# API Reference — mini-rag-knowledge

## embedding_retrieve.h — RAG Pipeline

### Types

| Type | Description |
|------|-------------|
| `RagConfig` | Pipeline configuration (embedding dim, chunk size, overlap, top-k) |
| `Document` | Input document with text, source, title, page metadata |
| `Chunk` | Text chunk with embedding and positional metadata |
| `RetrieveResult` | Top-k retrieval results with indices and scores |
| `EmbeddingModel` | BERT-like encoder simulation (projection-based) |
| `VectorIndex` | Flat-L2 or cosine vector index |
| `RagContext` | Complete RAG pipeline state |

### Functions

| Function | Description |
|----------|-------------|
| `rag_create(cfg)` | Create RAG context with configuration |
| `rag_destroy(r)` | Free all RAG resources |
| `rag_add_document(r, text, len, src, title, page)` | Add a document |
| `rag_add_documents(r, docs, n)` | Add document batch |
| `rag_chunk_documents(r)` | Split all documents into chunks |
| `rag_embed_chunks(r)` | Generate embeddings for all chunks |
| `rag_build_index(r)` | Build vector index from embeddings |
| `rag_retrieve(r, query, len)` | Retrieve top-k chunks for query |
| `rag_build_prompt(r, result, query, sys)` | Construct LLM prompt with context |
| `embed_model_create(dim, type)` | Create embedding model |
| `embed_encode(m, text, len, out)` | Encode text to embedding vector |
| `index_create(cap, dim, type)` | Create vector index |
| `index_add(idx, vec, id)` | Add vector to index |
| `index_search(idx, query, k)` | Search top-k nearest neighbors |

### Vector Utilities

| Function | Description |
|----------|-------------|
| `vec_dot(a, b, dim)` | Dot product |
| `vec_l2_dist(a, b, dim)` | Euclidean distance |
| `vec_cosine_sim(a, b, dim)` | Cosine similarity |
| `vec_normalize(v, dim)` | L2 normalize in-place |

---

## chunking_strategy.h — Document Chunking

### Types

| Type | Description |
|------|-------------|
| `ChunkStrategy` | Enum: CHAR, RECURSIVE_CHAR, TOKEN, SEMANTIC |
| `ChunkConfig` | Strategy selection + size/overlap parameters |
| `ChunkMetadata` | Source, title, page, positional info |
| `TextChunk` | Text slice with metadata |
| `ChunkList` | Dynamic array of chunks |
| `SmallBigIndex` | Small-to-big retrieval mapping |

### Functions

| Function | Description |
|----------|-------------|
| `chunklist_create(cap)` | Create empty chunk list |
| `chunklist_destroy(cl)` | Free chunk list |
| `chunk_split_characters(text, len, size, overlap)` | Fixed-size character split |
| `chunk_split_recursive(text, len, size, overlap)` | Recursive separator-priority split |
| `chunk_split_tokens(text, len, tokens, overlap)` | Token-count-based split |
| `chunk_split_semantic(text, len, embs, ns, dim, thresh)` | Embedding similarity split |
| `chunk_with_metadata(base, meta)` | Attach metadata to all chunks |
| `smallbig_create(big, n, size, overlap)` | Build small-to-big index |
| `smallbig_expand(sbi, id)` | Get big chunk from small chunk ID |
| `smallbig_destroy(sbi)` | Free small-to-big index |
| `count_tokens_approx(text, len)` | Approximate token count |
| `count_sentences(text, len)` | Count sentence boundaries |

---

## reranking_model.h — Reranking

### Types

| Type | Description |
|------|-------------|
| `RerankStrategy` | Enum: CROSS_ENCODER, RRF, MMR, COMBINED |
| `RerankConfig` | Reranking parameters (k, lambda, lost-in-middle) |
| `CrossEncoder` | Simple cross-encoder model (query+doc → score) |
| `RerankCandidate` | Document candidate with BM25/vector/final scores |
| `CandidatePool` | Pool of rerank candidates |

### Functions

| Function | Description |
|----------|-------------|
| `candidate_pool_create(cap)` | Create candidate pool |
| `candidate_pool_add(pool, text, id, bm25, vec)` | Add candidate |
| `cross_encoder_create(emb_dim, hidden_dim)` | Create cross-encoder |
| `cross_encoder_score(ce, q_emb, d_emb)` | Score query-doc pair |
| `rrf_score(dense_rank, sparse_rank, k)` | Compute RRF score |
| `rrf_fuse(pool, dr, sr, k)` | Fuse rankings via RRF |
| `mmr_rerank(pool, q_emb, embs, dim, lambda, k)` | MMR diversity rerank |
| `lost_in_middle_reorder(pool, k)` | Reorder best docs to edges |
| `rerank_pipeline(pool, cfg, ce, q_emb, embs, dim)` | Full reranking pipeline |

---

## hybrid_search.h — Hybrid Search

### Types

| Type | Description |
|------|-------------|
| `HybridFusion` | Enum: WEIGHTED_SUM, RRF, COLBERT, SCORE_COMBINE |
| `HybridConfig` | Fusion parameters (alpha, RRF k, top-k) |
| `SparseVector` | Sparse vector for BM25 (indices + values) |
| `BM25Index` | BM25 document index with IDF/TF |
| `HybridResult` | Combined result (dense+sparse scores+ranks) |
| `ColBERTRepr` | Multi-vector token-level representation |
| `MultiModalQuery` | Text+image embedding fusion |

### Functions

| Function | Description |
|----------|-------------|
| `bm25_create(cap)` | Create BM25 index |
| `bm25_add_document(idx, text, len)` | Add document to BM25 index |
| `bm25_build(idx)` | Build IDF statistics |
| `bm25_score(idx, id, terms, n)` | Compute BM25 score |
| `dense_search(embs, query, dim, n, k)` | Pure vector search |
| `hybrid_weighted_sum(...)` | Alpha-weighted dense+sparse fusion |
| `hybrid_rrf(...)` | RRF-based fusion |
| `colbert_late_interaction(qr, dr)` | ColBERT max-sim score |
| `query_expand(query, model, n)` | Generate query variants |
| `hyde_generate(query, dim)` | HyDE hypothetical document embedding |
| `multimodal_create(dim)` | Create multi-modal query |
| `multimodal_fuse(mq, text_weight)` | Fuse text+image embeddings |

---

## guardrails_gen.h — Safety Guardrails

### Types

| Type | Description |
|------|-------------|
| `GuardType` | Bitmask: INJECTION, TOXICITY, PII, HALLUCIN, NLI, REFUSAL, CITATION |
| `Severity` | Enum: SAFE, LOW, MEDIUM, HIGH, CRITICAL |
| `GuardViolation` | Violation record (type, severity, position, confidence) |
| `GuardResult` | Collection of violations with pass/fail status |
| `NLILabel` | Enum: ENTAILMENT, CONTRADICTION, NEUTRAL |
| `Citation` | Source reference with excerpt |
| `GuardContext` | Pipeline state (query, docs, generated text) |
| `GuardConfig` | Guard activation flags and thresholds |

### Functions

| Function | Description |
|----------|-------------|
| `guard_check_injection(input, len)` | Detect prompt injection patterns |
| `guard_check_toxicity(input, len)` | Check for toxic content |
| `guard_check_pii(input, len)` | Detect PII (email, phone, credit card) |
| `guard_check_hallucination(gen, docs, n, threshold)` | Verify output against docs |
| `guard_nli_classify(premise, hyp)` | NLI entailment classification |
| `guard_check_factual_consistency(gen, docs, n)` | Check output supported by context |
| `guard_refuse_harmful(output, len)` | Detect harmful content in output |
| `citation_extract(gen, sources, texts, n)` | Extract [N] citations from text |
| `guard_chain_input(gc, cfg)` | Run all input guards |
| `guard_chain_context(gc, cfg)` | Run all context guards |
| `guard_chain_output(gc, cfg)` | Run all output guards |
| `guard_chain_full(gc, cfg)` | Run complete guard chain |
