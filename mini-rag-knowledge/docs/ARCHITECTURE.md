# Architecture — mini-rag-knowledge

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG System Pipeline                       │
│                                                                   │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐ │
│  │ Document │──▶│ Chunking │──▶│ Embedding│──▶│  Vector Index │ │
│  │  Input   │   │ Strategy │   │  Model   │   │  (Flat/Cosine)│ │
│  └──────────┘   └──────────┘   └──────────┘   └──────┬───────┘ │
│                                                        │          │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐          │          │
│  │  User    │──▶│  Query   │──▶│  Vector  │──────────┘          │
│  │  Query   │   │ Embedding│   │  Search  │                      │
│  └──────────┘   └──────────┘   └────┬─────┘                     │
│                                      │                            │
│  ┌──────────┐   ┌──────────┐   ┌────▼─────┐   ┌──────────────┐ │
│  │  Hybrid  │◀──│ Reranking│◀──│ Top-K     │◀──│  Retrieve    │ │
│  │  Search  │   │  (MMR)   │   │ Candidates│   │  Candidates  │ │
│  └────┬─────┘   └──────────┘   └──────────┘   └──────────────┘ │
│       │                                                           │
│  ┌────▼─────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐ │
│  │  Prompt  │──▶│   LLM    │──▶│  Guard   │──▶│   Response   │ │
│  │  Builder │   │Generate  │   │  Rails   │   │   Output     │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Module Architecture

### 1. embedding_retrieve (RAG Pipeline Core)

**Role**: Orchestrates the full RAG pipeline from document ingestion to retrieval.

**Data Flow**:
```
Document → ChunkList → [EmbeddingModel] → embedding[] → VectorIndex
                                                             │
Query → [EmbeddingModel] → query_emb → index_search() → RetrieveResult
                                                             │
                    rag_build_prompt() ◀──────────────────────┘
```

**Key Design Decisions**:
- Embedding model is simulated via projection-based encoding (no real transformer).
- Vector index uses flat exhaustive search (brute-force L2 or cosine).
- All operations are in-memory for simplicity.

### 2. chunking_strategy (Document Splitting)

**Role**: Split documents into overlapping text chunks for embedding.

**Strategies**:
| Strategy | Algorithm | Best For |
|----------|-----------|----------|
| Character Split | Fixed N characters + overlap | Simple, fast |
| Recursive Char | Try paragraph→sentence→word separators | Natural text |
| Token Split | Approximate N tokens + overlap | LLM-aware sizing |
| Semantic Split | Embedding similarity threshold | Topic boundaries |

**Small-to-Big Retrieval**:
```
Index small chunks (e.g. 256 tokens) for precise retrieval
       │
       ▼
Expand to big chunks (e.g. 1024 tokens) for context
```

### 3. reranking_model (Result Refinement)

**Role**: Improve retrieval quality by reordering/supplementing initial results.

**Techniques**:
| Technique | Formula | Purpose |
|-----------|---------|---------|
| Cross-Encoder | sigmoid(query·Wq + doc·Wd → MLP → score) | Deep relevance scoring |
| RRF | Σ 1/(k+rank_i) across lists | Combine BM25+vector ranks |
| MMR | λ·rel - (1-λ)·max_sim | Diversity-aware selection |
| Lost-in-Middle | Interleave best→edges | Mitigate positional bias |

**MMR Algorithm** (greedy):
```
S = {}                            // selected set
for step in 1..k:
    best = argmax_i [λ·sim(q,d_i) - (1-λ)·max_{j∈S} sim(d_i,d_j)]
    S = S ∪ {best}
```

### 4. hybrid_search (Dense + Sparse Fusion)

**Role**: Combine complementary retrieval signals for better recall.

**Fusion Methods**:
| Method | Formula | Characteristics |
|--------|---------|----------------|
| Weighted Sum | α·score_dense + (1-α)·score_sparse | Simple, tunable |
| RRF | 1/(k+rank_dense) + 1/(k+rank_sparse) | Rank-based, robust |
| ColBERT | Σ_{q_tokens} max_{d_tokens} cos(q_i, d_j) | Late interaction |

**BM25 Implementation**:
```
BM25(q, d) = Σ IDF(q_i) · (tf·(k1+1)) / (tf + k1·(1-b+b·dl/avgdl))
IDF(q_i) = ln((N - df + 0.5) / (df + 0.5) + 1)
```

**Query Expansion**:
- Generate multiple query variants
- HyDE: embed a hypothetical answer document, search with that embedding

### 5. guardrails_gen (Safety Layer)

**Role**: Three-layer safety filter around RAG pipeline.

**Guard Chain**:
```
User Input
    │
    ▼
┌─────────────┐
│ Input Guard  │ ← Injection detection (pattern matching)
│              │ ← Toxicity check (keyword-based)
│              │ ← PII detection (regex patterns)
└──────┬──────┘
       │ (if passed)
       ▼
┌─────────────┐
│ RAG Pipeline │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│Context Guard │ ← Hallucination detection (overlap NLI)
│              │ ← Fact verification against sources
└──────┬──────┘
       │
       ▼
┌─────────────┐
│Output Guard  │ ← NLI factual consistency
│              │ ← Harmful content refusal
│              │ ← Citation completeness check
└──────┬──────┘
       │
       ▼
   Response
```

**NLI (Natural Language Inference)**: Simple overlap-based implementation that classifies (premise, hypothesis) pairs as ENTAILMENT, CONTRADICTION, or NEUTRAL. Character overlap ratio serves as a proxy for semantic entailment.

**PII Detection**: Pattern-matching for emails (@-based), phone numbers (digit runs), and credit cards (13-19 digit sequences).

## Memory Model

All components use heap allocation with explicit create/destroy lifecycle:

```
rag_create() ──▶ allocates RagContext, EmbeddingModel, VectorIndex
      │
rag_add_document() ──▶ copies document text (strdup)
      │
rag_chunk_documents() ──▶ allocates Chunk.text buffers
      │
rag_embed_chunks() ──▶ allocates Chunk.embedding vectors
      │
rag_build_index() ──▶ copies vectors into index
      │
   ... use ...
      │
rag_destroy() ──▶ frees all allocated memory recursively
```

## Build System

```
Makefile
  ├── all      → libminirag.a (static library)
  ├── examples → example_basic_rag, example_hybrid_search, example_guardrails
  ├── demos    → demo_rag_pipeline, demo_full_system
  └── test     → test suite
```

## Dependencies

- C99 standard library only (`stdlib.h`, `stdio.h`, `string.h`, `math.h`)
- No external libraries required
- Compiles with `-std=c99 -lm`
