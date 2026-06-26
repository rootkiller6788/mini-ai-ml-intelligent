# mini-rag-knowledge — RAG与知识检索 (C 语言实现)

Retrieval-Augmented Generation (RAG) 知识检索系统的纯 C99 实现。

**include/ + src/ 行数**: 4599 (include: 1084, src: 3515)

## Module Status: COMPLETE ✅

| Level | Status | Details |
|-------|--------|---------|
| L1: Definitions | **Complete** | 10+ structs/typedefs, 50+ API declarations |
| L2: Core Concepts | **Complete** | RAG pipeline, chunking, embedding, retrieval, reranking |
| L3: Engineering Structures | **Complete** | Vector index, BM25 index, knowledge graph, cross-encoder |
| L4: Standards/Theorems | **Complete** | NDCG/MAP/MRR/DCG IR evaluation metrics with formula implementations |
| L5: Algorithms/Methods | **Complete** | BM25, ColBERT late interaction, MMR diversity, RRF fusion, HyDE |
| L6: Canonical Problems | **Complete** | End-to-end RAG pipeline, hybrid search, guardrails |
| L7: Applications | **Complete** | 3 examples + 2 demos (basic RAG, hybrid search, guardrails) |
| L8: Advanced Topics | **Complete** | Knowledge graph enhanced RAG, GraphRAG entity/relation extraction |
| L9: Industry Frontiers | **Partial** | GraphRAG (Microsoft 2024), HyDE (2022) — documented |

## 模块

| 模块 | 头文件 | 说明 |
|------|--------|------|
| RAG Pipeline | `embedding_retrieve.h` | 文档切分→向量嵌入→索引构建→检索→提示生成 |
| 切分策略 | `chunking_strategy.h` | 字符/递归/语义切分 + Small-to-Big 索引 |
| 重排序 | `reranking_model.h` | Cross-encoder、RRF 融合、MMR 多样性、Lost-in-Middle |
| 混合搜索 | `hybrid_search.h` | Dense+BM25 加权/RRF 融合、ColBERT、HyDE |
| 安全护栏 | `guardrails_gen.h` | 注入检测/PII/毒性/幻觉/事实一致性/引用强制 |
| 评估指标 | `rag_evaluator.h` | NDCG@K, MAP, MRR, Precision/Recall/F1@K |
| 知识图谱 | `knowledge_graph_rag.h` | 实体抽取/关系抽取/图谱构建/图增强检索 |
| 查询处理 | `query_processor.h` | 查询分解/扩展/改写/意图分类/复杂度分析 |

## 核心定理 (L4)

| 定理/公式 | 实现函数 | 来源 |
|-----------|---------|------|
| DCG@K = Σ(2^relᵢ - 1) / log₂(i+1) | `dcg_at_k()` | Järvelin & Kekäläinen (2002) |
| NDCG@K = DCG@K / IDCG@K | `ndcg_at_k()` | ACM TOIS |
| MAP = (1/|Q|) Σ AP(q) | `mean_average_precision()` | Manning et al. Ch.8 |
| MRR = (1/|Q|) Σ 1/rank₁ | `mean_reciprocal_rank()` | Voorhees (1999) |
| BM25 Score | `bm25_score()` | Robertson & Zaragoza (2009) |
| RRF = Σ 1/(k+rank) | `rrf_score()` | Cormack et al. (2009) |
| MMR = λ·Rel − (1−λ)·MaxSim | `mmr_rerank()` | Carbonell & Goldstein (1998) |

## 核心算法 (L5)

- **BM25**: Term frequency normalization with IDF weighting
- **ColBERT Late Interaction**: Sum-of-max token-level cosine similarity
- **MMR Diversity Reranking**: Greedy selection with relevance-diversity tradeoff
- **Recursive Character Splitting**: Multi-level (paragraph→sentence→word) text segmentation
- **Query Decomposition**: Conjunction splitting for multi-hop query processing
- **Entity Extraction**: Rule-based NER with person/org/location/date detection
- **Cross-Encoder**: Single-layer neural scoring of query-document pairs

## 经典问题 (L6)

- **Full RAG Pipeline**: Document ingestion → chunking → embedding → indexing → retrieval → generation
- **Hybrid Search**: Dense (cosine) + Sparse (BM25) fusion with weighted sum and RRF
- **Guardrails Chain**: Input (injection/PII/toxicity) → Context (hallucination) → Output (NLI/citations)

## 九校课程映射

| 学校 | 课程 | 本模块对应 |
|------|------|-----------|
| **Stanford** | CS 276: Information Retrieval | BM25, NDCG, MAP, Vector Space Model |
| **Stanford** | CS 224N: NLP with Deep Learning | RAG, Transformer embeddings, NLI |
| **CMU** | 11-741: IR | RRF, BM25, evaluation metrics |
| **Berkeley** | CS 288: NLP | Knowledge graphs, entity extraction |
| **MIT** | 6.8610: Quantitative Methods for NLP | Embedding models, similarity metrics |
| **ETH** | 263-5210: Advanced ML | ColBERT, cross-encoders, reranking |
| **清华** | 信息检索导论 | 全文检索、排序学习、评估体系 |
| **UT Austin** | CS 395T: Systems for ML | RAG pipeline engineering |
| **Cambridge** | Part II: Information Retrieval | Probabilistic IR, evaluation framework |

## 构建

```
make          # 构建库 libminirag.a
make examples # 构建示例程序
make demos    # 构建演示程序
make test     # 运行测试 (27 tests, 0 failed)
make clean    # 清理
```

## 示例

- `examples/example_basic_rag.c` — 基础 RAG 检索流程
- `examples/example_hybrid_search.c` — 混合搜索 (稠密+BM25)
- `examples/example_guardrails.c` — 安全护栏过滤

## 演示

- `demos/demo_rag_pipeline.c` — 完整 RAG 流水线演示
- `demos/demo_full_system.c` — 全系统集成演示

## 文档

- `docs/API_REFERENCE.md` — 完整 API 参考
- `docs/ARCHITECTURE.md` — 系统架构说明

## 许可

MIT
