# mini-rag-knowledge — RAG与知识检索 (C 语言实现)

Retrieval-Augmented Generation (RAG) 知识检索系统的纯 C99 实现。

## 模块

| 模块 | 头文件 | 说明 |
|------|--------|------|
| RAG Pipeline | `embedding_retrieve.h` | 文档切分→向量嵌入→索引构建→检索→提示生成 |
| 切分策略 | `chunking_strategy.h` | 固定/递归/语义切分 + 小到大检索 |
| 重排序 | `reranking_model.h` | Cross-encoder、RRF 融合、MMR 多样性 |
| 混合搜索 | `hybrid_search.h` | 稠密+稀疏融合、ColBERT、HyDE 查询扩展 |
| 安全护栏 | `guardrails_gen.h` | 输入/上下文/输出护栏、幻觉检测、引用强制 |

## 构建

```
make          # 构建库 libminirag.a
make examples # 构建示例程序
make demos    # 构建演示程序
make test     # 运行测试
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
