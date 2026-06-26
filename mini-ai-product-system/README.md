# mini-ai-product-system — AI产品系统 (C 语言实现)

C99 实现的 AI 产品核心系统：推荐引擎、ChatBot RLHF、Copilot 上下文、评估监控、A/B 测试。

## 模块

| 文件 | 功能 |
|------|------|
| `recommendation.h` | 推荐系统：协同过滤、双塔召回、排序、重排、冷启动 |
| `chatbot_rlhf.h` | ChatBot 管线：SFT→RM→RLHF(PPO/DPO)→安全对齐 |
| `copilot_context.h` | Copilot：上下文收集、Prompt 构建、补全后处理 |
| `eval_monitor.h` | 评估监控：离线评估、在线实验、漂移检测、仪表盘 |
| `model_ab_test.h` | A/B 测试：流量分割、指标统计、SRM、渐进放量 |

## 编译运行

```bash
make all
./demo_ai_product
./demo_full_pipeline
./example_recommendation
./example_chatbot
./example_copilot
```

## 依赖

- C99 编译器 (gcc/clang)
- libm (数学库)
- 标准 C 库

## 许可证

MIT
