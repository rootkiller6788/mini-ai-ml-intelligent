# mini-ai-product-system — AI Product System (C99)

> C99 implementation of production AI systems: recommendation engine,
> ChatBot RLHF pipeline, Copilot context, evaluation/monitoring,
> A/B testing, and multi-armed bandit model selection.

---

## Module Status: COMPLETE ✅

| Criterion | Status | Detail |
|-----------|--------|--------|
| Lines (include/ + src/) | **PASS** | 3534 ≥ 3000 |
| `make test` | **PASS** | 13/13 tests passing |
| L1-L6 Knowledge | **Complete** | All covered |
| L7 Applications | **Complete** | 3+ applications |
| L8 Advanced Topics | **Complete** | 3 advanced topics |
| L9 Industry Frontiers | **Partial** | Documented |

---

## Nine-Layer Knowledge Coverage

### L1: Core Definitions (Complete)
- `RecEmbedding`, `RecCandidate`, `RecUserTower`, `RecItemTower` — Two-tower recall
- `RecRankingDNN`, `RecRecallChannel` — Multi-channel recall + ranking
- `CBBaseModel`, `CBRewardModel`, `CBPPOConfig`, `CBDPOConfig` — ChatBot RLHF
- `CCContextState`, `CCPromptMessage`, `CCUserRequest` — Copilot context
- `EVGoldenDataset`, `EVDriftReport`, `EVDashboard` — Evaluation/monitoring
- `MABExperiment`, `MABVariantMetrics`, `MABGuardrails` — A/B testing
- `BanditAllocator`, `BanditArm`, `LinUCBArm` — Multi-armed bandit

### L2: Core Concepts (Complete)
- Two-tower DNN for recommendation (user/item embeddings)
- RLHF pipeline: SFT → Reward Model → PPO/DPO → Safety alignment
- Copilot context gathering, ranking, and prompt construction
- Classification + NLG (BLEU/ROUGE) evaluation metrics
- Online A/B testing with t-test, chi-square, guardrail checks
- Explore-exploit tradeoff in traffic allocation (bandit)

### L3: Engineering Structures (Complete)
- Multi-channel recall pipeline: COLLAB + ANN + POPULAR + RANDOM
- PPO with GAE advantage estimation and KL penalty
- Copilot context ranking via snippet relevance scoring
- Data drift detection with PSI and KS statistics
- Model lifecycle: INIT → DEV → STAGING → CANARY → PRODUCTION → ARCHIVED
- Bandit state machine: UCB / Thompson / Epsilon-Greedy / LinUCB

### L4: Standards/Theorems (Complete)
- **UCB Regret Bound** (Auer et al. 2002): R_T ≤ 8 Σ (ln T / Δ_k) + (1+π²/3) Σ Δ_k
- **Lai-Robbins Lower Bound** (1985): lim inf E[N_k]/log T ≥ 1/KL(ν_k || ν*)
- **Thompson Sampling Bayesian Regret**: O(√(K T log T))
- **Welch's t-test**: independent two-sample with unequal variance
- **PSI (Population Stability Index)**: Σ (P_i - Q_i) · ln(P_i / Q_i)
- **Kolmogorov-Smirnov test**: D = max |F_ref(x) - F_cur(x)|

### L5: Algorithms/Methods (Complete)
- UCB1 arm selection with exploration coefficient c·√(2 ln t / N_k)
- Thompson Sampling via Gamma-distributed Beta samples
- Epsilon-Greedy with multiplicative decay schedule
- LinUCB with Cholesky-decomposed ridge regression
- Collaborative filtering via dot-product scoring
- Ranking DNN with ReLU activations and sigmoid output
- PPO policy gradient with clipped surrogate objective
- DPO (Direct Preference Optimization) loss: -ln σ(β(log π_w - log π_l))
- GAE (Generalized Advantage Estimation): δ_t + γλ · A_{t+1}
- BLEU-N and ROUGE-N/L n-gram overlap metrics

### L6: Canonical Problems (Complete)
- Recommendation system with recall → rank → rerank pipeline (`examples/`)
- RLHF pipeline for chatbot alignment (`examples/example_chatbot.c`)
- Copilot inline code completion (`examples/example_copilot.c`)
- End-to-end AI product pipeline (`examples/demo_full_pipeline.c`)
- Offline evaluation with golden dataset + drift monitoring

### L7: Applications (Complete — 3)
1. **Dynamic LLM traffic routing** — `bandit_route_request()` allocates users to best model variant adaptively
2. **Real-time A/B experiment** — `mab_traffic_assign()` with hash-based deterministic bucketing
3. **Online dashboard** — `ev_dashboard_render()` renders live evaluation metrics

### L8: Advanced Topics (Complete — 3)
1. **Multi-Armed Bandit** — UCB1, Thompson Sampling, LinUCB for continuous model optimization
2. **Bayesian Probability of Superiority** — P(μ_a > μ_b | data) via Monte Carlo Beta posterior sampling
3. **Expected Improvement** — Bayesian optimization criterion for model selection

### L9: Industry Frontiers (Partial)
- AI Compiler integration (Triton/MLIR) for model serving — documented
- Confidential computing for model inference — documented
- Online continual RLHF with streaming human feedback — documented

---

## Core Theorems

| Theorem | Formula | Code Location |
|---------|---------|---------------|
| UCB Regret Bound | E[R_T] ≤ 8 Σ (ln T / Δ_k) + (1+π²/3) Σ Δ_k | `bandit_ucb_regret_bound()` |
| Lai-Robbins Lower Bound | lim inf E[N_k]/log T ≥ 1/KL(ν_k\|\|ν*) | `bandit_lai_robbins_lower_bound()` |
| GAE | A_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l} | `cb_ppo_advantage()` |
| PSI | Σ (P_i - Q_i) · ln(P_i / Q_i) | `ev_psi_compute()` |
| Welch's t-test | t = (X̄₁ - X̄₂) / √(s₁²/n₁ + s₂²/n₂) | `mab_ttest_independent()` |

## Core Algorithms

| Algorithm | Complexity | Implementation |
|-----------|-----------|----------------|
| UCB1 | O(K log T) regret | `bandit_select_ucb()` |
| Thompson Sampling | O(√(KT log T)) Bayesian regret | `bandit_select_thompson()` |
| LinUCB | O(Kd²) per round | `bandit_select_linucb()` |
| ANN Search (brute-force) | O(N·d) | `rec_ann_search()` |
| PPO Clip | O(batch·seq·d²) | `cb_ppo_update()` |
| Chi-Square Test | O(1) | `mab_chi_square_test()` |

---

## Nine-School Course Alignment

| School | Course | Coverage |
|--------|--------|----------|
| **MIT** | 6.858 Computer Security | Safety alignment, guardrails |
| **Stanford** | CS 229 Machine Learning | DNN ranking, RLHF, bandits |
| **Berkeley** | CS 294 AI Systems | Full AI product pipeline |
| **CMU** | 15-445 Database Systems | Recommendation recall/rank |
| **UT Austin** | CS 395T Systems ML | PPO, A/B testing, drift |
| **ETH** | 263-3501 Parallel Programming | Batch routing, LinUCB Cholesky |
| **Cambridge** | Part II: Concurrent Systems | Concurrent traffic routing |
| **清华** | 操作系统/编译原理 | Model lifecycle management |
| **Georgia Tech** | CS 7641 Machine Learning | RLHF, bandit algorithms |

---

## Build & Test

```bash
make all          # Build all targets
make test         # Run 13 unit tests (must pass)
make run-demo     # Run AI product demo
make run-pipeline # Run full pipeline demo
make run-examples # Run recommendation, chatbot, copilot examples
make clean        # Clean build artifacts
```

## Dependencies

- C99 compiler (gcc ≥ 9 or clang ≥ 10)
- libm (math library)
- Standard C library

## Module Files

| Header | Source | Lines | Description |
|--------|--------|-------|-------------|
| `recommendation.h` | `recommendation.c` | 530 | Two-tower recall, ranking DNN, cold start |
| `chatbot_rlhf.h` | `chatbot_rlhf.c` | 591 | SFT, RM, PPO/DPO, safety alignment |
| `copilot_context.h` | `copilot_context.c` | 450 | Context gathering, prompt construction |
| `eval_monitor.h` | `eval_monitor.c` | 602 | Classification/NLG eval, drift, dashboard |
| `model_ab_test.h` | `model_ab_test.c` | 615 | A/B test, t-test, chi-square, ramp-up |
| `bandit_allocator.h` | `bandit_allocator.c` | 746 | UCB, Thompson, Epsilon-Greedy, LinUCB |
| **Total** | | **3534** | |

## License

MIT
