# Architecture — mini-ai-product-system

## Overview

The system implements five core AI product modules in C99:

```
+------------------------------------------------------------------+
|                     AI Product System                             |
+------------------------------------------------------------------+
|  Recommendation  | ChatBot RLHF | Copilot | Eval/Monitor | A/B   |
+------------------------------------------------------------------+
```

## Module Architecture

### 1. Recommendation System (`recommendation.c`)

```
User Features ──> User Tower ──> User Embedding
                                      │
                              Dot Product Scoring
                                      │
Item Features ──> Item Tower ──> Item Embeddings
                                      │
                      ┌───────────────┼───────────────┐
                      │                               │
              Multi-Channel Recall              Ranking DNN
              (Collab/ANN/Popular/            (User+Item+Session
               Random/Content)                  Features)
                      │                               │
                      └───────────────┬───────────────┘
                                      │
                              Re-Ranking Layer
                         (Diversity/Freshness/Business Rules)
                                      │
                              Final Recommendations
```

**Recall Layer**: Uses multiple channels - collaborative filtering (dot product), ANN
approximate search, popularity-based, random exploration, and content-based.

**Ranking Layer**: Deep neural network with user features, item embeddings, and
session context. ReLU activations, sigmoid output.

**Re-Ranking**: Diversity via MMR-like deduplication, freshness time-decay boost,
and business rule filtering (block lists).

**Cold Start**: Content-based feature extraction from title, category, tags.
Similarity scoring against user profile for new items.

### 2. ChatBot RLHF (`chatbot_rlhf.c`)

```
Base Model (Transformer)
      │
      ├──> SFT (Supervised Fine-Tuning)
      │      │
      │      └──> Human demonstrations -> instruction tuning
      │
      ├──> Reward Model Training
      │      │
      │      └──> Preference pairs (chosen vs rejected) -> scalar reward
      │
      ├──> RLHF (PPO)
      │      │
      │      └──> Policy update with KL penalty to reference model
      │           GAE advantage estimation. Clip epsilon for stability.
      │
      ├──> DPO (Direct Preference Optimization)
      │      │
      │      └──> Implicit reward = beta * log(P_model / P_ref)
      │           No separate reward model needed.
      │
      └──> Safety Alignment
             │
             └──> Red-teaming, harmlessness rewards
                  Constitutional AI feedback
```

### 3. Copilot Context (`copilot_context.c`)

```
Context Gathering
      │
      ├──> Open files (+ content, language detection)
      ├──> Cursor position (file, line, col)
      ├──> Recent edits (before/after diffs)
      ├──> Imports (language-specific parsing)
      ├──> Project structure (file tree)
      └──> Git diff (uncommitted changes)
      │
Context Ranking
      │
      └──> Relevance scoring: proximity to cursor + recency
      │
Prompt Construction
      │
      ├──> System message: role + capabilities
      ├──> Context block: language, cursor, files, diffs
      └──> User intent: complete/explain/fix/refactor/generate/chat
      │
Completion Generation
      │
      ├──> Single/multi-line code generation
      └──> Post-processing: trim whitespace, filter incomplete tokens
```

### 4. Evaluation & Monitoring (`eval_monitor.c`)

```
Offline Evaluation
      │
      ├──> Classification: accuracy, precision, recall, F1, confusion matrix
      ├──> NLG: BLEU (1-4 gram), ROUGE (1, 2, L)
      └──> Golden dataset benchmark
      │
Online Evaluation
      │
      ├──> A/B experiment comparison
      ├──> Engagement, CTR, task completion, retention
      └──> Statistical significance testing
      │
LLM-as-Judge
      │
      └──> Rate helpfulness, correctness, safety, coherence
      │
Monitoring
      │
      ├──> Data drift detection (PSI)
      ├──> Prediction drift (KS test)
      ├──> Regression detection (metric drop alerts)
      └──> Dashboard rendering
      │
Feedback
      │
      └──> Thumbs up/down collection and satisfaction summary
```

### 5. Model A/B Testing (`model_ab_test.c`)

```
Experiment Design
      │
      ├──> Variants: control / treatment / shadow
      ├──> Traffic split: percentage-based, user hash bucketing
      └──> Feature flags: user-level rollout control
      │
Metrics Collection
      │
      ├──> Latency, success rate, satisfaction, error rate
      └──> Per-variant aggregation
      │
Statistical Testing
      │
      ├──> t-Test (independent samples, Welch's)
      ├──> Chi-square test (categorical outcomes)
      └──> Sample size estimation
      │
Ramp-Up Process
      │
      ├──> 1% -> 5% -> 25% -> 50% -> 100%
      └──> Guardrail checks at each step
      │
Rollback
      │
      ├──> Auto-detection: latency > 1.5x, error > 2x
      ├──> Guardrail violations
      └──> Instant traffic re-route to control
      │
Model Registry
      │
      ├──> Stages: init -> dev -> staging -> canary -> production -> archived
      └──> Promotion tracking
      │
Shadow Testing
      │
      └──> Dark launch: validate without user impact
```

## Data Flow

```
User Request
      │
      ├──> [Recommendation] Get candidates -> rank -> re-rank -> serve
      ├──> [ChatBot] Tokenize -> generate -> detokenize -> safety filter
      ├──> [Copilot] Gather context -> rank -> prompt -> complete -> post-process
      ├──> [Eval] Collect metrics -> compare -> detect regressions -> alert
      └──> [A/B] Assign variant -> record metrics -> test significance -> ramp/rollback
```

## Memory Model

All allocations use stack/static buffers with compile-time constants (MAX_* macros).
Heap allocation only where dynamic sizing is unavoidable (variable-length arrays).
No external dependencies beyond libc and libm.
