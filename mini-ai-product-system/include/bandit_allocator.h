#ifndef BANDIT_ALLOCATOR_H
#define BANDIT_ALLOCATOR_H

/*
 * bandit_allocator.h — Multi-Armed Bandit for Dynamic Model Selection
 *
 * L8 Advanced Topic: Online learning algorithms for explore-exploit
 * tradeoff in production model selection. Replaces static A/B testing
 * with adaptive traffic allocation that minimizes regret.
 *
 * Knowledge Coverage:
 *   L4: Lai-Robbins Asymptotic Lower Bound on Regret (Ω(log T))
 *   L5: UCB1 (Auer et al. 2002), Thompson Sampling (Beta-Bernoulli),
 *       Epsilon-Greedy with Decay, Contextual LinUCB
 *   L8: Regret-minimizing dynamic model selection
 *   L7: Application — Real-time traffic allocation for LLM variants
 *
 * Theorem: For K arms with unknown reward distributions, UCB1 achieves
 * expected regret R_T ≤ O(K log T / Δ_min). Thompson Sampling achieves
 * Bayesian regret O(√(K T log T)).
 *
 * Refs:
 *   - Lai & Robbins (1985): Asymptotically efficient adaptive allocation
 *   - Auer, Cesa-Bianchi, Fischer (2002): Finite-time analysis of UCB
 *   - Chapelle & Li (2011): Empirical evaluation of Thompson sampling
 *   - Li, Chu, Langford, Schapire (2010): Contextual bandit (LinUCB)
 */

#include <stddef.h>
#include <stdint.h>
#include <time.h>

#define BANDIT_MAX_ARMS         16
#define BANDIT_MAX_CONTEXT_DIM  32
#define BANDIT_HISTORY_MAX      10000

/* ── L1: Core Types ── */

/** Bandit arm: one model variant being tested */
typedef struct {
    char    name[64];
    int32_t arm_id;
    int32_t pulls;
    double  sum_rewards;
    double  sum_rewards_sq;
    double  empirical_mean;
    double  empirical_var;
    double  ucb_value;
    double  thompson_alpha;
    double  thompson_beta;
    double  last_reward;
    int64_t last_pull_time;
} BanditArm;

/** Thompson Sampling configuration */
typedef struct {
    double alpha_prior;
    double beta_prior;
} BanditTSConfig;

/** UCB configuration */
typedef struct {
    double exploration_coef;
} BanditUCBConfig;

/** Epsilon-Greedy configuration */
typedef struct {
    double epsilon_start;
    double epsilon_decay;
    double epsilon_min;
} BanditEpsilonConfig;

/** LinUCB context vector */
typedef struct {
    double features[BANDIT_MAX_CONTEXT_DIM];
    int32_t dim;
} BanditContext;

/** LinUCB per-arm parameters (disjoint model) */
typedef struct {
    double A[BANDIT_MAX_CONTEXT_DIM][BANDIT_MAX_CONTEXT_DIM];
    double A_inv[BANDIT_MAX_CONTEXT_DIM][BANDIT_MAX_CONTEXT_DIM];
    double b[BANDIT_MAX_CONTEXT_DIM];
    double theta[BANDIT_MAX_CONTEXT_DIM];
    int    initialized;
    double alpha;  /* regularization for ridge regression */
} LinUCBArm;

/** Main bandit state */
typedef struct {
    BanditArm   arms[BANDIT_MAX_ARMS];
    int32_t     num_arms;
    int32_t     total_pulls;
    double      cumulative_reward;
    double      cumulative_regret;
    int64_t     start_time;

    int         use_ucb;
    int         use_thompson;
    int         use_epsilon;
    int         use_linucb;

    BanditUCBConfig     ucb_cfg;
    BanditTSConfig      ts_cfg;
    BanditEpsilonConfig eps_cfg;

    LinUCBArm   linucb_arms[BANDIT_MAX_ARMS];
    BanditContext current_context;

    double  regret_history[BANDIT_HISTORY_MAX];
    int32_t regret_len;
} BanditAllocator;

/* ── L1: API Declarations ── */

void bandit_init(BanditAllocator *ba);
void bandit_add_arm(BanditAllocator *ba, const char *name, int32_t arm_id);
void bandit_configure_ucb(BanditAllocator *ba, double exploration_coef);
void bandit_configure_thompson(BanditAllocator *ba, double alpha_prior, double beta_prior);
void bandit_configure_epsilon(BanditAllocator *ba, double start, double decay, double min_eps);
void bandit_set_context(BanditAllocator *ba, const double *features, int32_t dim);

/* L5: Arm selection strategies */
int  bandit_select_ucb(BanditAllocator *ba);
int  bandit_select_thompson(BanditAllocator *ba);
int  bandit_select_epsilon_greedy(BanditAllocator *ba);
int  bandit_select_linucb(BanditAllocator *ba);

void bandit_update(BanditAllocator *ba, int32_t arm_idx, double reward);

/* L4: Regret computation & theoretical bounds */
double bandit_cumulative_regret(const BanditAllocator *ba);
double bandit_ucb_regret_bound(const BanditAllocator *ba);
double bandit_lai_robbins_lower_bound(const BanditAllocator *ba);

/* L8: Bayesian decision theory & expected improvement */
double bandit_expected_improvement(const BanditAllocator *ba, int32_t arm_idx);
double bandit_prob_superiority_wrapper(const BanditAllocator *ba,
                                        int32_t arm_a, int32_t arm_b, uint64_t seed);

/* Statistics */
void   bandit_get_arm_stats(const BanditAllocator *ba, int32_t arm_idx,
                            double *mean, double *std_err, int32_t *pulls);
int    bandit_best_arm(const BanditAllocator *ba);
double bandit_empirical_gap(const BanditAllocator *ba, int32_t arm_i, int32_t arm_j);
void   bandit_allocation_summary(const BanditAllocator *ba, double *allocations);

/* L7 Application: Dynamic traffic routing for LLM variants */
int    bandit_route_request(BanditAllocator *ba, int32_t user_id, uint64_t seed);
void   bandit_route_batch(BanditAllocator *ba, int32_t *user_ids, int32_t n,
                          uint64_t seed, int32_t *assignments);

/* Persistence / reporting */
void   bandit_export_csv(const BanditAllocator *ba, char *buf, int32_t max_len);
void   bandit_reset_arms(BanditAllocator *ba);

#endif /* BANDIT_ALLOCATOR_H */
