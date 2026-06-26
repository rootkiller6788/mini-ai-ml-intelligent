/*
 * bandit_allocator.c ? Multi-Armed Bandit Implementation
 *
 * Implements four bandit algorithms:
 *   1. UCB1 (Auer et al. 2002) ? Optimism in the face of uncertainty
 *   2. Thompson Sampling ? Bayesian posterior sampling (Beta-Bernoulli)
 *   3. Epsilon-Greedy ? Simple exploration with decaying epsilon
 *   4. LinUCB (Li et al. 2010) ? Contextual bandit with linear payoffs
 *
 * Each algorithm solves the same fundamental problem: sequentially
 * allocate traffic to K model variants to maximize cumulative reward
 * while minimizing regret.
 *
 * L4 Theorem Verification (UCB Regret Bound):
 *   Let ?_k = ?* - ?_k for suboptimal arm k. UCB1 achieves:
 *     E[R_T] ? 8 ?_{k:?_k>0} (ln T / ?_k) + (1 + ??/3) ?_k ?_k
 *   ? O(K log T) regret, matching the Lai-Robbins lower bound ?(log T).
 *
 * L4 Theorem Verification (Thompson Sampling):
 *   For Beta-Bernoulli bandit with Beta(1,1) prior, the Bayesian regret
 *   of Thompson Sampling satisfies E[R_T] ? O(?(K T log T)).
 *   This is minimax-optimal for K-armed bandits.
 *
 * L4 Theorem Verification (Lai-Robbins Lower Bound):
 *   For any uniformly good policy ? and suboptimal arm k:
 *     lim inf_{T??} E[N_k(T)] / log T ? 1 / KL(?_k || ?*)
 *   where ?* is the optimal arm distribution.
 */

#include "bandit_allocator.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ?? Pseudo-RNG (SplitMix64) ?? */

static uint64_t bandit_splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static double bandit_randf(uint64_t *state) {
    return (double)(bandit_splitmix64(state) & 0xFFFFFFFFFFFFULL) / 281474976710656.0;
}

/* Box-Muller: generate standard normal N(0,1) */
static double bandit_randn(uint64_t *state) {
    double u1 = bandit_randf(state) + 1e-12;
    double u2 = bandit_randf(state);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static double bandit_gamma_sample(double shape, uint64_t *rng);

/*
 * L8: Bayesian probability of superiority P(mu_a > mu_b | data)
 * Computes the posterior probability that arm a beats arm b using
 * the Beta-Binomial conjugate model. This is the exact Bayesian
 * alternative to frequentist t-tests for A/B comparisons.
 *
 * Key insight: under Beta(alpha_a, beta_a) and Beta(alpha_b, beta_b)
 * priors, P(mu_a > mu_b) = integral_0^1 CDF_Beta(x; alpha_a, beta_a)
 * * PDF_Beta(x; alpha_b, beta_b) dx. We approximate this by Monte Carlo.
 *
 * L4: Bayesian decision theory — minimize expected loss by selecting
 * the arm with highest posterior win probability (Savage 1954).
 */
static double bandit_prob_superiority(double alpha_a, double beta_a,
                                       double alpha_b, double beta_b,
                                       uint64_t *rng) {
    int wins = 0;
    int n_samples = 10000;
    for (int s = 0; s < n_samples; s++) {
        double g1a = bandit_gamma_sample(alpha_a, rng);
        double g2a = bandit_gamma_sample(beta_a, rng);
        double sample_a = g1a / (g1a + g2a + 1e-15);
        double g1b = bandit_gamma_sample(alpha_b, rng);
        double g2b = bandit_gamma_sample(beta_b, rng);
        double sample_b = g1b / (g1b + g2b + 1e-15);
        if (sample_a > sample_b) wins++;
    }
    return (double)wins / (double)n_samples;
}

/*
 * L8: Expected improvement over best arm — used for Bayesian
 * optimization of model selection. Estimates E[max(0, mu_new - mu_best)].
 */
double bandit_expected_improvement(const BanditAllocator *ba, int32_t arm_idx) {
    if (arm_idx < 0 || arm_idx >= ba->num_arms) return 0.0;
    const BanditArm *arm = &ba->arms[arm_idx];
    double best_mean = -INFINITY;
    int best_idx = 0;
    for (int i = 0; i < ba->num_arms; i++) {
        if (i != arm_idx && ba->arms[i].empirical_mean > best_mean) {
            best_mean = ba->arms[i].empirical_mean;
            best_idx = i;
        }
    }
    if (best_mean <= 0.0) return arm->empirical_mean;
    uint64_t rng = (uint64_t)(ba->total_pulls + arm_idx + 1);
    double prob_win = bandit_prob_superiority(
        arm->thompson_alpha, arm->thompson_beta,
        ba->arms[best_idx].thompson_alpha,
        ba->arms[best_idx].thompson_beta, &rng);
    return prob_win * (arm->empirical_mean - best_mean);
}

/* Matrix operations for LinUCB */

/* Cholesky decomposition A = L L^T, solve for LinUCB */
static int bandit_cholesky_solve(int n, double A[], double b[], double x[]) {
    /* In-place Cholesky: A -> L (lower triangular) */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = A[i * BANDIT_MAX_CONTEXT_DIM + j];
            for (int k = 0; k < j; k++)
                sum -= A[i * BANDIT_MAX_CONTEXT_DIM + k] *
                       A[j * BANDIT_MAX_CONTEXT_DIM + k];
            if (i == j) {
                if (sum <= 0.0) return 0;
                A[i * BANDIT_MAX_CONTEXT_DIM + j] = sqrt(sum);
            } else {
                A[j * BANDIT_MAX_CONTEXT_DIM + i] = 0.0;
                A[i * BANDIT_MAX_CONTEXT_DIM + j] =
                    sum / A[j * BANDIT_MAX_CONTEXT_DIM + j];
            }
        }
    }
    /* Forward substitution L y = b */
    double y[BANDIT_MAX_CONTEXT_DIM];
    for (int i = 0; i < n; i++) {
        double sum = b[i];
        for (int j = 0; j < i; j++)
            sum -= A[i * BANDIT_MAX_CONTEXT_DIM + j] * y[j];
        y[i] = sum / A[i * BANDIT_MAX_CONTEXT_DIM + i];
    }
    /* Backward substitution L^T x = y */
    for (int i = n - 1; i >= 0; i--) {
        double sum = y[i];
        for (int j = i + 1; j < n; j++)
            sum -= A[j * BANDIT_MAX_CONTEXT_DIM + i] * x[j];
        x[i] = sum / A[i * BANDIT_MAX_CONTEXT_DIM + i];
    }
    return 1;
}

/* ?? Initialization ?? */

void bandit_init(BanditAllocator *ba) {
    memset(ba, 0, sizeof(BanditAllocator));
    ba->start_time = (int64_t)time(NULL);
    ba->use_ucb = 1;
    ba->ucb_cfg.exploration_coef = 1.414213562;
    ba->ts_cfg.alpha_prior = 1.0;
    ba->ts_cfg.beta_prior = 1.0;
    ba->eps_cfg.epsilon_start = 0.1;
    ba->eps_cfg.epsilon_decay = 0.999;
    ba->eps_cfg.epsilon_min = 0.01;
}

void bandit_add_arm(BanditAllocator *ba, const char *name, int32_t arm_id) {
    if (ba->num_arms >= BANDIT_MAX_ARMS) return;
    BanditArm *arm = &ba->arms[ba->num_arms];
    strncpy(arm->name, name, 63);
    arm->name[63] = '\0';
    arm->arm_id = arm_id;
    arm->pulls = 0;
    arm->sum_rewards = 0.0;
    arm->sum_rewards_sq = 0.0;
    arm->empirical_mean = 0.0;
    arm->empirical_var = 1.0;
    arm->ucb_value = INFINITY;
    arm->thompson_alpha = ba->ts_cfg.alpha_prior;
    arm->thompson_beta = ba->ts_cfg.beta_prior;
    arm->last_reward = 0.0;
    arm->last_pull_time = 0;
    ba->num_arms++;
}

void bandit_configure_ucb(BanditAllocator *ba, double exploration_coef) {
    ba->use_ucb = 1;
    ba->use_thompson = 0;
    ba->use_epsilon = 0;
    ba->ucb_cfg.exploration_coef = exploration_coef;
}

void bandit_configure_thompson(BanditAllocator *ba, double alpha_prior,
                                double beta_prior) {
    ba->use_thompson = 1;
    ba->use_ucb = 0;
    ba->use_epsilon = 0;
    ba->ts_cfg.alpha_prior = alpha_prior;
    ba->ts_cfg.beta_prior = beta_prior;
    for (int i = 0; i < ba->num_arms; i++) {
        ba->arms[i].thompson_alpha = alpha_prior;
        ba->arms[i].thompson_beta = beta_prior;
    }
}

void bandit_configure_epsilon(BanditAllocator *ba, double start, double decay,
                               double min_eps) {
    ba->use_epsilon = 1;
    ba->use_ucb = 0;
    ba->use_thompson = 0;
    ba->eps_cfg.epsilon_start = start;
    ba->eps_cfg.epsilon_decay = decay;
    ba->eps_cfg.epsilon_min = min_eps;
}

void bandit_set_context(BanditAllocator *ba, const double *features, int32_t dim) {
    int32_t d = dim < BANDIT_MAX_CONTEXT_DIM ? dim : BANDIT_MAX_CONTEXT_DIM;
    ba->current_context.dim = d;
    memcpy(ba->current_context.features, features, (size_t)d * sizeof(double));
}

/*
 * L5: UCB1 Selection (Auer et al. 2002)
 *
 * At round t, select arm k maximizing:
 *   UCB_k(t) = mu_hat_k + c * sqrt(2 ln t / N_k)
 *
 * Theoretical guarantee: With probability >= 1 - 1/t, suboptimal arm k
 * is pulled at most O(log t / Delta_k^2) times.
 */
int bandit_select_ucb(BanditAllocator *ba) {
    if (ba->num_arms == 0) return -1;
    int64_t t = ba->total_pulls + 1;
    double log_t = log((double)t);
    /* Pull each arm at least once */
    for (int i = 0; i < ba->num_arms; i++) {
        if (ba->arms[i].pulls == 0) return i;
    }
    int best = 0;
    double best_ucb = -INFINITY;
    double c = ba->ucb_cfg.exploration_coef;
    for (int i = 0; i < ba->num_arms; i++) {
        double bonus = c * sqrt(2.0 * log_t / (double)ba->arms[i].pulls);
        double ucb = ba->arms[i].empirical_mean + bonus;
        ba->arms[i].ucb_value = ucb;
        if (ucb > best_ucb) { best_ucb = ucb; best = i; }
    }
    return best;
}

/*
 * L5: Thompson Sampling (Thompson 1933; Chapelle & Li 2011)
 *
 * For each arm k, sample theta_k ~ Beta(alpha_k, beta_k).
 * Select arm with largest theta_k.
 * Update: alpha_k += r, beta_k += (1-r) for reward r in [0,1].
 */
static double bandit_gamma_sample(double shape, uint64_t *rng) {
    if (shape < 1.0) {
        /* Johnk's generator for shape < 1 */
        double u = bandit_randf(rng);
        double g = pow(u, 1.0 / shape);
        double v = pow(bandit_randf(rng), 1.0 / (1.0 - shape));
        while (g + v > 1.0) {
            u = bandit_randf(rng);
            g = pow(u, 1.0 / shape);
            v = pow(bandit_randf(rng), 1.0 / (1.0 - shape));
        }
        double eta = -log(bandit_randf(rng) + 1e-12);
        return g / (g + v) * eta;
    }
    /* Marsaglia-Tsang for shape >= 1 */
    double d = shape - 1.0 / 3.0;
    double c = 1.0 / sqrt(9.0 * d);
    for (;;) {
        double z = bandit_randn(rng);
        double v = 1.0 + c * z;
        if (v <= 0.0) continue;
        v = v * v * v;
        double u = bandit_randf(rng);
        if (u < 1.0 - 0.0331 * (z * z) * (z * z)) return d * v;
        if (log(u) < 0.5 * z * z + d * (1.0 - v + log(v))) return d * v;
    }
}

int bandit_select_thompson(BanditAllocator *ba) {
    if (ba->num_arms == 0) return -1;
    int best = 0;
    double best_sample = -INFINITY;
    uint64_t rng = (uint64_t)(ba->total_pulls + 1);
    for (int i = 0; i < ba->num_arms; i++) {
        double a = ba->arms[i].thompson_alpha;
        double b_val = ba->arms[i].thompson_beta;
        double g1 = bandit_gamma_sample(a, &rng);
        double g2 = bandit_gamma_sample(b_val, &rng);
        double sample = g1 / (g1 + g2 + 1e-15);
        if (sample > best_sample) { best_sample = sample; best = i; }
    }
    return best;
}

/*
 * L5: Epsilon-Greedy with decay
 * epsilon_t = max(epsilon_min, epsilon_start * decay^t)
 */
int bandit_select_epsilon_greedy(BanditAllocator *ba) {
    if (ba->num_arms == 0) return -1;
    double eps = ba->eps_cfg.epsilon_start *
                 pow(ba->eps_cfg.epsilon_decay, (double)ba->total_pulls);
    if (eps < ba->eps_cfg.epsilon_min) eps = ba->eps_cfg.epsilon_min;
    uint64_t rng = (uint64_t)(ba->total_pulls + 1);
    double r = bandit_randf(&rng);
    if (r < eps) {
        return (int)(bandit_randf(&rng) * (double)ba->num_arms);
    }
    /* Exploit: pick best empirical mean */
    int best = 0;
    double best_mean = -INFINITY;
    for (int i = 0; i < ba->num_arms; i++) {
        double mean = ba->arms[i].pulls > 0 ?
                      ba->arms[i].empirical_mean : INFINITY;
        if (mean > best_mean) { best_mean = mean; best = i; }
    }
    return best;
}

/*
 * L5: LinUCB (Li et al. 2010) ? Contextual Bandit
 *
 * Assumes E[r | x, a] = x^T theta_a (linear payoff).
 * At round t, for each arm a:
 *   theta_hat_a = A_a^{-1} b_a  (ridge regression)
 *   p_a = theta_hat_a^T x + alpha * sqrt(x^T A_a^{-1} x)
 */
int bandit_select_linucb(BanditAllocator *ba) {
    if (ba->num_arms == 0) return -1;
    int d = ba->current_context.dim;
    if (d == 0) return 0;
    const double *x = ba->current_context.features;
    int best = 0;
    double best_p = -INFINITY;
    for (int i = 0; i < ba->num_arms; i++) {
        LinUCBArm *la = &ba->linucb_arms[i];
        if (!la->initialized) {
            la->alpha = 1.0;
            for (int r = 0; r < d; r++) {
                for (int c = 0; c < d; c++)
                    la->A[r][c] = (r == c) ? la->alpha : 0.0;
                la->b[r] = 0.0;
                la->theta[r] = 0.0;
            }
            la->initialized = 1;
        }
        /* Copy A to work buffer for Cholesky */
        double A_work[BANDIT_MAX_CONTEXT_DIM * BANDIT_MAX_CONTEXT_DIM];
        for (int r = 0; r < d; r++)
            for (int c = 0; c < d; c++)
                A_work[r * BANDIT_MAX_CONTEXT_DIM + c] = la->A[r][c];
        int ok = bandit_cholesky_solve(d, A_work, la->b, la->theta);
        if (!ok) continue;
        /* Compute x^T A^{-1} x */
        double A_copy[BANDIT_MAX_CONTEXT_DIM * BANDIT_MAX_CONTEXT_DIM];
        for (int r = 0; r < d; r++)
            for (int c = 0; c < d; c++)
                A_copy[r * BANDIT_MAX_CONTEXT_DIM + c] = la->A[r][c];
        double x_copy[BANDIT_MAX_CONTEXT_DIM];
        for (int r = 0; r < d; r++) x_copy[r] = x[r];
        double A_inv_x[BANDIT_MAX_CONTEXT_DIM] = {0};
        int ok2 = bandit_cholesky_solve(d, A_copy, x_copy, A_inv_x);
        if (!ok2) continue;
        double xt_Ainv_x = 0.0;
        for (int r = 0; r < d; r++) xt_Ainv_x += x[r] * A_inv_x[r];
        double theta_dot_x = 0.0;
        for (int r = 0; r < d; r++) theta_dot_x += la->theta[r] * x[r];
        double p = theta_dot_x + la->alpha * sqrt(xt_Ainv_x + 1e-10);
        if (p > best_p) { best_p = p; best = i; }
    }
    return best;
}

/* ?? L5: Reward Update ?? */

void bandit_update(BanditAllocator *ba, int32_t arm_idx, double reward) {
    if (arm_idx < 0 || arm_idx >= ba->num_arms) return;
    BanditArm *arm = &ba->arms[arm_idx];
    arm->pulls++;
    arm->sum_rewards += reward;
    arm->sum_rewards_sq += reward * reward;
    arm->last_reward = reward;
    arm->last_pull_time = (int64_t)time(NULL);
    double n = (double)arm->pulls;
    arm->empirical_mean = arm->sum_rewards / n;
    double mean_sq = arm->sum_rewards_sq / n;
    arm->empirical_var = mean_sq - arm->empirical_mean * arm->empirical_mean;
    if (arm->empirical_var < 1e-10) arm->empirical_var = 1e-10;
    arm->thompson_alpha += reward;
    arm->thompson_beta += (1.0 - reward);
    if (ba->use_linucb && ba->current_context.dim > 0) {
        LinUCBArm *la = &ba->linucb_arms[arm_idx];
        if (la->initialized) {
            const double *x = ba->current_context.features;
            int d = ba->current_context.dim;
            for (int r = 0; r < d; r++) {
                for (int c = 0; c < d; c++)
                    la->A[r][c] += x[r] * x[c];
                la->b[r] += reward * x[r];
            }
        }
    }
    ba->total_pulls++;
    ba->cumulative_reward += reward;
    double best_mean = -INFINITY;
    for (int i = 0; i < ba->num_arms; i++) {
        if (ba->arms[i].empirical_mean > best_mean)
            best_mean = ba->arms[i].empirical_mean;
    }
    double instant_regret = best_mean - reward;
    ba->cumulative_regret += instant_regret;
    if (ba->regret_len < BANDIT_HISTORY_MAX) {
        ba->regret_history[ba->regret_len++] = instant_regret;
    }
}

/* ?? L4: Regret & Theoretical Bounds ?? */

double bandit_cumulative_regret(const BanditAllocator *ba) {
    return ba->cumulative_regret;
}

double bandit_ucb_regret_bound(const BanditAllocator *ba) {
    if (ba->num_arms == 0 || ba->total_pulls == 0) return 0.0;
    int best = bandit_best_arm(ba);
    double best_mean = ba->arms[best].empirical_mean;
    double T = (double)ba->total_pulls;
    double bound = 0.0;
    double pi_sq_over_3 = 3.289868133696453;
    for (int i = 0; i < ba->num_arms; i++) {
        if (i == best) continue;
        double delta = best_mean - ba->arms[i].empirical_mean;
        if (delta <= 0.0) continue;
        bound += 8.0 * log(T) / delta + (1.0 + pi_sq_over_3) * delta;
    }
    return bound;
}

double bandit_lai_robbins_lower_bound(const BanditAllocator *ba) {
    if (ba->num_arms == 0 || ba->total_pulls == 0) return 0.0;
    int best = bandit_best_arm(ba);
    double p_star = ba->arms[best].empirical_mean;
    if (p_star < 0.001) p_star = 0.001;
    if (p_star > 0.999) p_star = 0.999;
    double T = (double)ba->total_pulls;
    double lb = 0.0;
    for (int i = 0; i < ba->num_arms; i++) {
        if (i == best) continue;
        double p_k = ba->arms[i].empirical_mean;
        if (p_k < 0.001) p_k = 0.001;
        if (p_k > 0.999) p_k = 0.999;
        if (p_star <= p_k) continue;
        double kl = p_k * log(p_k / p_star) +
                    (1.0 - p_k) * log((1.0 - p_k) / (1.0 - p_star));
        if (kl > 1e-10) {
            double n_k_lower = log(T) / kl;
            double delta = p_star - p_k;
            lb += n_k_lower * delta;
        }
    }
    return lb;
}

/* ?? Statistics ?? */

void bandit_get_arm_stats(const BanditAllocator *ba, int32_t arm_idx,
                          double *mean, double *std_err, int32_t *pulls) {
    if (arm_idx < 0 || arm_idx >= ba->num_arms) {
        *mean = 0.0; *std_err = 0.0; *pulls = 0;
        return;
    }
    const BanditArm *arm = &ba->arms[arm_idx];
    *mean = arm->empirical_mean;
    *pulls = arm->pulls;
    *std_err = arm->pulls > 1 ?
               sqrt(arm->empirical_var / (double)arm->pulls) : 1.0;
}

int bandit_best_arm(const BanditAllocator *ba) {
    int best = 0;
    double best_mean = -INFINITY;
    for (int i = 0; i < ba->num_arms; i++) {
        if (ba->arms[i].pulls > 0 && ba->arms[i].empirical_mean > best_mean) {
            best_mean = ba->arms[i].empirical_mean;
            best = i;
        }
    }
    return best;
}

double bandit_empirical_gap(const BanditAllocator *ba, int32_t arm_i,
                             int32_t arm_j) {
    if (arm_i < 0 || arm_i >= ba->num_arms) return 0.0;
    if (arm_j < 0 || arm_j >= ba->num_arms) return 0.0;
    return ba->arms[arm_i].empirical_mean - ba->arms[arm_j].empirical_mean;
}

void bandit_allocation_summary(const BanditAllocator *ba, double *allocations) {
    double total = (double)ba->total_pulls;
    if (total == 0.0) total = 1.0;
    for (int i = 0; i < ba->num_arms; i++)
        allocations[i] = (double)ba->arms[i].pulls / total;
}

/* ?? L7 Application: Dynamic Traffic Routing ?? */

int bandit_route_request(BanditAllocator *ba, int32_t user_id, uint64_t seed) {
    (void)user_id;
    (void)seed;
    if (ba->use_ucb)       return bandit_select_ucb(ba);
    if (ba->use_thompson)  return bandit_select_thompson(ba);
    if (ba->use_epsilon)   return bandit_select_epsilon_greedy(ba);
    if (ba->use_linucb)    return bandit_select_linucb(ba);
    return ba->total_pulls % ba->num_arms;
}

void bandit_route_batch(BanditAllocator *ba, int32_t *user_ids, int32_t n,
                         uint64_t seed, int32_t *assignments) {
    for (int32_t i = 0; i < n; i++) {
        assignments[i] = bandit_route_request(ba, user_ids[i],
                                               seed + (uint64_t)i);
    }
}

/* ?? Reporting ?? */

void bandit_export_csv(const BanditAllocator *ba, char *buf, int32_t max_len) {
    int off = 0;
    off += snprintf(buf + off, (size_t)(max_len - off),
                    "arm_name,pulls,mean,ucb,alpha,beta\n");
    for (int i = 0; i < ba->num_arms; i++) {
        const BanditArm *a = &ba->arms[i];
        off += snprintf(buf + off, (size_t)(max_len - off),
                        "%s,%d,%.6f,%.6f,%.3f,%.3f\n",
                        a->name, a->pulls, a->empirical_mean,
                        a->ucb_value, a->thompson_alpha, a->thompson_beta);
    }
    off += snprintf(buf + off, (size_t)(max_len - off),
                    "total_pulls,%d,cum_reward,%.4f,cum_regret,%.4f\n",
                    ba->total_pulls, ba->cumulative_reward,
                    ba->cumulative_regret);
}

void bandit_reset_arms(BanditAllocator *ba) {
    for (int i = 0; i < ba->num_arms; i++) {
        BanditArm *arm = &ba->arms[i];
        arm->pulls = 0;
        arm->sum_rewards = 0.0;
        arm->sum_rewards_sq = 0.0;
        arm->empirical_mean = 0.0;
        arm->empirical_var = 1.0;
        arm->ucb_value = INFINITY;
        arm->thompson_alpha = ba->ts_cfg.alpha_prior;
        arm->thompson_beta = ba->ts_cfg.beta_prior;
    }
    ba->total_pulls = 0;
    ba->cumulative_reward = 0.0;
    ba->cumulative_regret = 0.0;
    ba->regret_len = 0;
}

/*
 * L8: Public wrapper for probability of superiority computation.
 * For arm_a vs arm_b, computes P(mu_a > mu_b | data) using Monte Carlo
 * over the Beta posteriors. This is the Bayesian answer to "is variant
 * A better than variant B?"
 *
 * Refs: Bayesian A/B Testing (Vwo, 2015); Stucchio (2015)
 */
double bandit_prob_superiority_wrapper(const BanditAllocator *ba,
                                        int32_t arm_a, int32_t arm_b,
                                        uint64_t seed) {
    if (arm_a < 0 || arm_a >= ba->num_arms) return -1.0;
    if (arm_b < 0 || arm_b >= ba->num_arms) return -1.0;
    uint64_t rng = seed;
    return bandit_prob_superiority(
        ba->arms[arm_a].thompson_alpha, ba->arms[arm_a].thompson_beta,
        ba->arms[arm_b].thompson_alpha, ba->arms[arm_b].thompson_beta,
        &rng);
}
