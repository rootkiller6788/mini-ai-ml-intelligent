#include "preprocessing.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

/* ═══════════════════════════════════════════════════════════════════════
   Standard Scaler (Z-score Normalisation)
   ───────────────────────────────────────────────────────────────────────
   Transforms features to zero mean and unit variance.
   z = (x − μ) / σ
   Often required for SVM, logistic regression, K-means (distance-based
   methods). μ̂ and σ̂ are computed from training data and reused for
   test data to prevent data leakage.
   ═══════════════════════════════════════════════════════════════════════ */

StandardScaler scaler_std_create(size_t n_features) {
    StandardScaler s;
    s.n_features = n_features;
    s.mean  = (double *)calloc(n_features, sizeof(double));
    s.std   = (double *)calloc(n_features, sizeof(double));
    s.fitted = false;
    return s;
}

void scaler_std_destroy(StandardScaler *s) {
    free(s->mean);
    free(s->std);
    s->mean = NULL;
    s->std  = NULL;
}

void scaler_std_fit(StandardScaler *s, const double *X, size_t n) {
    if (n == 0) return;
    size_t d = s->n_features;

    /* mean */
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j)
            s->mean[j] += X[i * d + j];
    for (size_t j = 0; j < d; ++j)
        s->mean[j] /= (double)n;

    /* std (population formula: divide by n) */
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            double diff = X[i * d + j] - s->mean[j];
            s->std[j] += diff * diff;
        }
    }
    for (size_t j = 0; j < d; ++j) {
        s->std[j] = sqrt(s->std[j] / (double)n);
        if (s->std[j] < 1e-12) s->std[j] = 1.0;  /* constant feature */
    }
    s->fitted = true;
}

void scaler_std_transform(const StandardScaler *s,
                           const double *X, double *out, size_t n) {
    size_t d = s->n_features;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j)
            out[i * d + j] = (X[i * d + j] - s->mean[j]) / s->std[j];
}

void scaler_std_inverse_transform(const StandardScaler *s,
                                    const double *X, double *out, size_t n) {
    size_t d = s->n_features;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j)
            out[i * d + j] = X[i * d + j] * s->std[j] + s->mean[j];
}

void scaler_std_fit_transform(StandardScaler *s,
                               const double *X, double *out, size_t n) {
    scaler_std_fit(s, X, n);
    scaler_std_transform(s, X, out, n);
}

/* ═══════════════════════════════════════════════════════════════════════
   Min-Max Scaler
   ───────────────────────────────────────────────────────────────────────
   Maps features to [feature_min, feature_max].
   x' = (x − min) / (max − min) × (feature_max − feature_min) + feature_min
   Preserves zero entries for sparse data; bounds all features to same
   range, which helps gradient-based optimisers.
   ═══════════════════════════════════════════════════════════════════════ */

MinMaxScaler scaler_minmax_create(size_t n_features,
                                    double feature_min, double feature_max) {
    MinMaxScaler s;
    s.n_features  = n_features;
    s.feature_min = feature_min;
    s.feature_max = feature_max;
    s.min_val = (double *)malloc(n_features * sizeof(double));
    s.max_val = (double *)malloc(n_features * sizeof(double));
    s.fitted  = false;
    return s;
}

void scaler_minmax_destroy(MinMaxScaler *s) {
    free(s->min_val);
    free(s->max_val);
    s->min_val = NULL;
    s->max_val = NULL;
}

void scaler_minmax_fit(MinMaxScaler *s, const double *X, size_t n) {
    if (n == 0) return;
    size_t d = s->n_features;
    for (size_t j = 0; j < d; ++j) {
        s->min_val[j] =  DBL_MAX;
        s->max_val[j] = -DBL_MAX;
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            double v = X[i * d + j];
            if (v < s->min_val[j]) s->min_val[j] = v;
            if (v > s->max_val[j]) s->max_val[j] = v;
        }
    }
    s->fitted = true;
}

void scaler_minmax_transform(const MinMaxScaler *s,
                               const double *X, double *out, size_t n) {
    size_t d = s->n_features;
    double range_in, scale;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            range_in = s->max_val[j] - s->min_val[j];
            if (range_in < 1e-12) {
                out[i * d + j] = 0.0;
                continue;
            }
            scale = (s->feature_max - s->feature_min) / range_in;
            out[i * d + j] = (X[i * d + j] - s->min_val[j]) * scale
                             + s->feature_min;
        }
    }
}

void scaler_minmax_inverse_transform(const MinMaxScaler *s,
                                       const double *X, double *out, size_t n) {
    size_t d = s->n_features;
    double range_out;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            range_out = s->max_val[j] - s->min_val[j];
            if (range_out < 1e-12) {
                out[i * d + j] = s->min_val[j];
                continue;
            }
            out[i * d + j] = s->min_val[j]
                + (X[i * d + j] - s->feature_min) / (s->feature_max - s->feature_min)
                  * range_out;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
   Label Encoder
   ───────────────────────────────────────────────────────────────────────
   Maps arbitrary categorical labels to [0, n_classes-1].
   Commonly used before OneHotEncoder or for models that require numeric
   targets starting from 0 (e.g., softmax).
   ═══════════════════════════════════════════════════════════════════════ */

LabelEncoder labelenc_create(void) {
    LabelEncoder le;
    le.classes_   = NULL;
    le.n_classes  = 0;
    return le;
}

void labelenc_destroy(LabelEncoder *le) {
    free(le->classes_);
    le->classes_ = NULL;
}

/* Simple insertion-sort for uniqueness */
static int cmp_int(const void *a, const void *b) {
    return (*(const int *)a - *(const int *)b);
}

void labelenc_fit(LabelEncoder *le, const int *y, size_t n) {
    if (n == 0) return;
    /* find unique labels */
    int *unique = (int *)malloc(n * sizeof(int));
    size_t nu = 0;
    for (size_t i = 0; i < n; ++i) {
        bool found = false;
        for (size_t k = 0; k < nu; ++k)
            if (unique[k] == y[i]) { found = true; break; }
        if (!found) unique[nu++] = y[i];
    }
    qsort(unique, nu, sizeof(int), cmp_int);
    le->classes_  = unique;
    le->n_classes = nu;
}

void labelenc_transform(const LabelEncoder *le, const int *y, int *out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        int encoded = 0;
        for (size_t k = 0; k < le->n_classes; ++k)
            if (le->classes_[k] == y[i]) { encoded = (int)k; break; }
        out[i] = encoded;
    }
}

int labelenc_inverse_transform(const LabelEncoder *le, int encoded) {
    if (encoded < 0 || (size_t)encoded >= le->n_classes) return -1;
    return le->classes_[encoded];
}

/* ═══════════════════════════════════════════════════════════════════════
   One-Hot Encoder
   ───────────────────────────────────────────────────────────────────────
   Converts each categorical feature with c categories into c binary
   columns (exactly one is 1, others 0). Increases dimensionality but
   removes ordinal assumptions from categorical data.
   ═══════════════════════════════════════════════════════════════════════ */

OneHotEncoder ohe_create(void) {
    OneHotEncoder ohe;
    ohe.n_categories      = 0;
    ohe.categories        = NULL;
    ohe.n_features        = 0;
    ohe.feature_categories = NULL;
    ohe.feature_offset    = NULL;
    ohe.total_columns     = 0;
    ohe.fitted            = false;
    return ohe;
}

void ohe_destroy(OneHotEncoder *ohe) {
    free(ohe->categories);
    free(ohe->feature_categories);
    free(ohe->feature_offset);
    ohe->categories         = NULL;
    ohe->feature_categories = NULL;
    ohe->feature_offset     = NULL;
}

void ohe_fit(OneHotEncoder *ohe, const int *X,
             size_t n_samples, size_t n_features) {
    ohe->n_features = n_features;
    ohe->feature_categories = (int *)malloc(n_features * sizeof(int));
    ohe->feature_offset     = (size_t *)malloc((n_features + 1) * sizeof(size_t));

    /* Count categories per feature */
    ohe->total_columns = 0;
    ohe->feature_offset[0] = 0;
    for (size_t f = 0; f < n_features; ++f) {
        int cats[256] = {0};
        int n_cats = 0;
        for (size_t i = 0; i < n_samples; ++i) {
            int val = X[i * n_features + f];
            bool found = false;
            for (int c = 0; c < n_cats; ++c)
                if (cats[c] == val) { found = true; break; }
            if (!found && n_cats < 256) cats[n_cats++] = val;
        }
        /* sort categories for deterministic encoding */
        for (int a = 0; a < n_cats; ++a)
            for (int b = a + 1; b < n_cats; ++b)
                if (cats[a] > cats[b]) { int t = cats[a]; cats[a] = cats[b]; cats[b] = t; }
        ohe->feature_categories[f] = n_cats;
        ohe->total_columns += (size_t)n_cats;
        ohe->feature_offset[f + 1] = ohe->total_columns;
    }
    ohe->fitted = true;
}

void ohe_transform(const OneHotEncoder *ohe,
                   const int *X, double *out, size_t n_samples) {
    size_t nf = ohe->n_features;
    for (size_t i = 0; i < n_samples; ++i) {
        /* zero-fill the row */
        for (size_t c = 0; c < ohe->total_columns; ++c)
            out[i * ohe->total_columns + c] = 0.0;
        for (size_t f = 0; f < nf; ++f) {
            int val = X[i * nf + f];
            int n_cats = ohe->feature_categories[f];
            size_t offset = ohe->feature_offset[f];
            /* linear search category index */
            int cat_idx = 0;
            /* categories aren't stored globally; we encode zero-based */
            if (val >= 0 && val < n_cats)
                cat_idx = val;
            out[i * ohe->total_columns + offset + (size_t)cat_idx] = 1.0;
        }
    }
}

size_t ohe_n_features_out(const OneHotEncoder *ohe) {
    return ohe->total_columns;
}

/* ═══════════════════════════════════════════════════════════════════════
   Simple Imputer — Missing Value Handling
   ───────────────────────────────────────────────────────────────────────
   Replaces missing values (NaN, flagged by isnan) with strategies:
   mean, median, mode, or constant fill. Common in real-world datasets
   where sensor failures or survey non-response create gaps.
   ═══════════════════════════════════════════════════════════════════════ */

SimpleImputer imputer_create(ImputeStrategy strategy,
                               double constant_value, size_t n_features) {
    SimpleImputer imp;
    imp.strategy       = strategy;
    imp.constant_value = constant_value;
    imp.fill_values    = (double *)calloc(n_features, sizeof(double));
    imp.n_features     = n_features;
    imp.fitted         = false;
    return imp;
}

void imputer_destroy(SimpleImputer *imp) {
    free(imp->fill_values);
    imp->fill_values = NULL;
}

static int cmp_double(const void *a, const void *b) {
    double diff = *(const double *)a - *(const double *)b;
    return (diff > 0) - (diff < 0);
}

void imputer_fit(SimpleImputer *imp, const double *X, size_t n) {
    size_t d = imp->n_features;
    if (n == 0) return;

    for (size_t j = 0; j < d; ++j) {
        double *col = (double *)malloc(n * sizeof(double));
        size_t valid_count = 0;
        for (size_t i = 0; i < n; ++i) {
            if (!isnan(X[i * d + j]))
                col[valid_count++] = X[i * d + j];
        }
        if (valid_count == 0) {
            imp->fill_values[j] = imp->constant_value;
        } else {
            switch (imp->strategy) {
            case IMPUTE_MEAN: {
                double sum = 0.0;
                for (size_t k = 0; k < valid_count; ++k) sum += col[k];
                imp->fill_values[j] = sum / (double)valid_count;
                break;
            }
            case IMPUTE_MEDIAN: {
                qsort(col, valid_count, sizeof(double), cmp_double);
                if (valid_count % 2 == 0)
                    imp->fill_values[j] = (col[valid_count/2 - 1] + col[valid_count/2]) / 2.0;
                else
                    imp->fill_values[j] = col[valid_count / 2];
                break;
            }
            case IMPUTE_MODE: {
                /* find most frequent value (simple binning for doubles) */
                qsort(col, valid_count, sizeof(double), cmp_double);
                double mode_val = col[0];
                size_t mode_cnt = 1, cur_cnt = 1;
                for (size_t k = 1; k < valid_count; ++k) {
                    if (fabs(col[k] - col[k-1]) < 1e-9)
                        cur_cnt++;
                    else {
                        if (cur_cnt > mode_cnt) {
                            mode_cnt = cur_cnt;
                            mode_val = col[k-1];
                        }
                        cur_cnt = 1;
                    }
                }
                if (cur_cnt > mode_cnt) mode_val = col[valid_count-1];
                imp->fill_values[j] = mode_val;
                break;
            }
            case IMPUTE_CONSTANT:
            default:
                imp->fill_values[j] = imp->constant_value;
                break;
            }
        }
        free(col);
    }
    imp->fitted = true;
}

void imputer_transform(const SimpleImputer *imp,
                        const double *X, double *out, size_t n) {
    size_t d = imp->n_features;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j)
            out[i * d + j] = isnan(X[i * d + j])
                             ? imp->fill_values[j] : X[i * d + j];
}

/* ═══════════════════════════════════════════════════════════════════════
   Polynomial Feature Expansion
   ───────────────────────────────────────────────────────────────────────
   Generates polynomial and interaction features up to degree d.
   For n features and degree d, output size = C(n+d, d) combinations.
   Allows linear models to learn non-linear decision boundaries.
   Example: [x1, x2]^2 → [1, x1, x2, x1², x1·x2, x2²]
   ═══════════════════════════════════════════════════════════════════════ */

/* Generate all combinations of n_features variables with total power ≤ degree */
/* combinatorial indexing using multi-index enumeration */
static size_t count_combinations(size_t n, size_t d) {
    /* C(n+d, d) = (n+d)! / (n! * d!) */
    size_t result = 1;
    for (size_t i = 1; i <= d; ++i) {
        result = result * (n + i) / i;
    }
    return result;
}

static void generate_combinations(size_t n, size_t d, size_t *combos) {
    /* multi-index approach: iterate over all (α₁,...,αₙ) with Σαᵢ ≤ d */
    size_t idx = 0;
    size_t *alpha = (size_t *)calloc(n, sizeof(size_t));
    /* simple recursive enumeration */
    /* worklist-based generation */

    /* Allocate temporary stack */
    size_t stack_cap = count_combinations(n, d) * n;
    size_t *stack = (size_t *)malloc(stack_cap * sizeof(size_t));
    size_t stack_top = 0;

    /* seed: zero vector */
    for (size_t j = 0; j < n; ++j) stack[stack_top++] = 0;
    size_t combos_written = 0;

    /* Iterative DFS over multi-indices */
    size_t pos = 0;
    size_t *cur = (size_t *)calloc(n, sizeof(size_t));

    /* simple enumeration by generating all possible exponent vectors */
    for (size_t e0 = 0; e0 <= d; ++e0) {
        if (n == 1) {
            if (e0 <= d) {
                combos[combos_written * (d + 1) + 0] = e0;  /* for n=1 simple case */
                combos_written++;
                continue;
            }
        }
    }
    /* For general n, use nested loops up to the max degree        */
    /* Simplified approach: generate all tuples via combinatorial */
    if (n == 1) {
        for (size_t p = 0; p <= d; ++p) {
            combos[idx++] = p;
        }
    } else if (n == 2) {
        for (size_t p1 = 0; p1 <= d; ++p1) {
            for (size_t p2 = 0; p2 <= d; ++p2) {
                if (p1 + p2 <= d) {
                    combos[idx * 2]     = p1;
                    combos[idx * 2 + 1] = p2;
                    idx++;
                }
            }
        }
    } else {
        /* General case: recursive enumeration via nested loops approach */
        /* Use a simple combinatorial number system traversal */
        size_t *powers = (size_t *)calloc(n, sizeof(size_t));
        while (1) {
            size_t sum = 0;
            for (size_t j = 0; j < n; ++j) sum += powers[j];
            if (sum <= d) {
                for (size_t j = 0; j < n; ++j)
                    combos[idx * n + j] = powers[j];
                idx++;
            }
            /* increment */
            size_t carry_pos = 0;
            powers[carry_pos]++;
            while (carry_pos < n && powers[carry_pos] > d) {
                powers[carry_pos] = 0;
                carry_pos++;
                if (carry_pos < n) powers[carry_pos]++;
            }
            if (carry_pos >= n) break;
        }
        free(powers);
    }

    free(cur); free(stack); free(alpha);
    (void)stack_top;
}

PolynomialFeatures polyfeat_create(size_t degree, size_t n_features,
                                    bool include_bias) {
    PolynomialFeatures pf;
    pf.degree        = degree;
    pf.n_features_in = n_features;
    pf.include_bias  = include_bias;

    /* count total combinations including bias, then adjust output count */
    size_t n_combos_total = count_combinations(n_features, degree);
    size_t n_combos_out   = include_bias ? n_combos_total : n_combos_total - 1;

    pf.n_features_out = n_combos_out;
    /* allocate for total combos (generator writes all combos including bias) */
    pf.combinations   = (size_t *)calloc(n_combos_total * n_features, sizeof(size_t));
    generate_combinations(n_features, degree, pf.combinations);
    return pf;
}

void polyfeat_destroy(PolynomialFeatures *pf) {
    free(pf->combinations);
    pf->combinations = NULL;
}

size_t polyfeat_n_output(const PolynomialFeatures *pf) {
    return pf->n_features_out;
}

void polyfeat_transform(const PolynomialFeatures *pf,
                         const double *X, double *out,
                         size_t n_samples) {
    size_t n_in  = pf->n_features_in;
    size_t n_out = pf->n_features_out;

    for (size_t i = 0; i < n_samples; ++i) {
        size_t out_idx = 0;
        /* Start from 1 (skip bias) if include_bias is false */
        size_t start_c = pf->include_bias ? 0 : 1;
        size_t n_combos_total = pf->include_bias ? n_out : n_out + 1;
        for (size_t c = start_c; c < n_combos_total; ++c) {
            double prod = 1.0;
            size_t *powers = pf->combinations + c * n_in;
            for (size_t j = 0; j < n_in; ++j) {
                if (powers[j] > 0) {
                    double xj = X[i * n_in + j];
                    for (size_t p = 0; p < powers[j]; ++p)
                        prod *= xj;
                }
            }
            out[i * n_out + out_idx] = prod;
            out_idx++;
        }
    }
}