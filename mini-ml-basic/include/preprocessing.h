#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <stdbool.h>
#include <stddef.h>

/*
 * preprocessing.h ? Feature Engineering & Data Transformation
 *
 * L1: StandardScaler, MinMaxScaler, LabelEncoder structs
 * L2: Feature engineering ? transforming raw data to model-ready features
 * L3: Pipeline transformation chain (fit ? transform ? inverse_transform)
 * L4: Z-score formula: z = (x ? ?) / ?; MinMax: x' = (x ? min) / (max ? min)
 * L5: Standard scaling, MinMax scaling, One-hot encoding, Label encoding
 */

/* ??????????????????????????????????????????????
   Standard Scaler (Z-score Normalisation)
   x'? = (x? ? ??) / ??
   ?????????????????????????????????????????????? */
typedef struct {
    double *mean;                /* (n_features,)                     */
    double *std;                 /* (n_features,)                     */
    size_t  n_features;
    bool    fitted;
} StandardScaler;

StandardScaler  scaler_std_create(size_t n_features);
void            scaler_std_destroy(StandardScaler *s);
void            scaler_std_fit(StandardScaler *s, const double *X, size_t n);
void            scaler_std_transform(const StandardScaler *s,
                                      const double *X, double *out, size_t n);
void            scaler_std_inverse_transform(const StandardScaler *s,
                                              const double *X, double *out, size_t n);
void            scaler_std_fit_transform(StandardScaler *s,
                                         const double *X, double *out, size_t n);

/* ??????????????????????????????????????????????
   Min-Max Scaler
   x'? = (x? ? min?) / (max? ? min?)
   Maps features to range [feature_min, feature_max]
   ?????????????????????????????????????????????? */
typedef struct {
    double *min_val;             /* (n_features,)                     */
    double *max_val;             /* (n_features,)                     */
    double  feature_min;         /* target range lower bound          */
    double  feature_max;         /* target range upper bound          */
    size_t  n_features;
    bool    fitted;
} MinMaxScaler;

MinMaxScaler  scaler_minmax_create(size_t n_features,
                                    double feature_min, double feature_max);
void          scaler_minmax_destroy(MinMaxScaler *s);
void          scaler_minmax_fit(MinMaxScaler *s, const double *X, size_t n);
void          scaler_minmax_transform(const MinMaxScaler *s,
                                       const double *X, double *out, size_t n);
void          scaler_minmax_inverse_transform(const MinMaxScaler *s,
                                               const double *X, double *out, size_t n);

/* ??????????????????????????????????????????????
   Label Encoder ? maps categorical strings/local labels ? 0..C-1
   Simplified: maps int labels to contiguous range
   ?????????????????????????????????????????????? */
typedef struct {
    int    *classes_;            /* unique sorted class values       */
    size_t  n_classes;
} LabelEncoder;

LabelEncoder  labelenc_create(void);
void          labelenc_destroy(LabelEncoder *le);
void          labelenc_fit(LabelEncoder *le, const int *y, size_t n);
void          labelenc_transform(const LabelEncoder *le,
                                  const int *y, int *out, size_t n);
int           labelenc_inverse_transform(const LabelEncoder *le, int encoded);

/* ??????????????????????????????????????????????
   One-Hot Encoder
   For feature with c possible values, creates c binary columns
   ?????????????????????????????????????????????? */
typedef struct {
    int     n_categories;        /* unique values in this feature   */
    int    *categories;          /* sorted unique value list        */
    size_t  n_features;          /* original feature count          */
    int    *feature_categories;  /* per-feature category count      */
    /* accumulated offsets for multi-feature encoding */
    size_t *feature_offset;
    size_t  total_columns;       /* ? categories                    */
    bool    fitted;
} OneHotEncoder;

OneHotEncoder  ohe_create(void);
void           ohe_destroy(OneHotEncoder *ohe);
void           ohe_fit(OneHotEncoder *ohe, const int *X,
                      size_t n_samples, size_t n_features);
void           ohe_transform(const OneHotEncoder *ohe,
                             const int *X, double *out,
                             size_t n_samples);
size_t         ohe_n_features_out(const OneHotEncoder *ohe);

/* ??????????????????????????????????????????????
   Missing Value Imputation
   ?????????????????????????????????????????????? */
typedef enum {
    IMPUTE_MEAN = 0,
    IMPUTE_MEDIAN,
    IMPUTE_MODE,
    IMPUTE_CONSTANT
} ImputeStrategy;

typedef struct {
    ImputeStrategy strategy;
    double         constant_value;
    double        *fill_values;    /* (n_features,) computed values */
    size_t         n_features;
    bool           fitted;
} SimpleImputer;

SimpleImputer  imputer_create(ImputeStrategy strategy,
                                double constant_value, size_t n_features);
void           imputer_destroy(SimpleImputer *imp);
void           imputer_fit(SimpleImputer *imp, const double *X, size_t n);
void           imputer_transform(const SimpleImputer *imp,
                                  const double *X, double *out, size_t n);

/* ??????????????????????????????????????????????
   Polynomial Feature Expansion
   Generates interaction terms and powers up to degree d
   ?????????????????????????????????????????????? */
typedef struct {
    size_t  degree;
    size_t  n_features_in;
    size_t  n_features_out;      /* C(n+d, d) combinations           */
    size_t *combinations;        /* (n_out, degree) term indices     */
    bool    include_bias;
} PolynomialFeatures;

PolynomialFeatures  polyfeat_create(size_t degree, size_t n_features,
                                     bool include_bias);
void                polyfeat_destroy(PolynomialFeatures *pf);
size_t              polyfeat_n_output(const PolynomialFeatures *pf);
void                polyfeat_transform(const PolynomialFeatures *pf,
                                        const double *X, double *out,
                                        size_t n_samples);

#endif /* PREPROCESSING_H */
