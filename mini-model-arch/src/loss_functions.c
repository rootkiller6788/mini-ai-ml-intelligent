#include "loss_functions.h"
#include <stdio.h>

/*
 * L2: MSE Loss = (1/n) * sum((pred_i - target_i)^2)
 * L4: Maximum Likelihood Estimation: minimizing MSE = maximizing likelihood
 *     for Gaussian error with fixed variance.
 */
float mse_loss(const float *pred, const float *target, int n) {
    float loss = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = pred[i] - target[i];
        loss += diff * diff;
    }
    return loss / (float)n;
}

float mse_loss_grad(const float *pred, const float *target, float *grad, int n) {
    float loss = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = pred[i] - target[i];
        loss += diff * diff;
        grad[i] = 2.0f * diff / (float)n;
    }
    return loss / (float)n;
}

/*
 * L4: Binary Cross-Entropy (BCE)
 * L(y, p) = -[y * log(p) + (1-y) * log(1-p)]
 * Derived from Bernoulli MLE.
 * Uses clip(p, 1e-7, 1-1e-7) for numerical stability.
 */
float bce_loss(const float *pred, const float *target, int n) {
    float loss = 0.0f, eps = 1e-7f;
    for (int i = 0; i < n; i++) {
        float p = pred[i];
        if (p < eps) p = eps;
        if (p > 1.0f - eps) p = 1.0f - eps;
        loss -= target[i] * logf(p) + (1.0f - target[i]) * logf(1.0f - p);
    }
    return loss / (float)n;
}

float bce_loss_grad(const float *pred, const float *target, float *grad, int n) {
    float loss = 0.0f, eps = 1e-7f;
    for (int i = 0; i < n; i++) {
        float p = pred[i];
        if (p < eps) p = eps;
        if (p > 1.0f - eps) p = 1.0f - eps;
        loss -= target[i] * logf(p) + (1.0f - target[i]) * logf(1.0f - p);
        grad[i] = -(target[i] / p - (1.0f - target[i]) / (1.0f - p)) / (float)n;
    }
    return loss / (float)n;
}

/*
 * L4: Categorical Cross-Entropy with softmax
 * L = -(1/N) * sum_n log(softmax(logits)[target])
 * Gradient: dL/d(logits_i) = softmax(logits_i) - 1[i==target]
 */
static void softmax_stable_inline(float *x, int n) {
    float maxv = x[0];
    for (int i = 1; i < n; i++) if (x[i] > maxv) maxv = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - maxv);
        sum += x[i];
    }
    if (sum < 1e-8f) sum = 1e-8f;
    for (int i = 0; i < n; i++) x[i] /= sum;
}

float cross_entropy_loss(const float *logits, const int *targets, int batch, int classes) {
    float loss = 0.0f;
    for (int b = 0; b < batch; b++) {
        float *probs = (float *)malloc(classes * sizeof(float));
        memcpy(probs, logits + b * classes, classes * sizeof(float));
        softmax_stable_inline(probs, classes);
        float p = probs[targets[b]];
        if (p < 1e-8f) p = 1e-8f;
        loss -= logf(p);
        free(probs);
    }
    return loss / (float)batch;
}

void cross_entropy_grad(const float *logits, const int *targets,
                        float *grad, int batch, int classes) {
    for (int b = 0; b < batch; b++) {
        float *probs = (float *)malloc(classes * sizeof(float));
        memcpy(probs, logits + b * classes, classes * sizeof(float));
        softmax_stable_inline(probs, classes);
        for (int c = 0; c < classes; c++) {
            grad[b * classes + c] = (probs[c] - (c == targets[b] ? 1.0f : 0.0f)) / (float)batch;
        }
        free(probs);
    }
}

/*
 * L5: Huber Loss (Smooth L1)
 * L(d) = 0.5*d^2         if |d| <= delta
 *        delta*(|d|-0.5*delta)  otherwise
 * Robust to outliers; used in object detection (Fast R-CNN).
 */
float huber_loss(const float *pred, const float *target, int n, float delta) {
    float loss = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = pred[i] - target[i];
        if (fabsf(d) <= delta)
            loss += 0.5f * d * d;
        else
            loss += delta * (fabsf(d) - 0.5f * delta);
    }
    return loss / (float)n;
}

float huber_loss_grad(const float *pred, const float *target, float *grad, int n, float delta) {
    float loss = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = pred[i] - target[i];
        if (fabsf(d) <= delta) {
            loss += 0.5f * d * d;
            grad[i] = d / (float)n;
        } else {
            loss += delta * (fabsf(d) - 0.5f * delta);
            grad[i] = (d > 0.0f ? delta : -delta) / (float)n;
        }
    }
    return loss / (float)n;
}

/*
 * L8: Focal Loss (Lin et al., ICCV 2017)
 * FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
 * Down-weights easy examples; focus on hard ones.
 * Used in RetinaNet for dense object detection.
 */
float focal_loss(const float *logits, const int *targets, int batch, int classes,
                 float gamma, float alpha) {
    float loss = 0.0f;
    for (int b = 0; b < batch; b++) {
        float *probs = (float *)malloc(classes * sizeof(float));
        memcpy(probs, logits + b * classes, classes * sizeof(float));
        softmax_stable_inline(probs, classes);
        float pt = probs[targets[b]];
        if (pt < 1e-8f) pt = 1e-8f;
        loss -= alpha * powf(1.0f - pt, gamma) * logf(pt);
        free(probs);
    }
    return loss / (float)batch;
}

void focal_loss_grad(const float *logits, const int *targets, float *grad,
                      int batch, int classes, float gamma, float alpha) {
    for (int b = 0; b < batch; b++) {
        float *probs = (float *)malloc(classes * sizeof(float));
        memcpy(probs, logits + b * classes, classes * sizeof(float));
        softmax_stable_inline(probs, classes);
        float pt = probs[targets[b]];
        if (pt < 1e-8f) pt = 1e-8f;
        float weight = alpha * powf(1.0f - pt, gamma);
        float focal_factor = weight * (gamma * (1.0f - pt) * logf(pt) + pt - 1.0f) / pt;
        for (int c = 0; c < classes; c++) {
            float indicator = (c == targets[b]) ? 1.0f : 0.0f;
            grad[b * classes + c] = focal_factor * (probs[c] - indicator) / (float)batch;
        }
        free(probs);
    }
}

/*
 * L4: KL Divergence D_KL(P||Q) = sum_i P(i) * (log P(i) - log Q(i))
 * Measures information lost when Q approximates P.
 * Always non-negative (Gibbs' inequality).
 */
float kl_divergence(const float *p_log_prob, const float *q_log_prob, int n) {
    float kl = 0.0f;
    for (int i = 0; i < n; i++)
        kl += expf(p_log_prob[i]) * (p_log_prob[i] - q_log_prob[i]);
    return kl;
}

/*
 * L7: Cosine Embedding Loss ¡ª for face verification, contrastive learning
 * L = 1 - cos(x1,x2)  if y==1 (similar)
 *     max(0, cos(x1,x2)-margin) if y==-1 (dissimilar)
 */
float cosine_embedding_loss(const float *x1, const float *x2, int dim, float y, float margin) {
    float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
    for (int i = 0; i < dim; i++) {
        dot += x1[i] * x2[i];
        norm1 += x1[i] * x1[i];
        norm2 += x2[i] * x2[i];
    }
    float cos_sim = dot / (sqrtf(norm1) * sqrtf(norm2) + 1e-8f);
    if (y > 0.0f)
        return 1.0f - cos_sim;
    else
        return fmaxf(0.0f, cos_sim - margin);
}

void cosine_embedding_grad(const float *x1, const float *x2, int dim, float y,
                            float margin, float *grad1, float *grad2) {
    float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
    for (int i = 0; i < dim; i++) {
        dot += x1[i] * x2[i];
        norm1 += x1[i] * x1[i];
        norm2 += x2[i] * x2[i];
    }
    float n1 = sqrtf(norm1) + 1e-8f, n2 = sqrtf(norm2) + 1e-8f;
    float cos_sim = dot / (n1 * n2);
    int skip = (y < 0.0f && cos_sim < margin);
    float scale = (y > 0.0f) ? -1.0f : (skip ? 0.0f : 1.0f);
    for (int i = 0; i < dim; i++) {
        float dcos_dx1 = (x2[i] / (n1 * n2)) - (cos_sim * x1[i] / (n1 * n1));
        float dcos_dx2 = (x1[i] / (n1 * n2)) - (cos_sim * x2[i] / (n2 * n2));
        grad1[i] = scale * dcos_dx1;
        grad2[i] = scale * dcos_dx2;
    }
}

/* L5: Softmax with numerical stability (subtract max) */
void softmax_stable(float *x, int n) {
    float maxv = x[0];
    for (int i = 1; i < n; i++) if (x[i] > maxv) maxv = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - maxv);
        sum += x[i];
    }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

void log_softmax(const float *x, float *out, int n) {
    float maxv = x[0];
    for (int i = 1; i < n; i++) if (x[i] > maxv) maxv = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += expf(x[i] - maxv);
    float log_sum = logf(sum) + maxv;
    for (int i = 0; i < n; i++) out[i] = x[i] - log_sum;
}
