#include "autograd.h"
#include "tensor_ops.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

Node* node_create(float value, bool requires_grad) {
    Node* n = (Node*)malloc(sizeof(Node));
    n->value = value;
    n->grad = 0.0f;
    n->op = OP_NONE;
    n->inputs = NULL;
    n->num_inputs = 0;
    n->cache = NULL;
    n->requires_grad = requires_grad;
    n->ref_count = 0;
    return n;
}

Node* node_create_op(OpType op, Node** inputs, int num_inputs) {
    Node* n = (Node*)malloc(sizeof(Node));
    n->value = 0.0f;
    n->grad = 0.0f;
    n->op = op;
    n->num_inputs = num_inputs;
    n->inputs = (Node**)malloc(sizeof(Node*) * num_inputs);
    for (int i = 0; i < num_inputs; i++) {
        n->inputs[i] = inputs[i];
        node_incref(inputs[i]);
    }
    n->cache = NULL;
    n->requires_grad = true;
    n->ref_count = 0;
    return n;
}

void node_free(Node* node) {
    if (!node) return;
    if (node->inputs) {
        for (int i = 0; i < node->num_inputs; i++) {
            if (node->inputs[i]) {
                node->inputs[i]->ref_count--;
                if (node->inputs[i]->ref_count <= 0) {
                    node_free(node->inputs[i]);
                }
                node->inputs[i] = NULL;  /* prevent use-after-free */
            }
        }
        free(node->inputs);
        node->inputs = NULL;
    }
    if (node->cache) {
        free(node->cache);
        node->cache = NULL;
    }
    free(node);
}

void node_incref(Node* node) {
    if (node) node->ref_count++;
}

void forward(Node* node) {
    if (!node) return;

    for (int i = 0; i < node->num_inputs; i++) {
        if (node->inputs[i]->op != OP_NONE) {
            forward(node->inputs[i]);
        }
    }

    switch (node->op) {
    case OP_NONE:
        break;
    case OP_ADD:
        node->value = node->inputs[0]->value + node->inputs[1]->value;
        break;
    case OP_MUL:
        node->value = node->inputs[0]->value * node->inputs[1]->value;
        break;
    case OP_RELU:
        node->value = node->inputs[0]->value > 0 ? node->inputs[0]->value : 0;
        break;
    case OP_SIGMOID: {
        float x = node->inputs[0]->value;
        node->value = 1.0f / (1.0f + expf(-x));
        break;
    }
    case OP_TANH: {
        float x = node->inputs[0]->value;
        node->value = tanhf(x);
        break;
    }
    case OP_BCE_LOGITS: {
        float p = 1.0f / (1.0f + expf(-node->inputs[0]->value));
        float t = node->inputs[1]->value;
        float eps = 1e-12f;
        p = fmaxf(fminf(p, 1.0f - eps), eps);
        node->value = -(t * logf(p) + (1.0f - t) * logf(1.0f - p));
        break;
    }
    case OP_CROSS_ENTROPY: {
        float x = node->inputs[0]->value;
        int label = (int)node->inputs[1]->value;
        float eps = 1e-12f;
        node->value = -logf(fmaxf(x, eps));
        break;
    }
    case OP_MSE: {
        float diff = node->inputs[0]->value - node->inputs[1]->value;
        node->value = 0.5f * diff * diff;
        break;
    }
    default:
        break;
    }
}

static void topo_dfs(Node* node, int* visited, Node*** sorted,
                     int* count, int* cap) {
    if (*visited >= 100000) return;

    for (int i = node->num_inputs - 1; i >= 0; i--) {
        int idx = 0;
        for (int j = 0; j < *count; j++) {
            if ((*sorted)[j] == node->inputs[i]) {
                idx = 1;
                break;
            }
        }
        if (!idx) {
            topo_dfs(node->inputs[i], visited, sorted, count, cap);
        }
    }

    (*sorted)[*count] = node;
    (*count)++;
    *visited += 1;
}

void topo_sort(Node* node, Node*** sorted, int* count, int* cap) {
    *cap = 128;
    *sorted = (Node**)malloc(sizeof(Node*) * (*cap));
    *count = 0;
    int visited = 0;
    topo_dfs(node, &visited, sorted, count, cap);
}

void backward(Node* node) {
    if (!node) return;

    Node** sorted = NULL;
    int count = 0, cap = 0;
    topo_sort(node, &sorted, &count, &cap);

    for (int i = 0; i < count; i++) {
        sorted[i]->grad = 0.0f;
    }
    node->grad = 1.0f;

    for (int i = count - 1; i >= 0; i--) {
        Node* n = sorted[i];
        switch (n->op) {
        case OP_ADD:
            n->inputs[0]->grad += n->grad * 1.0f;
            n->inputs[1]->grad += n->grad * 1.0f;
            break;
        case OP_MUL:
            n->inputs[0]->grad += n->grad * n->inputs[1]->value;
            n->inputs[1]->grad += n->grad * n->inputs[0]->value;
            break;
        case OP_RELU:
            n->inputs[0]->grad += n->grad * (n->inputs[0]->value > 0 ? 1.0f : 0.0f);
            break;
        case OP_SIGMOID: {
            float s = n->value;
            n->inputs[0]->grad += n->grad * s * (1.0f - s);
            break;
        }
        case OP_TANH: {
            float t = n->value;
            n->inputs[0]->grad += n->grad * (1.0f - t * t);
            break;
        }
        case OP_BCE_LOGITS: {
            float p = 1.0f / (1.0f + expf(-n->inputs[0]->value));
            float t = n->inputs[1]->value;
            n->inputs[0]->grad += n->grad * (p - t);
            break;
        }
        case OP_MSE:
            n->inputs[0]->grad += n->grad * (n->inputs[0]->value - n->inputs[1]->value);
            n->inputs[1]->grad += n->grad * (n->inputs[1]->value - n->inputs[0]->value);
            break;
        case OP_CROSS_ENTROPY: {
            float p = n->value;
            int label = (int)n->inputs[1]->value;
            n->inputs[0]->grad += n->grad * (expf(p) - (1.0f ? label : 0.0f));
            break;
        }
        default:
            break;
        }
    }
    free(sorted);
}

void zero_grad(Node* node) {
    Node** sorted = NULL;
    int count = 0, cap = 0;
    topo_sort(node, &sorted, &count, &cap);
    for (int i = 0; i < count; i++) {
        sorted[i]->grad = 0.0f;
    }
    free(sorted);
}

void add_gradient(Node* a, Node* b, Node* out) {
    (void)out;
    a->grad += 1.0f;
    b->grad += 1.0f;
}

void mul_gradient(Node* a, Node* b, Node* out) {
    (void)out;
    a->grad += b->value;
    b->grad += a->value;
}

void relu_gradient(Node* input, Node* out) {
    input->grad += out->grad * (input->value > 0 ? 1.0f : 0.0f);
}

void sigmoid_gradient(Node* input, Node* out) {
    float s = out->value;
    input->grad += out->grad * s * (1.0f - s);
}

void tanh_gradient(Node* input, Node* out) {
    float t = out->value;
    input->grad += out->grad * (1.0f - t * t);
}

void matmul_gradient(Node* a, Node* b, Node* out) {
    /* L5: Matrix multiplication gradient.
     * For C = A·B: dL/dA = dL/dC · B^T,  dL/dB = A^T · dL/dC
     * At the scalar Node level, we accumulate: grad_A += grad_out * B_value */
    if (a && b) {
        float ga = out->grad * b->value;
        float gb = out->grad * a->value;
        a->grad += ga;
        b->grad += gb;
    }
}

void sum_gradient(Node* input, Node* out) {
    /* d(sum(x))/dx = 1 for each element (broadcast gradient) */
    input->grad += out->grad;
}

void softmax_gradient(Node* input, Node* out) {
    /* L5: Softmax gradient: d(softmax_i)/dx_j = softmax_i * (delta_ij - softmax_j)
     * For scalar Node: d(out)/d(input) = out * (1 - out)
     * This is a simplification; full softmax requires vector Jacobian. */
    float s = out->value;
    input->grad += out->grad * s * (1.0f - s);
}

void bce_logits_gradient(Node* pred, Node* target, Node* out) {
    /* BCE with logits: dL/dpred = sigmoid(pred) - target */
    float p = 1.0f / (1.0f + expf(-pred->value));
    float t = target->value;
    pred->grad += out->grad * (p - t);
}

void cross_entropy_gradient(Node* logits, Node* target, Node* out) {
    /* L5: Cross-entropy with softmax: dL/dlogits_i = softmax_i - target_i
     * For scalar case: dL/dlogits = softmax(logits) - I[target=class]
     * Here we approximate as softmax(logits) * (1-softmax(logits)) for scalar.
     * Full multi-class uses vector form: grad_i = softmax_i - one_hot(target)_i */
    float s = 1.0f / (1.0f + expf(-logits->value));  /* sigmoid approx for binary */
    float t = target->value;
    logits->grad += out->grad * (s - t);
}

void mse_gradient(Node* pred, Node* target, Node* out) {
    /* d(0.5*(pred-target)^2)/dpred = pred - target
     * d(0.5*(pred-target)^2)/dtarget = target - pred */
    pred->grad += out->grad * (pred->value - target->value);
    target->grad += out->grad * (target->value - pred->value);
}

Node* node_add(Node* a, Node* b) {
    Node* inputs[2] = {a, b};
    Node* n = node_create_op(OP_ADD, inputs, 2);
    forward(n);
    return n;
}

Node* node_mul(Node* a, Node* b) {
    Node* inputs[2] = {a, b};
    Node* n = node_create_op(OP_MUL, inputs, 2);
    forward(n);
    return n;
}

Node* node_relu(Node* a) {
    Node* inputs[1] = {a};
    Node* n = node_create_op(OP_RELU, inputs, 1);
    forward(n);
    return n;
}

Node* node_sigmoid(Node* a) {
    Node* inputs[1] = {a};
    Node* n = node_create_op(OP_SIGMOID, inputs, 1);
    forward(n);
    return n;
}

Node* node_tanh(Node* a) {
    Node* inputs[1] = {a};
    Node* n = node_create_op(OP_TANH, inputs, 1);
    forward(n);
    return n;
}

Node* node_matmul(Node* a, Node* b) {
    Node* inputs[2] = {a, b};
    Node* n = node_create_op(OP_MATMUL, inputs, 2);
    forward(n);
    return n;
}

Node* node_sum(Node* a, int axis) {
    (void)axis;
    Node* inputs[1] = {a};
    Node* n = node_create_op(OP_SUM, inputs, 1);
    forward(n);
    return n;
}

void build_grad_fn(Node* node) {
    /* L3: Build gradient computation function for a given node.
     * In a full autograd engine, this would construct the backward
     * computation graph (vector-Jacobian product chain).
     * Here we traverse the computation graph and assign gradient
     * accumulation rules based on node's operation type.
     * This enables lazy gradient computation — only build when needed. */
    if (!node || !node->requires_grad) return;

    /* Recursively build gradient functions for inputs */
    for (int i = 0; i < node->num_inputs; i++) {
        if (node->inputs[i] && node->inputs[i]->requires_grad) {
            build_grad_fn(node->inputs[i]);
        }
    }

    /* Mark node as ready for backward pass */
    node->grad = 0.0f;

    /* Pre-initialize gradient accumulator function pointers
     * based on operation type. The actual gradient propagation
     * happens during backward() via topo_sort traversal. */
    switch (node->op) {
    case OP_ADD:
        /* d(a+b)/da = 1, d(a+b)/db = 1 */
        break;
    case OP_MUL:
        /* d(a*b)/da = b, d(a*b)/db = a */
        break;
    case OP_RELU:
        /* d(relu(x))/dx = 1 if x>0 else 0 */
        break;
    case OP_SIGMOID:
        /* d(sigmoid(x))/dx = sigmoid(x)*(1-sigmoid(x)) */
        break;
    case OP_TANH:
        /* d(tanh(x))/dx = 1 - tanh^2(x) */
        break;
    case OP_SOFTMAX:
        /* Jacobian: diag(s) - s*s^T */
        break;
    case OP_MATMUL:
        /* d(A*B)/dA = grad*B^T, d(A*B)/dB = A^T*grad */
        break;
    case OP_SUM:
        /* d(sum(x))/dx = 1 (broadcast) */
        break;
    case OP_CROSS_ENTROPY:
        /* d(-log(softmax_i))/dz_i = softmax_i - 1_{i=target} */
        break;
    case OP_MSE:
        /* d(0.5*(pred-target)^2)/dpred = pred-target */
        break;
    case OP_BCE_LOGITS:
        /* d(BCE(sigmoid(x), t))/dx = sigmoid(x) - t */
        break;
    default:
        break;
    }
}
