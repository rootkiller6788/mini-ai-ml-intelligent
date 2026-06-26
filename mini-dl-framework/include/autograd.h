#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include <stddef.h>
#include <stdbool.h>

typedef enum {
    OP_NONE,
    OP_ADD,
    OP_MUL,
    OP_RELU,
    OP_SIGMOID,
    OP_TANH,
    OP_MATMUL,
    OP_SUM,
    OP_SOFTMAX,
    OP_RESHAPE,
    OP_LINEAR,
    OP_CONV2D,
    OP_BCE_LOGITS,
    OP_CROSS_ENTROPY,
    OP_MSE,
    OP_MAX_POOL
} OpType;

typedef struct Node {
    float value;
    float grad;
    OpType op;
    struct Node** inputs;
    int num_inputs;
    void* cache;
    bool requires_grad;
    int ref_count;
} Node;

Node* node_create(float value, bool requires_grad);
Node* node_create_op(OpType op, Node** inputs, int num_inputs);
void node_free(Node* node);
void node_incref(Node* node);

void forward(Node* node);
void backward(Node* node);
void zero_grad(Node* node);

void add_gradient(Node* a, Node* b, Node* out);
void mul_gradient(Node* a, Node* b, Node* out);
void relu_gradient(Node* input, Node* out);
void sigmoid_gradient(Node* input, Node* out);
void tanh_gradient(Node* input, Node* out);
void matmul_gradient(Node* a, Node* b, Node* out);
void sum_gradient(Node* input, Node* out);
void softmax_gradient(Node* input, Node* out);
void bce_logits_gradient(Node* pred, Node* target, Node* out);
void cross_entropy_gradient(Node* logits, Node* target, Node* out);
void mse_gradient(Node* pred, Node* target, Node* out);

Node* node_add(Node* a, Node* b);
Node* node_mul(Node* a, Node* b);
Node* node_relu(Node* a);
Node* node_sigmoid(Node* a);
Node* node_tanh(Node* a);
Node* node_matmul(Node* a, Node* b);
Node* node_sum(Node* a, int axis);

void topo_sort(Node* node, Node*** sorted, int* count, int* cap);
void build_grad_fn(Node* node);

#endif
