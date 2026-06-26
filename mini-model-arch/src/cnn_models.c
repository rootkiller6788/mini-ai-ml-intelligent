#include "cnn_models.h"

void tensor4d_init(Tensor4D *t, int b, int c, int h, int w) {
    t->batch = b; t->channels = c; t->height = h; t->width = w;
    t->data = (float *)calloc(b * c * h * w, sizeof(float));
}
void tensor4d_free(Tensor4D *t) { free(t->data); t->data = NULL; }
float tensor4d_get(const Tensor4D *t, int b, int c, int y, int x) {
    return t->data[((b * t->channels + c) * t->height + y) * t->width + x];
}
void tensor4d_set(Tensor4D *t, int b, int c, int y, int x, float v) {
    t->data[((b * t->channels + c) * t->height + y) * t->width + x] = v;
}

Conv2D *conv2d_create(int in_c, int out_c, int k, int s, int p) {
    Conv2D *c = (Conv2D *)malloc(sizeof(Conv2D));
    c->in_channels = in_c; c->out_channels = out_c;
    c->kernel_size = k; c->stride = s; c->padding = p;
    int w_sz = out_c * in_c * k * k;
    c->weights = (float *)malloc(w_sz * sizeof(float));
    c->bias = (float *)calloc(out_c, sizeof(float));
    for (int i = 0; i < w_sz; i++)
        c->weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    return c;
}
void conv2d_free(Conv2D *c) { free(c->weights); free(c->bias); free(c); }

Tensor4D *conv2d_forward(const Conv2D *c, const Tensor4D *in) {
    int out_h = (in->height + 2 * c->padding - c->kernel_size) / c->stride + 1;
    int out_w = (in->width + 2 * c->padding - c->kernel_size) / c->stride + 1;
    Tensor4D *out = (Tensor4D *)malloc(sizeof(Tensor4D));
    tensor4d_init(out, in->batch, c->out_channels, out_h, out_w);

    for (int b = 0; b < in->batch; b++) {
        for (int oc = 0; oc < c->out_channels; oc++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    float sum = c->bias[oc];
                    for (int ic = 0; ic < c->in_channels; ic++) {
                        for (int kh = 0; kh < c->kernel_size; kh++) {
                            for (int kw = 0; kw < c->kernel_size; kw++) {
                                int ih = oh * c->stride + kh - c->padding;
                                int iw = ow * c->stride + kw - c->padding;
                                if (ih >= 0 && ih < in->height && iw >= 0 && iw < in->width) {
                                    int widx = ((oc * c->in_channels + ic) * c->kernel_size + kh) * c->kernel_size + kw;
                                    sum += c->weights[widx] * tensor4d_get(in, b, ic, ih, iw);
                                }
                            }
                        }
                    }
                    tensor4d_set(out, b, oc, oh, ow, sum);
                }
            }
        }
    }
    return out;
}

Tensor4D *pool2d_forward(const Pool2D *p, const Tensor4D *in, int mode) {
    int out_h = (in->height - p->kernel_size) / p->stride + 1;
    int out_w = (in->width - p->kernel_size) / p->stride + 1;
    Tensor4D *out = (Tensor4D *)malloc(sizeof(Tensor4D));
    tensor4d_init(out, in->batch, in->channels, out_h, out_w);

    for (int b = 0; b < in->batch; b++) {
        for (int c = 0; c < in->channels; c++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    float val = (mode == 0) ? -1e9f : 0.0f;
                    for (int kh = 0; kh < p->kernel_size; kh++) {
                        for (int kw = 0; kw < p->kernel_size; kw++) {
                            int ih = oh * p->stride + kh;
                            int iw = ow * p->stride + kw;
                            float v = tensor4d_get(in, b, c, ih, iw);
                            if (mode == 0) val = fmaxf(val, v);
                            else val += v;
                        }
                    }
                    if (mode == 1) val /= (p->kernel_size * p->kernel_size);
                    tensor4d_set(out, b, c, oh, ow, val);
                }
            }
        }
    }
    return out;
}

void relu_inplace(Tensor4D *t) {
    int n = t->batch * t->channels * t->height * t->width;
    for (int i = 0; i < n; i++)
        if (t->data[i] < 0) t->data[i] = 0;
}

void softmax_inplace(Tensor4D *t) {
    int n = t->batch * t->height * t->width;
    for (int b = 0; b < t->batch; b++) {
        for (int h = 0; h < t->height; h++) {
            for (int w = 0; w < t->width; w++) {
                float maxv = -1e9f, sum = 0;
                for (int c = 0; c < t->channels; c++)
                    maxv = fmaxf(maxv, tensor4d_get(t, b, c, h, w));
                for (int c = 0; c < t->channels; c++) {
                    float v = expf(tensor4d_get(t, b, c, h, w) - maxv);
                    tensor4d_set(t, b, c, h, w, v);
                    sum += v;
                }
                for (int c = 0; c < t->channels; c++)
                    tensor4d_set(t, b, c, h, w, tensor4d_get(t, b, c, h, w) / sum);
            }
        }
    }
}

LeNet5 *lenet5_create(void) {
    LeNet5 *l = (LeNet5 *)malloc(sizeof(LeNet5));
    l->conv1 = conv2d_create(1, 6, 5, 1, 0);
    l->conv2 = conv2d_create(6, 16, 5, 1, 0);
    l->pool1 = (Pool2D *)malloc(sizeof(Pool2D)); l->pool1->kernel_size = 2; l->pool1->stride = 2;
    l->pool2 = (Pool2D *)malloc(sizeof(Pool2D)); l->pool2->kernel_size = 2; l->pool2->stride = 2;
    l->fc1 = (Linear *)malloc(sizeof(Linear)); l->fc1->in_features = 400; l->fc1->out_features = 120;
    l->fc1->weights = (float *)calloc(400 * 120, sizeof(float));
    l->fc1->bias = (float *)calloc(120, sizeof(float));
    l->fc2 = (Linear *)malloc(sizeof(Linear)); l->fc2->in_features = 120; l->fc2->out_features = 84;
    l->fc2->weights = (float *)calloc(120 * 84, sizeof(float));
    l->fc2->bias = (float *)calloc(84, sizeof(float));
    l->fc3 = (Linear *)malloc(sizeof(Linear)); l->fc3->in_features = 84; l->fc3->out_features = 10;
    l->fc3->weights = (float *)calloc(84 * 10, sizeof(float));
    l->fc3->bias = (float *)calloc(10, sizeof(float));
    l->num_classes = 10;
    return l;
}
void lenet5_free(LeNet5 *l) {
    conv2d_free(l->conv1); conv2d_free(l->conv2);
    free(l->pool1); free(l->pool2);
    free(l->fc1->weights); free(l->fc1->bias); free(l->fc1);
    free(l->fc2->weights); free(l->fc2->bias); free(l->fc2);
    free(l->fc3->weights); free(l->fc3->bias); free(l->fc3);
    free(l);
}
Tensor4D *lenet5_forward(const LeNet5 *l, const Tensor4D *in) {
    Tensor4D *x = conv2d_forward(l->conv1, in); relu_inplace(x);
    Tensor4D *p1 = pool2d_forward(l->pool1, x, 0); tensor4d_free(x);
    x = conv2d_forward(l->conv2, p1); relu_inplace(x); tensor4d_free(p1);
    Tensor4D *p2 = pool2d_forward(l->pool2, x, 0); tensor4d_free(x);
    return p2;
}

ResBlock *resblock_create(int in_c, int out_c, int stride, int bottleneck) {
    ResBlock *r = (ResBlock *)malloc(sizeof(ResBlock));
    if (bottleneck) {
        r->conv1 = conv2d_create(in_c, out_c / 4, 1, 1, 0);
        r->conv2 = conv2d_create(out_c / 4, out_c / 4, 3, stride, 1);
        r->conv3 = conv2d_create(out_c / 4, out_c, 1, 1, 0);
    } else {
        r->conv1 = conv2d_create(in_c, out_c, 3, stride, 1);
        r->conv2 = conv2d_create(out_c, out_c, 3, 1, 1);
        r->conv3 = NULL;
    }
    r->use_projection = (in_c != out_c || stride != 1);
    if (r->use_projection)
        r->shortcut = conv2d_create(in_c, out_c, 1, stride, 0);
    else
        r->shortcut = NULL;
    return r;
}
void resblock_free(ResBlock *r) {
    conv2d_free(r->conv1); conv2d_free(r->conv2);
    if (r->conv3) conv2d_free(r->conv3);
    if (r->shortcut) conv2d_free(r->shortcut);
    free(r);
}
Tensor4D *resblock_forward(const ResBlock *r, const Tensor4D *in) {
    Tensor4D *x = conv2d_forward(r->conv1, in); relu_inplace(x);
    Tensor4D *y = conv2d_forward(r->conv2, x); relu_inplace(y); tensor4d_free(x);
    if (r->conv3) { x = conv2d_forward(r->conv3, y); tensor4d_free(y); y = x; }
    Tensor4D *identity;
    if (r->use_projection) identity = conv2d_forward(r->shortcut, in);
    else {
        identity = (Tensor4D *)malloc(sizeof(Tensor4D));
        tensor4d_init(identity, in->batch, in->channels, in->height, in->width);
        memcpy(identity->data, in->data, in->batch * in->channels * in->height * in->width * sizeof(float));
    }
    int n = y->batch * y->channels * y->height * y->width;
    for (int i = 0; i < n; i++) y->data[i] += identity->data[i];
    relu_inplace(y); tensor4d_free(identity);
    return y;
}

InceptionModule *inception_create(int in_c, int c1, int c3r, int c3, int c5r, int c5, int pool_proj) {
    InceptionModule *im = (InceptionModule *)malloc(sizeof(InceptionModule));
    im->conv1x1_a = conv2d_create(in_c, c1, 1, 1, 0);
    im->conv3x3   = conv2d_create(c3r, c3, 3, 1, 1);
    im->conv5x5   = conv2d_create(c5r, c5, 5, 1, 2);
    im->conv1x1_b = conv2d_create(in_c, pool_proj, 1, 1, 0);
    im->pool_proj = (Pool2D *)malloc(sizeof(Pool2D));
    im->pool_proj->kernel_size = 3; im->pool_proj->stride = 1;
    return im;
}
void inception_free(InceptionModule *im) {
    conv2d_free(im->conv1x1_a); conv2d_free(im->conv3x3);
    conv2d_free(im->conv5x5); conv2d_free(im->conv1x1_b);
    free(im->pool_proj); free(im);
}
Tensor4D *inception_forward(const InceptionModule *im, const Tensor4D *in) {
    Tensor4D *b1 = conv2d_forward(im->conv1x1_a, in); relu_inplace(b1);
    Tensor4D *tmp = conv2d_create(im->conv3x3->in_channels, im->conv3x3->in_channels, 1, 1, 0);
    Tensor4D *b2r = conv2d_forward(tmp, in); relu_inplace(b2r);
    conv2d_free(tmp);
    Tensor4D *b2 = conv2d_forward(im->conv3x3, b2r); relu_inplace(b2); tensor4d_free(b2r);
    tmp = conv2d_create(im->conv5x5->in_channels, im->conv5x5->in_channels, 1, 1, 0);
    Tensor4D *b3r = conv2d_forward(tmp, in); relu_inplace(b3r);
    conv2d_free(tmp);
    Tensor4D *b3 = conv2d_forward(im->conv5x5, b3r); relu_inplace(b3); tensor4d_free(b3r);
    Tensor4D *p = pool2d_forward(im->pool_proj, in, 0);
    Tensor4D *b4 = conv2d_forward(im->conv1x1_b, p); relu_inplace(b4); tensor4d_free(p);
    int out_c = b1->channels + b2->channels + b3->channels + b4->channels;
    Tensor4D *out = (Tensor4D *)malloc(sizeof(Tensor4D));
    tensor4d_init(out, in->batch, out_c, b1->height, b1->width);
    int offset = 0;
    for (int i = 0; i < b1->channels; i++) {
        memcpy(out->data + offset, b1->data + i * b1->height * b1->width, b1->height * b1->width * sizeof(float));
        offset += b1->height * b1->width;
    }
    tensor4d_free(b1); tensor4d_free(b2); tensor4d_free(b3); tensor4d_free(b4);
    return out;
}

DepthwiseSepConv *dwsepconv_create(int in_c, int out_c, int k, int s, int p) {
    DepthwiseSepConv *d = (DepthwiseSepConv *)malloc(sizeof(DepthwiseSepConv));
    d->depthwise = conv2d_create(in_c, in_c, k, s, p);
    d->pointwise = conv2d_create(in_c, out_c, 1, 1, 0);
    return d;
}
void dwsepconv_free(DepthwiseSepConv *d) {
    conv2d_free(d->depthwise); conv2d_free(d->pointwise); free(d);
}
Tensor4D *dwsepconv_forward(const DepthwiseSepConv *d, const Tensor4D *in) {
    Tensor4D *x = conv2d_forward(d->depthwise, in);
    Tensor4D *y = conv2d_forward(d->pointwise, x);
    tensor4d_free(x);
    return y;
}
