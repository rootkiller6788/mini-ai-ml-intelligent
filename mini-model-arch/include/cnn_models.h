#ifndef CNN_MODELS_H
#define CNN_MODELS_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int batch, channels, height, width;
    float *data;
} Tensor4D;

typedef struct {
    int out_channels, in_channels, kernel_size, stride, padding;
    float *weights, *bias;
} Conv2D;

typedef struct {
    int kernel_size, stride;
} Pool2D;

typedef struct {
    int in_features, out_features;
    float *weights, *bias;
} Linear;

typedef struct {
    Tensor4D *features;
    int num_classes;
    Linear   *fc1, *fc2, *fc3;
    Conv2D   *conv1, *conv2;
    Pool2D   *pool1, *pool2;
} LeNet5;

typedef struct {
    int num_conv_layers;
    Conv2D **convs;
    Pool2D  *pool;
    int     *conv_channels;
} VGGBlock;

typedef struct {
    Conv2D *conv1, *conv2;
    Conv2D *shortcut;
    int     use_projection;
} ResBlock;

typedef struct {
    Conv2D *conv1, *conv2, *conv3;
    Conv2D *shortcut;
    int     expansion;
} BottleneckBlock;

typedef struct {
    Conv2D *conv1x1_a, *conv3x3, *conv5x5, *conv1x1_b;
    Pool2D *pool_proj;
} InceptionModule;

typedef struct {
    Conv2D *depthwise;
    Conv2D *pointwise;
} DepthwiseSepConv;

void  tensor4d_init(Tensor4D *t, int b, int c, int h, int w);
void  tensor4d_free(Tensor4D *t);
float tensor4d_get(const Tensor4D *t, int b, int c, int y, int x);
void  tensor4d_set(Tensor4D *t, int b, int c, int y, int x, float v);

Conv2D *conv2d_create(int in_c, int out_c, int k, int s, int p);
void    conv2d_free(Conv2D *c);
Tensor4D *conv2d_forward(const Conv2D *c, const Tensor4D *in);

Tensor4D *pool2d_forward(const Pool2D *p, const Tensor4D *in, int mode);

LeNet5 *lenet5_create(void);
void    lenet5_free(LeNet5 *l);
Tensor4D *lenet5_forward(const LeNet5 *l, const Tensor4D *in);

ResBlock *resblock_create(int in_c, int out_c, int stride, int bottleneck);
void      resblock_free(ResBlock *r);
Tensor4D  *resblock_forward(const ResBlock *r, const Tensor4D *in);

InceptionModule *inception_create(int in_c, int c1, int c3r, int c3, int c5r, int c5, int pool_proj);
void             inception_free(InceptionModule *im);
Tensor4D         *inception_forward(const InceptionModule *im, const Tensor4D *in);

DepthwiseSepConv *dwsepconv_create(int in_c, int out_c, int k, int s, int p);
void              dwsepconv_free(DepthwiseSepConv *d);
Tensor4D          *dwsepconv_forward(const DepthwiseSepConv *d, const Tensor4D *in);

void relu_inplace(Tensor4D *t);
void softmax_inplace(Tensor4D *t_channels);

#endif
