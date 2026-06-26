#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cnn_models.h"

int main(void) {
    printf("=== mini-model-arch: CNN Example ===\n\n");
    srand((unsigned)time(NULL));

    printf("1. LeNet-5 Forward Pass (batch=1, 28x28 grayscale)\n");
    LeNet5 *lenet = lenet5_create();
    Tensor4D input;
    tensor4d_init(&input, 1, 1, 28, 28);
    for (int i = 0; i < 28 * 28; i++)
        input.data[i] = ((float)rand() / RAND_MAX);
    Tensor4D *logits = lenet5_forward(lenet, &input);
    printf("   Output: %dx%dx%dx%d\n", logits->batch, logits->channels,
           logits->height, logits->width);
    tensor4d_free(logits);
    tensor4d_free(&input);
    lenet5_free(lenet);

    printf("\n2. ResBlock with Identity (in=64, out=64, stride=1)\n");
    ResBlock *rb = resblock_create(64, 64, 1, 0);
    Tensor4D in_rb;
    tensor4d_init(&in_rb, 2, 64, 32, 32);
    for (int i = 0; i < 2 * 64 * 32 * 32; i++)
        in_rb.data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    Tensor4D *out_rb = resblock_forward(rb, &in_rb);
    printf("   Output: %dx%dx%dx%d\n", out_rb->batch, out_rb->channels,
           out_rb->height, out_rb->width);
    tensor4d_free(out_rb);
    tensor4d_free(&in_rb);
    resblock_free(rb);

    printf("\n3. ResBlock with Projection (in=64, out=128, stride=2)\n");
    ResBlock *rbp = resblock_create(64, 128, 2, 0);
    Tensor4D in_rbp;
    tensor4d_init(&in_rbp, 1, 64, 56, 56);
    for (int i = 0; i < 64 * 56 * 56; i++)
        in_rbp.data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    Tensor4D *out_rbp = resblock_forward(rbp, &in_rbp);
    printf("   Output: %dx%dx%dx%d\n", out_rbp->batch, out_rbp->channels,
           out_rbp->height, out_rbp->width);
    tensor4d_free(out_rbp);
    tensor4d_free(&in_rbp);
    resblock_free(rbp);

    printf("\n4. Bottleneck Block (in=256, out=256, stride=1)\n");
    ResBlock *bb = resblock_create(256, 256, 1, 1);
    Tensor4D in_bb;
    tensor4d_init(&in_bb, 1, 256, 56, 56);
    for (int i = 0; i < 256 * 56 * 56; i++)
        in_bb.data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.05f;
    Tensor4D *out_bb = resblock_forward(bb, &in_bb);
    printf("   Output: %dx%dx%dx%d\n", out_bb->batch, out_bb->channels,
           out_bb->height, out_bb->width);
    tensor4d_free(out_bb);
    tensor4d_free(&in_bb);
    resblock_free(bb);

    printf("\n5. Inception Module (in=192, branches: 1x1->64, 3x3->128, 5x5->32, pool->32)\n");
    InceptionModule *im = inception_create(192, 64, 96, 128, 16, 32, 32);
    Tensor4D in_im;
    tensor4d_init(&in_im, 1, 192, 28, 28);
    for (int i = 0; i < 192 * 28 * 28; i++)
        in_im.data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    Tensor4D *out_im = inception_forward(im, &in_im);
    printf("   Output: %dx%dx%dx%d\n", out_im->batch, out_im->channels,
           out_im->height, out_im->width);
    tensor4d_free(out_im);
    tensor4d_free(&in_im);
    inception_free(im);

    printf("\n6. Depthwise Separable Conv (in=64, out=128, k=3, s=1)\n");
    DepthwiseSepConv *dsc = dwsepconv_create(64, 128, 3, 1, 1);
    Tensor4D in_ds;
    tensor4d_init(&in_ds, 1, 64, 32, 32);
    for (int i = 0; i < 64 * 32 * 32; i++)
        in_ds.data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    Tensor4D *out_ds = dwsepconv_forward(dsc, &in_ds);
    printf("   Output: %dx%dx%dx%d (params: depthwise=%d vs standard=%d)\n",
           out_ds->batch, out_ds->channels, out_ds->height, out_ds->width,
           64 * 3 * 3 + 64 * 128, 64 * 128 * 3 * 3);
    tensor4d_free(out_ds);
    tensor4d_free(&in_ds);
    dwsepconv_free(dsc);

    printf("\nAll CNN tests passed.\n");
    return 0;
}
