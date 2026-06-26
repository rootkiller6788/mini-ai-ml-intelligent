/*
 * mini-training-system — Core Benchmarks
 *
 * Benchmarks: distributed training, mixed precision, checkpointing,
 *             hyperparameter tuning, training loop.
 */
#include "../include/distributed_train.h"
#include "../include/mixed_precision.h"
#include "../include/checkpoint_save.h"
#include "../include/hyperparam_tune.h"
#include "../include/training_loop.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    return (double)clock() / (double)(CLOCKS_PER_SEC / 1000);
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 5000;
    double t0, t1;
    printf("=== mini-training-system Benchmarks (N=%d) ===\n\n", N);

    /* ── Ring All-Reduce ── */
    {
        float buf[1024];
        for (int i = 0; i < 1024; i++) buf[i] = 1.0f;
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            ring_all_reduce(buf, 1024, RING_REDUCE_OP_SUM, 0, 1);
        }
        t1 = now_ms();
        printf("  ring_all_reduce:    %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
    }

    /* ── FP32->FP16 Conversion ── */
    {
        float src[4096];
        uint16_t dst[4096];
        for (int i = 0; i < 4096; i++) src[i] = 1.0f;
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            fp32_to_fp16(src, dst, 4096);
            fp16_to_fp32(dst, src, 4096);
        }
        t1 = now_ms();
        printf("  fp32<->fp16 conv:   %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
    }

    /* ── FP32->BF16 Conversion ── */
    {
        float src[4096];
        uint16_t dst[4096];
        for (int i = 0; i < 4096; i++) src[i] = 1.0f;
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            fp32_to_bf16(src, dst, 4096);
            bf16_to_fp32(dst, src, 4096);
        }
        t1 = now_ms();
        printf("  fp32<->bf16 conv:   %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
    }

    /* ── MP Loss Scale Update ── */
    {
        mp_config_t cfg = {0};
        cfg.mode = MP_MODE_FP16;
        cfg.init_loss_scale = 65536.0f;
        mp_context_t ctx;
        mp_init(&ctx, &cfg);
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            mp_update_loss_scale(&ctx, false);
        }
        t1 = now_ms();
        printf("  mp_loss_scale:      %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
    }

    /* ── Checkpoint CRC32 ── */
    {
        uint8_t data[4096];
        memset(data, 0xAB, sizeof(data));
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            ckpt_crc32(data, sizeof(data));
        }
        t1 = now_ms();
        printf("  ckpt_crc32:         %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
    }

    /* ── HPT Suggest ── */
    {
        hpt_config_t cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.method = HPT_SEARCH_RANDOM;
        cfg.num_trials = 100;
        hpt_context_t ctx;
        hpt_init(&ctx, &cfg);
        hpt_add_float(&ctx, "lr", 0.0001f, 0.1f, 0.0f);
        hpt_add_int(&ctx, "bs", 16, 256, 0);
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            hpt_suggest(&ctx);
        }
        t1 = now_ms();
        printf("  hpt_suggest:        %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
        hpt_free(&ctx);
    }

    /* ── Training Loop Init ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 20; r++) {
            tl_train_config_t tc; memset(&tc, 0, sizeof(tc));
            tc.max_epochs = 1; tc.batch_size = 16;
            tl_optimizer_config_t oc; memset(&oc, 0, sizeof(oc));
            oc.type = TL_OPTIM_ADAM; oc.learning_rate = 0.001f;
            tl_scheduler_config_t sc; memset(&sc, 0, sizeof(sc));
            sc.type = TL_SCHEDULER_COSINE; sc.base_lr = 0.001f;
            tl_context_t ctx;
            tl_init(&ctx, &tc, &oc, &sc);
            tl_free(&ctx);
        }
        t1 = now_ms();
        printf("  tl_init+free:       %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 20, t1 - t0, (t1 - t0) / (double)(N / 20) * 1000.0);
    }

    /* ── LR Schedule ── */
    {
        t0 = now_ms();
        for (int r = 0; r < N / 10; r++) {
            tl_lr_cosine(50, 100, 0.1f, 0.001f);
        }
        t1 = now_ms();
        printf("  tl_lr_cosine:       %d ops in %.1f ms  (%.1f µs/op)\n",
               N / 10, t1 - t0, (t1 - t0) / (double)(N / 10) * 1000.0);
    }

    printf("\nDone.\n");
    return 0;
}
