#include "audio_whisper.h"
#include "clip_contrastive.h"
#include "image_generation.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

void mm_mel_filterbank_init(mm_mel_filterbank_t* fbank, int n_mels, int n_fft,
                            int sample_rate) {
    fbank->n_mels = n_mels;
    fbank->n_fft = n_fft;
    fbank->hop_length = MM_WHISPER_HOP_LENGTH;
    fbank->sample_rate = sample_rate;

    int n_freqs = n_fft / 2 + 1;
    fbank->mel_filters = (float*)calloc((size_t)n_mels * n_freqs, sizeof(float));

    float f_min = 0.0f;
    float f_max = (float)sample_rate / 2.0f;
    float mel_min = 1127.0f * logf(1.0f + f_min / 700.0f);
    float mel_max = 1127.0f * logf(1.0f + f_max / 700.0f);

    for (int m = 0; m < n_mels; m++) {
        float mel_center = mel_min + (mel_max - mel_min) * (float)(m + 1) / (float)(n_mels + 1);
        float mel_left = mel_min + (mel_max - mel_min) * (float)m / (float)(n_mels + 1);
        float mel_right = mel_min + (mel_max - mel_min) * (float)(m + 2) / (float)(n_mels + 1);

        float fc = 700.0f * (expf(mel_center / 1127.0f) - 1.0f);
        float fl = 700.0f * (expf(mel_left / 1127.0f) - 1.0f);
        float fr = 700.0f * (expf(mel_right / 1127.0f) - 1.0f);

        for (int k = 0; k < n_freqs; k++) {
            float freq = (float)k * (float)sample_rate / (float)n_fft;
            if (freq < fl) fbank->mel_filters[m * n_freqs + k] = 0.0f;
            else if (freq <= fc) fbank->mel_filters[m * n_freqs + k] = (freq - fl) / (fc - fl);
            else if (freq <= fr) fbank->mel_filters[m * n_freqs + k] = (fr - freq) / (fr - fc);
            else fbank->mel_filters[m * n_freqs + k] = 0.0f;
        }
    }
}

void mm_mel_filterbank_free(mm_mel_filterbank_t* fbank) {
    free(fbank->mel_filters);
}

void mm_audio_stft(const float* audio, int audio_len, int n_fft, int hop_len,
                   float* spec_real, float* spec_imag, int* n_frames, int* n_freqs) {
    int nf = n_fft / 2 + 1;
    int nfr = (audio_len - n_fft) / hop_len + 1;
    if (nfr < 1) nfr = 1;

    *n_frames = nfr;
    *n_freqs = nf;

    float* window = (float*)malloc((size_t)n_fft * sizeof(float));
    for (int i = 0; i < n_fft; i++) {
        window[i] = 0.5f * (1.0f - cosf(2.0f * 3.14159265f * (float)i / (float)(n_fft - 1)));
    }

    for (int f = 0; f < nfr; f++) {
        int start = f * hop_len;
        for (int k = 0; k < nf; k++) {
            float real = 0.0f, imag = 0.0f;
            for (int n = 0; n < n_fft; n++) {
                int idx = start + n;
                float sample = (idx < audio_len && idx >= 0) ? audio[idx] * window[n] : 0.0f;
                float angle = -2.0f * 3.14159265f * (float)k * (float)n / (float)n_fft;
                real += sample * cosf(angle);
                imag += sample * sinf(angle);
            }
            spec_real[f * nf + k] = real;
            spec_imag[f * nf + k] = imag;
        }
    }
    free(window);
}

void mm_audio_mel_spectrogram(const float* audio, int audio_len,
                              const mm_mel_filterbank_t* fbank,
                              mm_mel_spectrogram_t* mel) {
    int n_fft = fbank->n_fft;
    int hop_len = fbank->hop_length;
    int n_mels = fbank->n_mels;
    int n_freqs = n_fft / 2 + 1;
    int n_frames;
    int nf_buf;

    float* spec_r = (float*)malloc((size_t)((audio_len / hop_len + 1) * n_freqs) * sizeof(float));
    float* spec_i = (float*)malloc((size_t)((audio_len / hop_len + 1) * n_freqs) * sizeof(float));
    mm_audio_stft(audio, audio_len, n_fft, hop_len, spec_r, spec_i, &n_frames, &nf_buf);

    mel->n_frames = n_frames;
    mel->n_mels = n_mels;
    mel->data = (float*)calloc((size_t)n_frames * n_mels, sizeof(float));

    for (int f = 0; f < n_frames; f++) {
        for (int m = 0; m < n_mels; m++) {
            float sum = 0.0f;
            for (int k = 0; k < nf_buf; k++) {
                float power = spec_r[f * nf_buf + k] * spec_r[f * nf_buf + k]
                            + spec_i[f * nf_buf + k] * spec_i[f * nf_buf + k];
                sum += power * fbank->mel_filters[m * nf_buf + k];
            }
            mel->data[f * n_mels + m] = logf(sum + 1e-10f);
        }
    }

    float global_mean = 0.0f, global_std = 0.0f;
    int total = n_frames * n_mels;
    for (int i = 0; i < total; i++) global_mean += mel->data[i];
    global_mean /= (float)total;
    for (int i = 0; i < total; i++) {
        float diff = mel->data[i] - global_mean;
        global_std += diff * diff;
    }
    global_std = sqrtf(global_std / (float)total) + 1e-6f;
    for (int i = 0; i < total; i++) {
        mel->data[i] = (mel->data[i] - global_mean) / global_std;
    }

    free(spec_r);
    free(spec_i);
}

void mm_mel_spectrogram_free(mm_mel_spectrogram_t* mel) {
    free(mel->data);
}

float mm_vad_energy(const float* audio, int len) {
    float energy = 0.0f;
    for (int i = 0; i < len; i++) energy += audio[i] * audio[i];
    return energy / (float)len;
}

int mm_vad_is_speech(const float* audio, int len, float threshold_ms) {
    float energy = mm_vad_energy(audio, len);
    int zero_cross = 0;
    for (int i = 1; i < len; i++) {
        if ((audio[i] >= 0 && audio[i - 1] < 0) || (audio[i] < 0 && audio[i - 1] >= 0))
            zero_cross++;
    }
    return (energy > threshold_ms) ? 1 : 0;
}

void mm_vad_split(const float* audio, int total_len, int chunk_ms,
                  float threshold, int** speech_starts, int** speech_ends,
                  int* num_segments) {
    int chunk_size = chunk_ms * MM_WHISPER_SAMPLE_RATE / 1000;
    int num_chunks = total_len / chunk_size + 1;
    *num_segments = 0;

    *speech_starts = (int*)malloc((size_t)num_chunks * sizeof(int));
    *speech_ends = (int*)malloc((size_t)num_chunks * sizeof(int));

    int in_speech = 0;
    for (int i = 0; i < num_chunks; i++) {
        int start = i * chunk_size;
        int len = (start + chunk_size <= total_len) ? chunk_size : total_len - start;
        if (len <= 0) break;

        int is_speech = mm_vad_is_speech(audio + start, len, threshold);
        if (is_speech && !in_speech) {
            (*speech_starts)[*num_segments] = start;
            in_speech = 1;
        }
        if (!is_speech && in_speech) {
            (*speech_ends)[*num_segments] = start;
            (*num_segments)++;
            in_speech = 0;
        }
    }
    if (in_speech) {
        (*speech_ends)[*num_segments] = total_len;
        (*num_segments)++;
    }
}

void mm_whisper_attn_init(mm_whisper_attn_t* attn, int dim, int num_heads) {
    attn->dim = dim;
    attn->num_heads = num_heads;
    attn->head_dim = dim / num_heads;

    attn->qkv_weight = (float*)calloc((size_t)dim * dim * 3, sizeof(float));
    attn->qkv_bias = (float*)calloc((size_t)dim * 3, sizeof(float));
    attn->proj_weight = (float*)calloc((size_t)dim * dim, sizeof(float));
    attn->proj_bias = (float*)calloc((size_t)dim, sizeof(float));

    float scale = sqrtf(2.0f / (float)dim);
    for (int i = 0; i < dim * dim * 3; i++) {
        attn->qkv_weight[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
    }
    for (int i = 0; i < dim * dim; i++) {
        attn->proj_weight[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
    }
}

void mm_whisper_attn_free(mm_whisper_attn_t* attn) {
    free(attn->qkv_weight);
    free(attn->qkv_bias);
    free(attn->proj_weight);
    free(attn->proj_bias);
}

void mm_whisper_attn_forward(const mm_whisper_attn_t* attn, const float* x,
                             int seq_len, float* out) {
    int dim = attn->dim;
    int nh = attn->num_heads;
    int hd = attn->head_dim;
    int total_dim3 = dim * 3;

    float* qkv = (float*)malloc((size_t)seq_len * total_dim3 * sizeof(float));
    for (int s = 0; s < seq_len; s++) {
        for (int d = 0; d < total_dim3; d++) {
            float sum = attn->qkv_bias[d];
            for (int i = 0; i < dim; i++)
                sum += x[s * dim + i] * attn->qkv_weight[i * total_dim3 + d];
            qkv[s * total_dim3 + d] = sum;
        }
    }

    float* q = qkv, *k = qkv + seq_len * dim, *v = qkv + seq_len * dim * 2;

    for (int s = 0; s < seq_len; s++) {
        float* attn_out = (float*)calloc((size_t)dim, sizeof(float));
        for (int h = 0; h < nh; h++) {
            float* scores = (float*)malloc((size_t)seq_len * sizeof(float));
            float max_sc = -1e9f;
            for (int j = 0; j < seq_len; j++) {
                float dot = 0.0f;
                for (int d = 0; d < hd; d++)
                    dot += q[s * dim + h * hd + d] * k[j * dim + h * hd + d];
                scores[j] = dot / sqrtf((float)hd);
                if (scores[j] > max_sc) max_sc = scores[j];
            }
            float sum = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                scores[j] = expf(scores[j] - max_sc);
                sum += scores[j];
            }
            for (int j = 0; j < seq_len; j++) {
                scores[j] /= sum;
                for (int d = 0; d < hd; d++)
                    attn_out[h * hd + d] += scores[j] * v[j * dim + h * hd + d];
            }
            free(scores);
        }
        for (int d = 0; d < dim && d < 512; d++) {
            float sum = attn->proj_bias[d];
            for (int i = 0; i < dim; i++)
                sum += attn_out[i] * attn->proj_weight[i * dim + d];
            out[s * dim + d] = sum;
        }
        free(attn_out);
    }
    free(qkv);
}

void mm_whisper_cross_attn_init(mm_whisper_cross_attn_t* attn, int dim,
                                int cross_dim, int num_heads) {
    attn->dim = dim;
    attn->cross_dim = cross_dim;
    attn->num_heads = num_heads;
    attn->head_dim = dim / num_heads;

    attn->q_weight = (float*)calloc((size_t)dim * dim, sizeof(float));
    attn->q_bias = (float*)calloc((size_t)dim, sizeof(float));
    attn->k_weight = (float*)calloc((size_t)cross_dim * dim, sizeof(float));
    attn->k_bias = (float*)calloc((size_t)dim, sizeof(float));
    attn->v_weight = (float*)calloc((size_t)cross_dim * dim, sizeof(float));
    attn->v_bias = (float*)calloc((size_t)dim, sizeof(float));
    attn->proj_weight = (float*)calloc((size_t)dim * dim, sizeof(float));
    attn->proj_bias = (float*)calloc((size_t)dim, sizeof(float));

    float scale = sqrtf(2.0f / (float)dim);
    for (int i = 0; i < dim * dim; i++)
        attn->q_weight[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
    for (int i = 0; i < cross_dim * dim; i++)
        attn->k_weight[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
    for (int i = 0; i < cross_dim * dim; i++)
        attn->v_weight[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
    for (int i = 0; i < dim * dim; i++)
        attn->proj_weight[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
}

void mm_whisper_cross_attn_free(mm_whisper_cross_attn_t* attn) {
    free(attn->q_weight); free(attn->q_bias);
    free(attn->k_weight); free(attn->k_bias);
    free(attn->v_weight); free(attn->v_bias);
    free(attn->proj_weight); free(attn->proj_bias);
}

void mm_whisper_cross_attn_forward(const mm_whisper_cross_attn_t* attn,
                                   const float* x, const float* context,
                                   int seq_len, int ctx_len, float* out) {
    int dim = attn->dim;
    int nh = attn->num_heads;
    int hd = attn->head_dim;

    float* q = (float*)malloc((size_t)seq_len * dim * sizeof(float));
    float* k = (float*)malloc((size_t)ctx_len * dim * sizeof(float));
    float* v = (float*)malloc((size_t)ctx_len * dim * sizeof(float));

    for (int s = 0; s < seq_len; s++) {
        for (int d = 0; d < dim; d++) {
            float sq = attn->q_bias[d];
            for (int i = 0; i < dim; i++) sq += x[s * dim + i] * attn->q_weight[i * dim + d];
            q[s * dim + d] = sq;
        }
    }
    for (int s = 0; s < ctx_len; s++) {
        for (int d = 0; d < dim; d++) {
            float sk = attn->k_bias[d], sv = attn->v_bias[d];
            for (int i = 0; i < attn->cross_dim; i++) {
                sk += context[s * attn->cross_dim + i] * attn->k_weight[i * dim + d];
                sv += context[s * attn->cross_dim + i] * attn->v_weight[i * dim + d];
            }
            k[s * dim + d] = sk; v[s * dim + d] = sv;
        }
    }

    for (int s = 0; s < seq_len; s++) {
        float* attn_out = (float*)calloc((size_t)dim, sizeof(float));
        for (int h = 0; h < nh; h++) {
            float* scores = (float*)malloc((size_t)ctx_len * sizeof(float));
            float max_sc = -1e9f;
            for (int j = 0; j < ctx_len; j++) {
                float dot = 0.0f;
                for (int d = 0; d < hd; d++)
                    dot += q[s * dim + h * hd + d] * k[j * dim + h * hd + d];
                scores[j] = dot / sqrtf((float)hd);
                if (scores[j] > max_sc) max_sc = scores[j];
            }
            float sum = 0.0f;
            for (int j = 0; j < ctx_len; j++) {
                scores[j] = expf(scores[j] - max_sc);
                sum += scores[j];
            }
            for (int j = 0; j < ctx_len; j++) {
                scores[j] /= sum;
                for (int d = 0; d < hd; d++)
                    attn_out[h * hd + d] += scores[j] * v[j * dim + h * hd + d];
            }
            free(scores);
        }
        for (int d = 0; d < dim; d++) {
            float sum = attn->proj_bias[d];
            for (int i = 0; i < dim; i++)
                sum += attn_out[i] * attn->proj_weight[i * dim + d];
            out[s * dim + d] = sum;
        }
        free(attn_out);
    }
    free(q); free(k); free(v);
}

void mm_whisper_ffn_init(mm_whisper_ffn_t* ffn, int dim, int ffn_dim) {
    ffn->dim = dim;
    ffn->ffn_dim = ffn_dim;
    ffn->fc1_weight = (float*)calloc((size_t)dim * ffn_dim, sizeof(float));
    ffn->fc1_bias = (float*)calloc((size_t)ffn_dim, sizeof(float));
    ffn->fc2_weight = (float*)calloc((size_t)ffn_dim * dim, sizeof(float));
    ffn->fc2_bias = (float*)calloc((size_t)dim, sizeof(float));
    float scale = sqrtf(2.0f / (float)dim);
    for (int i = 0; i < dim * ffn_dim; i++)
        ffn->fc1_weight[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
    for (int i = 0; i < ffn_dim * dim; i++)
        ffn->fc2_weight[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale;
}

void mm_whisper_ffn_free(mm_whisper_ffn_t* ffn) {
    free(ffn->fc1_weight); free(ffn->fc1_bias);
    free(ffn->fc2_weight); free(ffn->fc2_bias);
}

void mm_whisper_ffn_forward(const mm_whisper_ffn_t* ffn, const float* x,
                            int seq_len, float* out) {
    int dim = ffn->dim;
    int ffd = ffn->ffn_dim;
    for (int s = 0; s < seq_len; s++) {
        float h[2048] = {0};
        for (int o = 0; o < ffd; o++) {
            float sum = ffn->fc1_bias[o];
            for (int i = 0; i < dim; i++)
                sum += x[s * dim + i] * ffn->fc1_weight[i * ffd + o];
            h[o] = sum;
        }
        mm_gelu_forward(h, ffd, h);
        for (int o = 0; o < dim; o++) {
            float sum = ffn->fc2_bias[o];
            for (int i = 0; i < ffd; i++)
                sum += h[i] * ffn->fc2_weight[i * dim + o];
            out[s * dim + o] = sum;
        }
    }
}

void mm_whisper_encoder_layer_init(mm_whisper_encoder_layer_t* layer, int dim,
                                   int num_heads, int ffn_dim) {
    layer->dim = dim;
    mm_whisper_attn_init(&layer->self_attn, dim, num_heads);
    mm_whisper_ffn_init(&layer->ffn, dim, ffn_dim);
    layer->ln1_weight = (float*)calloc((size_t)dim, sizeof(float));
    layer->ln1_bias = (float*)calloc((size_t)dim, sizeof(float));
    layer->ln2_weight = (float*)calloc((size_t)dim, sizeof(float));
    layer->ln2_bias = (float*)calloc((size_t)dim, sizeof(float));
    for (int i = 0; i < dim; i++) {
        layer->ln1_weight[i] = 1.0f; layer->ln2_weight[i] = 1.0f;
    }
}

void mm_whisper_encoder_layer_free(mm_whisper_encoder_layer_t* layer) {
    mm_whisper_attn_free(&layer->self_attn);
    mm_whisper_ffn_free(&layer->ffn);
    free(layer->ln1_weight); free(layer->ln1_bias);
    free(layer->ln2_weight); free(layer->ln2_bias);
}

void mm_whisper_encoder_layer_forward(const mm_whisper_encoder_layer_t* layer,
                                      const float* x, int seq_len, float* out) {
    int dim = layer->dim;
    int total = seq_len * dim;

    float* ln1 = (float*)malloc((size_t)total * sizeof(float));
    for (int s = 0; s < seq_len; s++)
        mm_layernorm(x + s * dim, layer->ln1_weight, layer->ln1_bias, dim, ln1 + s * dim);

    float* attn = (float*)malloc((size_t)total * sizeof(float));
    mm_whisper_attn_forward(&layer->self_attn, ln1, seq_len, attn);

    float* h = (float*)malloc((size_t)total * sizeof(float));
    for (int i = 0; i < total; i++) h[i] = x[i] + attn[i];

    float* ln2 = (float*)malloc((size_t)total * sizeof(float));
    for (int s = 0; s < seq_len; s++)
        mm_layernorm(h + s * dim, layer->ln2_weight, layer->ln2_bias, dim, ln2 + s * dim);

    float* ffn = (float*)malloc((size_t)total * sizeof(float));
    mm_whisper_ffn_forward(&layer->ffn, ln2, seq_len, ffn);

    for (int i = 0; i < total; i++) out[i] = h[i] + ffn[i];

    free(ln1); free(attn); free(h); free(ln2); free(ffn);
}

void mm_whisper_decoder_layer_init(mm_whisper_decoder_layer_t* layer, int dim,
                                   int num_heads, int ffn_dim) {
    layer->dim = dim;
    mm_whisper_attn_init(&layer->self_attn, dim, num_heads);
    mm_whisper_cross_attn_init(&layer->cross_attn, dim, dim, num_heads);
    mm_whisper_ffn_init(&layer->ffn, dim, ffn_dim);

    layer->ln1_weight = (float*)calloc((size_t)dim, sizeof(float));
    layer->ln1_bias = (float*)calloc((size_t)dim, sizeof(float));
    layer->ln2_weight = (float*)calloc((size_t)dim, sizeof(float));
    layer->ln2_bias = (float*)calloc((size_t)dim, sizeof(float));
    layer->ln3_weight = (float*)calloc((size_t)dim, sizeof(float));
    layer->ln3_bias = (float*)calloc((size_t)dim, sizeof(float));
    for (int i = 0; i < dim; i++) {
        layer->ln1_weight[i] = 1.0f;
        layer->ln2_weight[i] = 1.0f;
        layer->ln3_weight[i] = 1.0f;
    }
}

void mm_whisper_decoder_layer_free(mm_whisper_decoder_layer_t* layer) {
    mm_whisper_attn_free(&layer->self_attn);
    mm_whisper_cross_attn_free(&layer->cross_attn);
    mm_whisper_ffn_free(&layer->ffn);
    free(layer->ln1_weight); free(layer->ln1_bias);
    free(layer->ln2_weight); free(layer->ln2_bias);
    free(layer->ln3_weight); free(layer->ln3_bias);
}

void mm_whisper_decoder_layer_forward(const mm_whisper_decoder_layer_t* layer,
                                      const float* x, const float* enc_out,
                                      int seq_len, int enc_len, float* out) {
    int dim = layer->dim;
    int total = seq_len * dim;

    float* ln1 = (float*)malloc((size_t)total * sizeof(float));
    float* attn = (float*)malloc((size_t)total * sizeof(float));
    float* h1 = (float*)malloc((size_t)total * sizeof(float));
    float* ln2 = (float*)malloc((size_t)total * sizeof(float));
    float* cross = (float*)malloc((size_t)total * sizeof(float));
    float* h2 = (float*)malloc((size_t)total * sizeof(float));
    float* ln3 = (float*)malloc((size_t)total * sizeof(float));
    float* ffn = (float*)malloc((size_t)total * sizeof(float));

    for (int s = 0; s < seq_len; s++)
        mm_layernorm(x + s * dim, layer->ln1_weight, layer->ln1_bias, dim, ln1 + s * dim);
    mm_whisper_attn_forward(&layer->self_attn, ln1, seq_len, attn);
    for (int i = 0; i < total; i++) h1[i] = x[i] + attn[i];

    for (int s = 0; s < seq_len; s++)
        mm_layernorm(h1 + s * dim, layer->ln2_weight, layer->ln2_bias, dim, ln2 + s * dim);
    mm_whisper_cross_attn_forward(&layer->cross_attn, ln2, enc_out, seq_len, enc_len, cross);
    for (int i = 0; i < total; i++) h2[i] = h1[i] + cross[i];

    for (int s = 0; s < seq_len; s++)
        mm_layernorm(h2 + s * dim, layer->ln3_weight, layer->ln3_bias, dim, ln3 + s * dim);
    mm_whisper_ffn_forward(&layer->ffn, ln3, seq_len, ffn);
    for (int i = 0; i < total; i++) out[i] = h2[i] + ffn[i];

    free(ln1); free(attn); free(h1); free(ln2);
    free(cross); free(h2); free(ln3); free(ffn);
}

void mm_whisper_model_init(mm_whisper_model_t* model, int encoder_dim,
                           int decoder_dim, int encoder_layers,
                           int decoder_layers, int n_mels) {
    model->sample_rate = MM_WHISPER_SAMPLE_RATE;
    model->n_mels = n_mels;
    model->encoder_dim = encoder_dim;
    model->decoder_dim = decoder_dim;

    mm_mel_filterbank_init(&model->mel_filterbank, n_mels, MM_WHISPER_N_FFT, model->sample_rate);

    mm_conv2d_init(&model->encoder.conv1, 1, encoder_dim, 3, 1, 1, 1);
    mm_conv2d_init(&model->encoder.conv2, encoder_dim, encoder_dim, 3, 2, 1, 1);

    model->encoder.pos_embed = (float*)calloc((size_t)MM_WHISPER_MAX_MEL_FRAMES * encoder_dim, sizeof(float));
    for (int p = 0; p < MM_WHISPER_MAX_MEL_FRAMES; p++) {
        for (int d = 0; d < encoder_dim; d++) {
            float angle = (float)p / powf(10000.0f, (float)(d / 2 * 2) / (float)encoder_dim);
            model->encoder.pos_embed[p * encoder_dim + d] = (d % 2 == 0) ? sinf(angle) : cosf(angle);
        }
    }

    model->encoder.layers = (mm_whisper_encoder_layer_t*)malloc((size_t)encoder_layers * sizeof(mm_whisper_encoder_layer_t));
    for (int i = 0; i < encoder_layers; i++)
        mm_whisper_encoder_layer_init(&model->encoder.layers[i], encoder_dim, MM_WHISPER_ENCODER_HEADS, MM_WHISPER_FFN_DIM);

    model->encoder.num_layers = encoder_layers;
    model->encoder.dim = encoder_dim;
    model->encoder.ln_post_weight = (float*)calloc((size_t)encoder_dim, sizeof(float));
    model->encoder.ln_post_bias = (float*)calloc((size_t)encoder_dim, sizeof(float));
    for (int i = 0; i < encoder_dim; i++) model->encoder.ln_post_weight[i] = 1.0f;

    mm_linear_init(&model->decoder.token_embed, MM_WHISPER_VOCAB_SIZE, decoder_dim);

    model->decoder.pos_embed = (float*)calloc((size_t)MM_WHISPER_MAX_DEC_LEN * decoder_dim, sizeof(float));
    for (int p = 0; p < MM_WHISPER_MAX_DEC_LEN; p++) {
        for (int d = 0; d < decoder_dim; d++) {
            float angle = (float)p / powf(10000.0f, (float)(d / 2 * 2) / (float)decoder_dim);
            model->decoder.pos_embed[p * decoder_dim + d] = (d % 2 == 0) ? sinf(angle) : cosf(angle);
        }
    }

    model->decoder.layers = (mm_whisper_decoder_layer_t*)malloc((size_t)decoder_layers * sizeof(mm_whisper_decoder_layer_t));
    for (int i = 0; i < decoder_layers; i++)
        mm_whisper_decoder_layer_init(&model->decoder.layers[i], decoder_dim, MM_WHISPER_DECODER_HEADS, MM_WHISPER_FFN_DIM);

    model->decoder.num_layers = decoder_layers;
    model->decoder.dim = decoder_dim;
    model->decoder.vocab_size = MM_WHISPER_VOCAB_SIZE;
    model->decoder.max_seq_len = MM_WHISPER_MAX_DEC_LEN;

    model->decoder.ln_final_weight = (float*)calloc((size_t)decoder_dim, sizeof(float));
    model->decoder.ln_final_bias = (float*)calloc((size_t)decoder_dim, sizeof(float));
    for (int i = 0; i < decoder_dim; i++) model->decoder.ln_final_weight[i] = 1.0f;

    mm_linear_init(&model->decoder.lm_head, decoder_dim, MM_WHISPER_VOCAB_SIZE);
}

void mm_whisper_model_free(mm_whisper_model_t* model) {
    mm_mel_filterbank_free(&model->mel_filterbank);
    mm_conv2d_free(&model->encoder.conv1);
    mm_conv2d_free(&model->encoder.conv2);
    free(model->encoder.pos_embed);
    for (int i = 0; i < model->encoder.num_layers; i++)
        mm_whisper_encoder_layer_free(&model->encoder.layers[i]);
    free(model->encoder.layers);
    free(model->encoder.ln_post_weight);
    free(model->encoder.ln_post_bias);

    mm_linear_free(&model->decoder.token_embed);
    free(model->decoder.pos_embed);
    for (int i = 0; i < model->decoder.num_layers; i++)
        mm_whisper_decoder_layer_free(&model->decoder.layers[i]);
    free(model->decoder.layers);
    free(model->decoder.ln_final_weight);
    free(model->decoder.ln_final_bias);
    mm_linear_free(&model->decoder.lm_head);
}

void mm_whisper_encoder_forward(const mm_whisper_encoder_t* enc,
                                const mm_mel_spectrogram_t* mel,
                                float* encoder_hidden) {
    int nf = mel->n_frames;
    int nm = mel->n_mels;
    int dim = enc->dim;

    float* padded = (float*)calloc((size_t)nf * 2 * dim, sizeof(float));
    float* c1 = (float*)calloc((size_t)nf * dim, sizeof(float));

    mm_conv2d_forward(&enc->conv1, mel->data, nf, nm, c1);
    mm_gelu_forward(c1, nf * dim, c1);
    mm_conv2d_forward(&enc->conv2, c1, nf, dim, padded);

    int seq_len = nf;
    for (int s = 0; s < seq_len; s++) {
        for (int d = 0; d < dim; d++) {
            padded[s * dim + d] += enc->pos_embed[s * dim + d];
        }
    }

    for (int l = 0; l < enc->num_layers; l++) {
        float* temp = (float*)malloc((size_t)seq_len * dim * sizeof(float));
        mm_whisper_encoder_layer_forward(&enc->layers[l], padded, seq_len, temp);
        memcpy(padded, temp, (size_t)seq_len * dim * sizeof(float));
        free(temp);
    }

    for (int s = 0; s < seq_len; s++) {
        mm_layernorm(padded + s * dim, enc->ln_post_weight, enc->ln_post_bias, dim, encoder_hidden + s * dim);
    }

    free(padded);
    free(c1);
}

void mm_whisper_decoder_forward(const mm_whisper_decoder_t* dec,
                                const float* encoder_hidden, int enc_len,
                                const int* prev_tokens, int num_prev,
                                float* logits) {
    int dim = dec->dim;
    int seq_len = num_prev;

    float* h = (float*)calloc((size_t)seq_len * dim, sizeof(float));
    for (int s = 0; s < seq_len; s++) {
        for (int d = 0; d < dim; d++) {
            h[s * dim + d] = dec->token_embed.weight[prev_tokens[s] * dim + d] + dec->pos_embed[s * dim + d];
        }
    }

    for (int l = 0; l < dec->num_layers; l++) {
        float* temp = (float*)malloc((size_t)seq_len * dim * sizeof(float));
        mm_whisper_decoder_layer_forward(&dec->layers[l], h, encoder_hidden, seq_len, enc_len, temp);
        memcpy(h, temp, (size_t)seq_len * dim * sizeof(float));
        free(temp);
    }

    float last_norm[512] = {0};
    mm_layernorm(h + (seq_len - 1) * dim, dec->ln_final_weight, dec->ln_final_bias, dim, last_norm);

    for (int o = 0; o < dec->vocab_size; o++) {
        float sum = dec->lm_head.bias[o];
        for (int i = 0; i < dim; i++) {
            sum += last_norm[i] * dec->lm_head.weight[i * dec->vocab_size + o];
        }
        logits[o] = sum;
    }

    free(h);
}

void mm_whisper_transcribe(const mm_whisper_model_t* model,
                           const float* audio, int audio_len,
                           const char* language,
                           mm_whisper_task_t task,
                           mm_whisper_result_t* result) {
    mm_mel_spectrogram_t mel;
    mm_audio_mel_spectrogram(audio, audio_len, &model->mel_filterbank, &mel);

    int enc_len = mel.n_frames;
    float* enc_hidden = (float*)malloc((size_t)enc_len * model->encoder_dim * sizeof(float));
    mm_whisper_encoder_forward(&model->encoder, &mel, enc_hidden);

    int prev_tokens[448];
    int num_prev = 0;
    prev_tokens[num_prev++] = MM_WHISPER_TOKEN_SOT;
    prev_tokens[num_prev++] = MM_WHISPER_TOKEN_LANG_BASE + mm_whisper_lang_id(language);
    prev_tokens[num_prev++] = (task == MM_WHISPER_TASK_TRANSLATE) ? MM_WHISPER_TOKEN_TRANSLATE : MM_WHISPER_TOKEN_TRANSCRIBE;
    prev_tokens[num_prev++] = MM_WHISPER_TOKEN_NO_TIM;

    result->num_segments = 0;
    char current_text[256] = "";
    int text_offset = 0;

    for (int step = 0; step < MM_WHISPER_MAX_DEC_LEN && num_prev < MM_WHISPER_MAX_DEC_LEN - 1; step++) {
        float logits[51865];
        mm_whisper_decoder_forward(&model->decoder, enc_hidden, enc_len,
                                   prev_tokens, num_prev, logits);

        float max_logit = logits[0];
        int next_tok = 0;
        for (int i = 1; i < MM_WHISPER_VOCAB_SIZE; i++) {
            if (logits[i] > max_logit) { max_logit = logits[i]; next_tok = i; }
        }

        if (next_tok == MM_WHISPER_TOKEN_EOT) break;

        prev_tokens[num_prev++] = next_tok;

        if (next_tok >= MM_WHISPER_TOKEN_TIME_BASE && next_tok < MM_WHISPER_TOKEN_TIME_BASE + 1500) {
            if (result->num_segments < result->max_segments) {
                mm_whisper_segment_t* seg = &result->segments[result->num_segments++];
                seg->start_ms = (next_tok - MM_WHISPER_TOKEN_TIME_BASE) * 20;
                seg->end_ms = seg->start_ms + 500;
                seg->prob = 0.9f;
                snprintf(seg->text, sizeof(seg->text), "%s", current_text);
                current_text[0] = '\0';
                text_offset = 0;
            }
        } else if (next_tok > 0 && next_tok < MM_WHISPER_TOKEN_SOT) {
            if (text_offset < 250) {
                current_text[text_offset++] = (char)((next_tok % 26) + 'a');
                current_text[text_offset] = '\0';
            }
        }
    }

    if (text_offset > 0 && result->num_segments < result->max_segments) {
        mm_whisper_segment_t* seg = &result->segments[result->num_segments++];
        seg->start_ms = 0;
        seg->end_ms = audio_len * 1000 / model->sample_rate;
        seg->prob = 0.8f;
        snprintf(seg->text, sizeof(seg->text), "%s", current_text);
    }

    if (language) snprintf(result->detected_lang, sizeof(result->detected_lang), "%s", language);
    result->lang_prob = 0.95f;

    mm_mel_spectrogram_free(&mel);
    free(enc_hidden);
}

void mm_whisper_translate(const mm_whisper_model_t* model,
                          const float* audio, int audio_len,
                          mm_whisper_result_t* result) {
    mm_whisper_transcribe(model, audio, audio_len, "en", MM_WHISPER_TASK_TRANSLATE, result);
}

void mm_whisper_detect_language(const mm_whisper_model_t* model,
                                const float* audio, int audio_len,
                                char* lang, float* prob) {
    mm_mel_spectrogram_t mel;
    mm_audio_mel_spectrogram(audio, audio_len, &model->mel_filterbank, &mel);

    int enc_len = mel.n_frames;
    float* enc_hidden = (float*)malloc((size_t)enc_len * model->encoder_dim * sizeof(float));
    mm_whisper_encoder_forward(&model->encoder, &mel, enc_hidden);

    const char* langs[] = {"en", "zh", "de", "fr", "ja", "ko", "es", "ru", "pt", "it"};
    float best_prob = 0.0f;
    int best_idx = 0;

    for (int l = 0; l < 10; l++) {
        float sim = 0.0f;
        for (int d = 0; d < model->encoder_dim && d < enc_len; d++) {
            sim += enc_hidden[d] * (float)(l + 1);
        }
        sim = sim / (1.0f + fabsf(sim));
        if (sim > best_prob) { best_prob = sim; best_idx = l; }
    }

    snprintf(lang, 8, "%s", langs[best_idx]);
    *prob = best_prob;

    mm_mel_spectrogram_free(&mel);
    free(enc_hidden);
}

void mm_whisper_result_init(mm_whisper_result_t* result, int max_segments) {
    result->max_segments = max_segments;
    result->num_segments = 0;
    result->segments = (mm_whisper_segment_t*)calloc((size_t)max_segments, sizeof(mm_whisper_segment_t));
    result->detected_lang[0] = '\0';
    result->lang_prob = 0.0f;
}

void mm_whisper_result_free(mm_whisper_result_t* result) {
    free(result->segments);
}

void mm_whisper_encode_token(int timestamp_ms, int* token) {
    *token = MM_WHISPER_TOKEN_TIME_BASE + (timestamp_ms / 20);
}

int mm_whisper_decode_time_token(int token) {
    if (token < MM_WHISPER_TOKEN_TIME_BASE) return -1;
    return (token - MM_WHISPER_TOKEN_TIME_BASE) * 20;
}

const char* mm_whisper_lang_name(int lang_id) {
    static const char* names[] = {"en", "zh", "de", "fr", "ja", "ko", "es", "ru", "pt", "it"};
    if (lang_id >= 0 && lang_id < 10) return names[lang_id];
    return "en";
}

int mm_whisper_lang_id(const char* lang) {
    const char* names[] = {"en", "zh", "de", "fr", "ja", "ko", "es", "ru", "pt", "it"};
    for (int i = 0; i < 10; i++) {
        if (strcmp(lang, names[i]) == 0) return i;
    }
    return 0;
}
