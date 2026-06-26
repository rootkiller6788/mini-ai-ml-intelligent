#include "model_serving.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    MS_ModelServer server;
    ms_server_init(&server);

    MS_ModelConfig config;
    ms_config_load(&config, "model_repository/llama2");

    ms_model_load(&server, "llama2-7b", 1);
    ms_model_load(&server, "llama2-7b", 1);

    MS_InferenceRequest* requests[4];
    for (int i = 0; i < 4; i++) {
        requests[i] = ms_request_create(i, 128, 64);
        requests[i]->priority = (i < 2) ? MS_PRIORITY_HIGH : MS_PRIORITY_NORMAL;
        for (size_t j = 0; j < requests[i]->input_len; j++) {
            requests[i]->input_data[j] = (float)(j + i * 100);
        }
        ms_request_submit(&server, requests[i]);
    }

    MS_InferenceRequest* batch[MS_MAX_BATCH_SIZE];
    int batch_size = ms_dynamic_batch(&server, batch, MS_MAX_BATCH_SIZE);
    printf("Dynamic batch assembled: %d requests\n", batch_size);

    ms_batch_infer(&server, batch, batch_size);

    for (int i = 0; i < batch_size; i++) {
        printf("  Request %llu output[0]=%.3f output[63]=%.3f\n",
               (unsigned long long)batch[i]->id,
               batch[i]->output_data[0],
               batch[i]->output_data[batch[i]->output_len - 1]);
    }

    printf("Throughput: %.1f req/s\n", ms_server_throughput(&server));
    printf("Avg latency: %.2f ms\n", ms_server_avg_latency_ms(&server));

    for (int i = 0; i < 4; i++) {
        ms_request_destroy(requests[i]);
    }

    ms_server_destroy(&server);
    printf("Model serving example complete.\n");
    return 0;
}
