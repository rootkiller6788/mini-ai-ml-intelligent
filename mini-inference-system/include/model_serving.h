#ifndef MODEL_SERVING_H
#define MODEL_SERVING_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define MS_MAX_MODEL_NAME     256
#define MS_MAX_INSTANCES      16
#define MS_MAX_BATCH_SIZE     256
#define MS_MAX_QUEUE_DEPTH    1024
#define MS_MAX_VERSION        100
#define MS_DEFAULT_PORT_GRPC  8001
#define MS_DEFAULT_PORT_HTTP  8000

typedef enum {
    MS_DTYPE_FP32 = 0,
    MS_DTYPE_FP16 = 1,
    MS_DTYPE_INT8 = 2,
    MS_DTYPE_INT4 = 3,
    MS_DTYPE_BF16 = 4,
} MS_DataType;

typedef enum {
    MS_PRIORITY_LOW    = 0,
    MS_PRIORITY_NORMAL = 1,
    MS_PRIORITY_HIGH   = 2,
    MS_PRIORITY_URGENT = 3,
} MS_Priority;

typedef enum {
    MS_SCHED_FIFO     = 0,
    MS_SCHED_PRIORITY = 1,
    MS_SCHED_ROUND_ROBIN = 2,
} MS_SchedulePolicy;

typedef enum {
    MS_STATE_UNLOADED  = 0,
    MS_STATE_LOADING   = 1,
    MS_STATE_READY     = 2,
    MS_STATE_UNLOADING = 3,
} MS_ModelState;

typedef struct {
    uint64_t id;
    float*   input_data;
    size_t   input_len;
    float*   output_data;
    size_t   output_len;
    MS_Priority priority;
    double   arrival_time;
    bool     completed;
    void*    user_data;
} MS_InferenceRequest;

typedef struct {
    char     name[MS_MAX_MODEL_NAME];
    int64_t  version;
    char     platform[32];
    int      max_batch_size;
    MS_DataType input_dtype;
    MS_DataType output_dtype;
    char     backend[64];
    int      instance_count;
    int      gpu_device_ids[MS_MAX_INSTANCES];
    bool     dynamic_batching_enabled;
    int      max_queue_delay_us;
    MS_SchedulePolicy schedule_policy;
    char     model_path[512];
} MS_ModelConfig;

typedef struct {
    MS_ModelState state;
    MS_ModelConfig config;
    void*  model_handle;
    size_t memory_bytes;
    int64_t loaded_version;
    double load_time_ms;
} MS_ModelInstance;

typedef struct {
    MS_InferenceRequest*  requests;
    int      head;
    int      tail;
    int      count;
    int      capacity;
    pthread_mutex_t mutex;
    pthread_cond_t  not_empty;
    pthread_cond_t  not_full;
} MS_RequestQueue;

typedef struct {
    MS_ModelInstance instances[MS_MAX_INSTANCES];
    int      instance_count;
    MS_RequestQueue queue;
    bool     running;
    int      active_batch_count;
    double   total_requests_served;
    double   total_inference_time_ms;
} MS_ModelServer;

bool ms_model_load(MS_ModelServer* server, const char* model_name, int64_t version);
bool ms_model_unload(MS_ModelServer* server, const char* model_name);
bool ms_model_reload(MS_ModelServer* server, const char* model_name);

bool ms_request_submit(MS_ModelServer* server, MS_InferenceRequest* req);
bool ms_request_cancel(MS_ModelServer* server, uint64_t request_id);
MS_InferenceRequest* ms_request_create(uint64_t id, size_t input_len, size_t output_len);
void  ms_request_destroy(MS_InferenceRequest* req);

void  ms_queue_init(MS_RequestQueue* q, int capacity);
void  ms_queue_destroy(MS_RequestQueue* q);
bool  ms_queue_push(MS_RequestQueue* q, MS_InferenceRequest* req);
MS_InferenceRequest* ms_queue_pop(MS_RequestQueue* q);
MS_InferenceRequest* ms_queue_pop_priority(MS_RequestQueue* q);

int   ms_dynamic_batch(MS_ModelServer* server, MS_InferenceRequest** batch, int max_batch);
void  ms_batch_infer(MS_ModelServer* server, MS_InferenceRequest** batch, int batch_size);
void  ms_split_batch_results(MS_InferenceRequest** batch, int batch_size, float* batch_output);

bool  ms_config_load(MS_ModelConfig* config, const char* config_path);
bool  ms_config_save(const MS_ModelConfig* config, const char* config_path);
char* ms_config_to_protobuf_text(const MS_ModelConfig* config);

void  ms_server_init(MS_ModelServer* server);
void  ms_server_run(MS_ModelServer* server);
void  ms_server_stop(MS_ModelServer* server);
void  ms_server_destroy(MS_ModelServer* server);

double ms_server_throughput(const MS_ModelServer* server);
double ms_server_avg_latency_ms(const MS_ModelServer* server);

bool  ms_grpc_serve(MS_ModelServer* server, int port);
bool  ms_http_serve(MS_ModelServer* server, int port);
int   ms_protocol_detect(const uint8_t* data, size_t len);

#endif
