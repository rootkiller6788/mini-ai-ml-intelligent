#ifndef TOOL_USE_H
#define TOOL_USE_H

#include <stddef.h>
#include <stdbool.h>

#define TOOL_MAX_NAME           64
#define TOOL_MAX_DESC           512
#define TOOL_MAX_SCHEMA_LEN     4096
#define TOOL_MAX_REGISTRY       128
#define TOOL_MAX_ARGS           16
#define TOOL_MAX_ARG_NAME       32
#define TOOL_MAX_ARG_VALUE      4096
#define TOOL_MAX_PARALLEL_CALLS 8
#define TOOL_MAX_RESULT_LEN     16384

typedef enum {
    TOOL_TYPE_CALCULATOR,
    TOOL_TYPE_WEB_SEARCH,
    TOOL_TYPE_FILE_READER,
    TOOL_TYPE_CODE_EXECUTOR,
    TOOL_TYPE_CUSTOM,
    TOOL_TYPE_COUNT
} tool_type_t;

typedef enum {
    TOOL_PARAM_STRING,
    TOOL_PARAM_INTEGER,
    TOOL_PARAM_FLOAT,
    TOOL_PARAM_BOOLEAN,
    TOOL_PARAM_OBJECT,
    TOOL_PARAM_ARRAY
} tool_param_type_t;

typedef struct {
    char name[TOOL_MAX_ARG_NAME];
    tool_param_type_t type;
    bool required;
    char description[TOOL_MAX_DESC];
    char default_value[TOOL_MAX_ARG_VALUE];
} tool_param_def_t;

typedef struct {
    char name[TOOL_MAX_NAME];
    char description[TOOL_MAX_DESC];
    tool_type_t type;
    int param_count;
    tool_param_def_t params[TOOL_MAX_ARGS];
    char json_schema[TOOL_MAX_SCHEMA_LEN];
    void *user_data;
} tool_def_t;

typedef struct {
    char tool_name[TOOL_MAX_NAME];
    char arguments[TOOL_MAX_SCHEMA_LEN];
    char result[TOOL_MAX_RESULT_LEN];
    bool success;
    char error_message[TOOL_MAX_DESC];
} tool_call_t;

typedef char* (*tool_func_t)(const char *args_json, void *user_data);

typedef struct {
    tool_def_t definition;
    tool_func_t func;
    void *user_data;
    bool enabled;
} tool_entry_t;

typedef struct {
    tool_entry_t entries[TOOL_MAX_REGISTRY];
    int count;
    int max_tools;
} tool_registry_t;

typedef struct {
    tool_call_t calls[TOOL_MAX_PARALLEL_CALLS];
    int count;
} tool_batch_t;

tool_registry_t* tool_registry_create(void);
tool_registry_t* tool_registry_create_capacity(int max_tools);
void tool_registry_destroy(tool_registry_t *reg);

bool tool_registry_add(tool_registry_t *reg, const tool_def_t *def, tool_func_t func, void *user_data);
bool tool_registry_add_builtin(tool_registry_t *reg, tool_type_t type);
bool tool_registry_remove(tool_registry_t *reg, const char *name);
tool_entry_t* tool_registry_find(tool_registry_t *reg, const char *name);
void tool_registry_clear(tool_registry_t *reg);
int tool_registry_count(const tool_registry_t *reg);

const char* tool_get_schemas_json(const tool_registry_t *reg, char *buf, size_t buf_size);
const char* tool_get_names_list(const tool_registry_t *reg, char *buf, size_t buf_size);

tool_call_t tool_call_execute(tool_registry_t *reg, const char *name, const char *args_json);
tool_batch_t tool_call_execute_parallel(tool_registry_t *reg, const tool_call_t *calls, int count);
tool_call_t tool_call_parse_json(const char *json_str);
tool_call_t tool_call_parse_text(const char *text_str);

bool tool_validate_args(const tool_def_t *def, const char *args_json, char *error_buf, size_t buf_size);
bool tool_call_succeeded(const tool_call_t *call);
const char* tool_call_result(const tool_call_t *call);
const char* tool_call_error(const tool_call_t *call);

const char* tool_format_for_llm(const void *reg, char *buf, size_t buf_size);
const char* tool_json_schema_generate(const tool_def_t *def, char *buf, size_t buf_size);

#endif
