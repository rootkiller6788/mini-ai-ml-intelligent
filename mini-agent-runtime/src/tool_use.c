#include "tool_use.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <math.h>

tool_registry_t* tool_registry_create(void) {
    return tool_registry_create_capacity(TOOL_MAX_REGISTRY);
}

tool_registry_t* tool_registry_create_capacity(int max_tools) {
    tool_registry_t *reg = (tool_registry_t*)calloc(1, sizeof(tool_registry_t));
    if (!reg) return NULL;
    reg->max_tools = max_tools > TOOL_MAX_REGISTRY ? TOOL_MAX_REGISTRY : max_tools;
    reg->count = 0;
    return reg;
}

void tool_registry_destroy(tool_registry_t *reg) {
    if (reg) free(reg);
}

bool tool_registry_add(tool_registry_t *reg, const tool_def_t *def, tool_func_t func, void *user_data) {
    if (!reg || !def || reg->count >= reg->max_tools) return false;
    if (tool_registry_find(reg, def->name)) return false;
    tool_entry_t *entry = &reg->entries[reg->count];
    memcpy(&entry->definition, def, sizeof(tool_def_t));
    entry->func = func;
    entry->user_data = user_data;
    entry->enabled = true;
    reg->count++;
    return true;
}

static char* builtin_calculator(const char *args_json, void *user_data);
static char* builtin_web_search(const char *args_json, void *user_data);
static char* builtin_file_reader(const char *args_json, void *user_data);
static char* builtin_code_executor(const char *args_json, void *user_data);

static const tool_def_t builtin_defs[] = {
    {"calculator", "Evaluate mathematical expressions. Supports +, -, *, /, ^, sqrt, sin, cos, log.",
     TOOL_TYPE_CALCULATOR, 1, {{"expression", TOOL_PARAM_STRING, true, "The math expression to evaluate", ""}}, "", NULL},
    {"web_search", "Search the web for information on a given query.",
     TOOL_TYPE_WEB_SEARCH, 1, {{"query", TOOL_PARAM_STRING, true, "The search query string", ""}}, "", NULL},
    {"file_reader", "Read the contents of a file from the filesystem.",
     TOOL_TYPE_FILE_READER, 2, {
         {"path", TOOL_PARAM_STRING, true, "File path to read", ""},
         {"encoding", TOOL_PARAM_STRING, false, "File encoding (utf8, ascii)", "utf8"}
     }, "", NULL},
    {"code_executor", "Execute code in a sandboxed environment and return output.",
     TOOL_TYPE_CODE_EXECUTOR, 2, {
         {"code", TOOL_PARAM_STRING, true, "Code to execute", ""},
         {"language", TOOL_PARAM_STRING, false, "Programming language", "python"}
     }, "", NULL}
};

bool tool_registry_add_builtin(tool_registry_t *reg, tool_type_t type) {
    if (!reg || type >= TOOL_TYPE_CUSTOM || type < 0) return false;
    tool_func_t f = NULL;
    switch (type) {
        case TOOL_TYPE_CALCULATOR:   f = builtin_calculator; break;
        case TOOL_TYPE_WEB_SEARCH:   f = builtin_web_search; break;
        case TOOL_TYPE_FILE_READER:  f = builtin_file_reader; break;
        case TOOL_TYPE_CODE_EXECUTOR: f = builtin_code_executor; break;
        default: return false;
    }
    return tool_registry_add(reg, &builtin_defs[type], f, NULL);
}

bool tool_registry_remove(tool_registry_t *reg, const char *name) {
    if (!reg || !name) return false;
    for (int i = 0; i < reg->count; i++) {
        if (strcmp(reg->entries[i].definition.name, name) == 0) {
            for (int j = i; j < reg->count - 1; j++) {
                reg->entries[j] = reg->entries[j + 1];
            }
            reg->count--;
            return true;
        }
    }
    return false;
}

tool_entry_t* tool_registry_find(tool_registry_t *reg, const char *name) {
    if (!reg || !name) return NULL;
    for (int i = 0; i < reg->count; i++) {
        if (strcmp(reg->entries[i].definition.name, name) == 0) {
            return &reg->entries[i];
        }
    }
    return NULL;
}

void tool_registry_clear(tool_registry_t *reg) {
    if (reg) reg->count = 0;
}

int tool_registry_count(const tool_registry_t *reg) {
    return reg ? reg->count : 0;
}

const char* tool_get_schemas_json(const tool_registry_t *reg, char *buf, size_t buf_size) {
    if (!reg || !buf) return NULL;
    int offset = snprintf(buf, buf_size, "[\n");
    for (int i = 0; i < reg->count; i++) {
        char schema[TOOL_MAX_SCHEMA_LEN];
        tool_json_schema_generate(&reg->entries[i].definition, schema, sizeof(schema));
        offset += snprintf(buf + offset, buf_size - offset, "  %s%s\n",
                          schema, (i < reg->count - 1) ? "," : "");
    }
    snprintf(buf + offset, buf_size - offset, "]");
    return buf;
}

const char* tool_get_names_list(const tool_registry_t *reg, char *buf, size_t buf_size) {
    if (!reg || !buf) return NULL;
    int offset = 0;
    for (int i = 0; i < reg->count; i++) {
        offset += snprintf(buf + offset, buf_size - offset, "%s%s",
                          reg->entries[i].definition.name,
                          (i < reg->count - 1) ? ", " : "");
    }
    return buf;
}

tool_call_t tool_call_execute(tool_registry_t *reg, const char *name, const char *args_json) {
    tool_call_t call;
    memset(&call, 0, sizeof(call));
    if (!reg || !name) {
        call.success = false;
        strncpy(call.error_message, "Invalid arguments", TOOL_MAX_DESC - 1);
        return call;
    }
    tool_entry_t *entry = tool_registry_find(reg, name);
    if (!entry || !entry->enabled) {
        call.success = false;
        snprintf(call.error_message, TOOL_MAX_DESC, "Tool '%s' not found or disabled", name);
        return call;
    }
    char error_buf[TOOL_MAX_DESC];
    if (!tool_validate_args(&entry->definition, args_json, error_buf, sizeof(error_buf))) {
        call.success = false;
        strncpy(call.error_message, error_buf, TOOL_MAX_DESC - 1);
        return call;
    }
    if (!entry->func) {
        call.success = false;
        strncpy(call.error_message, "Tool has no function implementation", TOOL_MAX_DESC - 1);
        return call;
    }
    char *result = entry->func(args_json, entry->user_data);
    if (result) {
        call.success = true;
        strncpy(call.tool_name, name, TOOL_MAX_NAME - 1);
        strncpy(call.arguments, args_json ? args_json : "", TOOL_MAX_SCHEMA_LEN - 1);
        strncpy(call.result, result, TOOL_MAX_RESULT_LEN - 1);
        free(result);
    } else {
        call.success = false;
        strncpy(call.error_message, "Tool execution returned NULL", TOOL_MAX_DESC - 1);
    }
    return call;
}

tool_batch_t tool_call_execute_parallel(tool_registry_t *reg, const tool_call_t *calls, int count) {
    tool_batch_t batch;
    memset(&batch, 0, sizeof(batch));
    if (!reg || !calls || count <= 0) return batch;
    int n = count > TOOL_MAX_PARALLEL_CALLS ? TOOL_MAX_PARALLEL_CALLS : count;
    batch.count = n;
    for (int i = 0; i < n; i++) {
        batch.calls[i] = tool_call_execute(reg, calls[i].tool_name, calls[i].arguments);
    }
    return batch;
}

tool_call_t tool_call_parse_json(const char *json_str) {
    tool_call_t call;
    memset(&call, 0, sizeof(call));
    if (!json_str) return call;
    const char *fn = strstr(json_str, "\"function\"");
    if (fn) {
        fn = strchr(fn, ':');
        if (fn) {
            fn = strchr(fn, '"');
            if (fn) {
                fn++;
                const char *end = strchr(fn, '"');
                if (end) {
                    size_t len = (size_t)(end - fn);
                    if (len >= TOOL_MAX_NAME) len = TOOL_MAX_NAME - 1;
                    strncpy(call.tool_name, fn, len);
                }
            }
        }
    }
    const char *args = strstr(json_str, "\"arguments\"");
    if (args) {
        args = strchr(args, ':');
        if (args) {
            while (*args == ':' || *args == ' ') args++;
            strncpy(call.arguments, args, TOOL_MAX_SCHEMA_LEN - 1);
        }
    }
    return call;
}

tool_call_t tool_call_parse_text(const char *text_str) {
    tool_call_t call;
    memset(&call, 0, sizeof(call));
    if (!text_str) return call;
    const char *action = strstr(text_str, "Action:");
    if (action) {
        action += 7;
        while (*action == ' ') action++;
        const char *end = strchr(action, '\n');
        if (!end) end = action + strlen(action);
        size_t len = (size_t)(end - action);
        if (len >= TOOL_MAX_NAME) len = TOOL_MAX_NAME - 1;
        strncpy(call.tool_name, action, len);
    }
    const char *input = strstr(text_str, "Action Input:");
    if (input) {
        input += 13;
        while (*input == ' ') input++;
        const char *end = strchr(input, '\n');
        if (!end) end = input + strlen(input);
        size_t len = (size_t)(end - input);
        if (len >= TOOL_MAX_SCHEMA_LEN) len = TOOL_MAX_SCHEMA_LEN - 1;
        strncpy(call.arguments, input, len);
    }
    return call;
}

bool tool_validate_args(const tool_def_t *def, const char *args_json, char *error_buf, size_t buf_size) {
    if (!def) return false;
    for (int i = 0; i < def->param_count; i++) {
        if (def->params[i].required) {
            if (!args_json || !strstr(args_json, def->params[i].name)) {
                if (error_buf) {
                    snprintf(error_buf, buf_size, "Missing required parameter: %s", def->params[i].name);
                }
                return false;
            }
        }
    }
    return true;
}

bool tool_call_succeeded(const tool_call_t *call) {
    return call && call->success;
}

const char* tool_call_result(const tool_call_t *call) {
    return call ? call->result : NULL;
}

const char* tool_call_error(const tool_call_t *call) {
    return call ? call->error_message : NULL;
}

const char* tool_format_for_llm(const void *reg_ptr, char *buf, size_t buf_size) {
    const tool_registry_t *reg = (const tool_registry_t*)reg_ptr;
    if (!reg || !buf) return NULL;
    int offset = 0;
    for (int i = 0; i < reg->count; i++) {
        const tool_def_t *def = &reg->entries[i].definition;
        offset += snprintf(buf + offset, buf_size - offset,
                          "- %s: %s\n  Parameters:\n", def->name, def->description);
        for (int j = 0; j < def->param_count; j++) {
            offset += snprintf(buf + offset, buf_size - offset,
                              "    %s (%s%s): %s\n",
                              def->params[j].name,
                              def->params[j].required ? "required" : "optional",
                              def->params[j].type == TOOL_PARAM_STRING ? ", string" :
                              def->params[j].type == TOOL_PARAM_INTEGER ? ", int" :
                              def->params[j].type == TOOL_PARAM_FLOAT ? ", float" :
                              def->params[j].type == TOOL_PARAM_BOOLEAN ? ", bool" : "",
                              def->params[j].description);
        }
    }
    return buf;
}

const char* tool_json_schema_generate(const tool_def_t *def, char *buf, size_t buf_size) {
    if (!def || !buf) return NULL;
    int off = snprintf(buf, buf_size,
        "{\"name\": \"%s\", \"description\": \"%s\", \"parameters\": {"
        "\"type\": \"object\", \"properties\": {", def->name, def->description);
    for (int i = 0; i < def->param_count; i++) {
        const char *type_str = "string";
        switch (def->params[i].type) {
            case TOOL_PARAM_STRING:  type_str = "string"; break;
            case TOOL_PARAM_INTEGER: type_str = "integer"; break;
            case TOOL_PARAM_FLOAT:   type_str = "number"; break;
            case TOOL_PARAM_BOOLEAN: type_str = "boolean"; break;
            case TOOL_PARAM_OBJECT:  type_str = "object"; break;
            case TOOL_PARAM_ARRAY:   type_str = "array"; break;
        }
        off += snprintf(buf + off, buf_size - off,
            "\"%s\": {\"type\": \"%s\", \"description\": \"%s\"}%s",
            def->params[i].name, type_str, def->params[i].description,
            (i < def->param_count - 1) ? ", " : "");
    }
    off += snprintf(buf + off, buf_size - off, "}, \"required\": [");
    bool first = true;
    for (int i = 0; i < def->param_count; i++) {
        if (def->params[i].required) {
            off += snprintf(buf + off, buf_size - off, "%s\"%s\"", first ? "" : ", ", def->params[i].name);
            first = false;
        }
    }
    snprintf(buf + off, buf_size - off, "]}}");
    return buf;
}

static double simple_eval(const char **expr);

static double parse_number(const char **s) {
    while (**s == ' ') (*s)++;
    double val = 0.0;
    double frac = 0.0;
    double div = 1.0;
    int sign = 1;
    if (**s == '-') { sign = -1; (*s)++; }
    if (**s == '+') { (*s)++; }
    while (isdigit((unsigned char)**s)) {
        val = val * 10.0 + (**s - '0');
        (*s)++;
    }
    if (**s == '.') {
        (*s)++;
        while (isdigit((unsigned char)**s)) {
            frac = frac * 10.0 + (**s - '0');
            div *= 10.0;
            (*s)++;
        }
    }
    val = sign * (val + frac / div);
    while (**s == ' ') (*s)++;
    return val;
}

static double parse_factor(const char **s) {
    while (**s == ' ') (*s)++;
    if (**s == '(') {
        (*s)++;
        double v = simple_eval(s);
        if (**s == ')') (*s)++;
        return v;
    }
    return parse_number(s);
}

static double parse_term(const char **s) {
    double left = parse_factor(s);
    while (**s) {
        while (**s == ' ') (*s)++;
        if (**s == '^') { (*s)++; double r = parse_factor(s); left = pow(left, r); continue; }
        if (**s == '*' || **s == '/') {
            char op = **s; (*s)++;
            double right = parse_factor(s);
            left = (op == '*') ? left * right : (right != 0 ? left / right : 0);
            continue;
        }
        break;
    }
    return left;
}

static double simple_eval(const char **expr) {
    double left = parse_term(expr);
    while (**expr) {
        while (**expr == ' ') (*expr)++;
        if (**expr == '+' || **expr == '-') {
            char op = **expr; (*expr)++;
            double right = parse_term(expr);
            left = (op == '+') ? left + right : left - right;
            continue;
        }
        break;
    }
    return left;
}

static char* builtin_calculator(const char *args_json, void *user_data) {
    (void)user_data;
    char *result = (char*)malloc(TOOL_MAX_RESULT_LEN);
    if (!result) return NULL;
    const char *expr = args_json;
    const char *eq = strstr(args_json, "expression");
    if (eq) {
        eq = strchr(eq, ':');
        if (eq) {
            eq = strchr(eq, '"');
            if (eq) eq++;
            const char *end = strchr(eq, '"');
            if (end) {
                static char expr_buf[4096];
                size_t len = (size_t)(end - eq);
                if (len >= sizeof(expr_buf)) len = sizeof(expr_buf) - 1;
                strncpy(expr_buf, eq, len);
                expr_buf[len] = '\0';
                expr = expr_buf;
            }
        }
    }
    double val = simple_eval(&expr);
    snprintf(result, TOOL_MAX_RESULT_LEN, "Result: %g", val);
    return result;
}

static char* builtin_web_search(const char *args_json, void *user_data) {
    (void)user_data;
    char *result = (char*)malloc(TOOL_MAX_RESULT_LEN);
    if (!result) return NULL;
    snprintf(result, TOOL_MAX_RESULT_LEN,
             "Search results for query. [Simulated search results would appear here. Args: %s]", args_json);
    return result;
}

static char* builtin_file_reader(const char *args_json, void *user_data) {
    (void)user_data;
    char *result = (char*)malloc(TOOL_MAX_RESULT_LEN);
    if (!result) return NULL;
    snprintf(result, TOOL_MAX_RESULT_LEN,
             "File contents. [Simulated file read. Args: %s]", args_json);
    return result;
}

static char* builtin_code_executor(const char *args_json, void *user_data) {
    (void)user_data;
    char *result = (char*)malloc(TOOL_MAX_RESULT_LEN);
    if (!result) return NULL;
    snprintf(result, TOOL_MAX_RESULT_LEN,
             "Code execution output. [Simulated code execution. Args: %s]", args_json);
    return result;
}
