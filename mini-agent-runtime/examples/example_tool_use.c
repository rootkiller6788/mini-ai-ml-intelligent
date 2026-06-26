#include "tool_use.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char* custom_weather_tool(const char *args_json, void *user_data) {
    (void)user_data;
    char *result = (char*)malloc(TOOL_MAX_RESULT_LEN);
    if (!result) return NULL;
    const char *city = "unknown";
    const char *city_tag = strstr(args_json, "city");
    if (city_tag) {
        city_tag = strchr(city_tag, '"');
        if (city_tag) {
            city_tag++;
            const char *end = strchr(city_tag, '"');
            if (end) {
                static char city_buf[256];
                size_t len = (size_t)(end - city_tag);
                if (len >= sizeof(city_buf)) len = sizeof(city_buf) - 1;
                strncpy(city_buf, city_tag, len);
                city_buf[len] = '\0';
                city = city_buf;
            }
        }
    }
    snprintf(result, TOOL_MAX_RESULT_LEN,
             "{\"city\": \"%s\", \"temperature\": 22.5, \"humidity\": 65, "
             "\"condition\": \"Partly Cloudy\", \"wind_speed\": 12.3}", city);
    return result;
}

int main(void) {
    printf("=== Tool Use Example ===\n\n");

    tool_registry_t *reg = tool_registry_create_capacity(16);
    if (!reg) { printf("Failed to create registry\n"); return 1; }

    printf("Adding built-in tools...\n");
    tool_registry_add_builtin(reg, TOOL_TYPE_CALCULATOR);
    tool_registry_add_builtin(reg, TOOL_TYPE_WEB_SEARCH);
    tool_registry_add_builtin(reg, TOOL_TYPE_FILE_READER);
    tool_registry_add_builtin(reg, TOOL_TYPE_CODE_EXECUTOR);
    printf("Built-in tools added. Registry count: %d\n\n", tool_registry_count(reg));

    tool_def_t weather_def;
    memset(&weather_def, 0, sizeof(weather_def));
    strcpy(weather_def.name, "get_weather");
    strcpy(weather_def.description, "Get current weather for a specified city");
    weather_def.type = TOOL_TYPE_CUSTOM;
    weather_def.param_count = 2;
    strcpy(weather_def.params[0].name, "city");
    weather_def.params[0].type = TOOL_PARAM_STRING;
    weather_def.params[0].required = true;
    strcpy(weather_def.params[0].description, "City name to get weather for");
    strcpy(weather_def.params[1].name, "units");
    weather_def.params[1].type = TOOL_PARAM_STRING;
    weather_def.params[1].required = false;
    strcpy(weather_def.params[1].description, "Temperature units (celsius/fahrenheit)");

    if (tool_registry_add(reg, &weather_def, custom_weather_tool, NULL)) {
        printf("Custom 'get_weather' tool added successfully.\n");
    }
    printf("Total tools: %d\n\n", tool_registry_count(reg));

    printf("--- Tool Descriptions for LLM ---\n");
    char desc_buf[8192];
    tool_format_for_llm(reg, desc_buf, sizeof(desc_buf));
    printf("%s\n", desc_buf);

    printf("--- JSON Schema Output ---\n");
    char schema_buf[8192];
    tool_get_schemas_json(reg, schema_buf, sizeof(schema_buf));
    printf("%s\n\n", schema_buf);

    printf("--- Tool Execution Tests ---\n\n");

    printf("Test 1: Calculator - 3.14 * 2 + 10\n");
    tool_call_t call1 = tool_call_execute(reg, "calculator",
        "{\"expression\": \"3.14 * 2 + 10\"}");
    if (tool_call_succeeded(&call1)) {
        printf("  Result: %s\n", tool_call_result(&call1));
    } else {
        printf("  Error: %s\n", tool_call_error(&call1));
    }

    printf("\nTest 2: Web Search - \"C programming\"\n");
    tool_call_t call2 = tool_call_execute(reg, "web_search",
        "{\"query\": \"C programming\"}");
    if (tool_call_succeeded(&call2)) {
        printf("  Result: %s\n", tool_call_result(&call2));
    } else {
        printf("  Error: %s\n", tool_call_error(&call2));
    }

    printf("\nTest 3: File Reader\n");
    tool_call_t call3 = tool_call_execute(reg, "file_reader",
        "{\"path\": \"/tmp/test.txt\", \"encoding\": \"utf8\"}");
    if (tool_call_succeeded(&call3)) {
        printf("  Result: %s\n", tool_call_result(&call3));
    } else {
        printf("  Error: %s\n", tool_call_error(&call3));
    }

    printf("\nTest 4: Code Executor\n");
    tool_call_t call4 = tool_call_execute(reg, "code_executor",
        "{\"code\": \"print('hello')\", \"language\": \"python\"}");
    if (tool_call_succeeded(&call4)) {
        printf("  Result: %s\n", tool_call_result(&call4));
    } else {
        printf("  Error: %s\n", tool_call_error(&call4));
    }

    printf("\nTest 5: Custom Weather Tool\n");
    tool_call_t call5 = tool_call_execute(reg, "get_weather",
        "{\"city\": \"Beijing\", \"units\": \"celsius\"}");
    if (tool_call_succeeded(&call5)) {
        printf("  Result: %s\n", tool_call_result(&call5));
    } else {
        printf("  Error: %s\n", tool_call_error(&call5));
    }

    printf("\nTest 6: Missing required argument\n");
    tool_call_t call6 = tool_call_execute(reg, "get_weather", "{\"units\": \"celsius\"}");
    if (!tool_call_succeeded(&call6)) {
        printf("  Expected error: %s\n", tool_call_error(&call6));
    }

    printf("\nTest 7: Non-existent tool\n");
    tool_call_t call7 = tool_call_execute(reg, "nonexistent_tool", "{}");
    if (!tool_call_succeeded(&call7)) {
        printf("  Expected error: %s\n", tool_call_error(&call7));
    }

    printf("\n--- JSON Function Call Parsing ---\n");
    const char *json_call = "{\"function\": \"calculator\", \"arguments\": {\"expression\": \"5*5\"}}";
    tool_call_t parsed_json = tool_call_parse_json(json_call);
    printf("  Parsed tool: %s\n", parsed_json.tool_name);
    printf("  Parsed args: %s\n", parsed_json.arguments);

    printf("\n--- Text Function Call Parsing ---\n");
    const char *text_call = "Action: search\nAction Input: latest AI news";
    tool_call_t parsed_text = tool_call_parse_text(text_call);
    printf("  Parsed tool: %s\n", parsed_text.tool_name);
    printf("  Parsed args: %s\n", parsed_text.arguments);

    printf("\n--- Parallel Tool Execution ---\n");
    tool_call_t parallel_calls[3];
    memset(&parallel_calls, 0, sizeof(parallel_calls));
    strcpy(parallel_calls[0].tool_name, "calculator");
    strcpy(parallel_calls[0].arguments, "{\"expression\": \"1+1\"}");
    strcpy(parallel_calls[1].tool_name, "calculator");
    strcpy(parallel_calls[1].arguments, "{\"expression\": \"2+2\"}");
    strcpy(parallel_calls[2].tool_name, "calculator");
    strcpy(parallel_calls[2].arguments, "{\"expression\": \"4+4\"}");

    tool_batch_t batch = tool_call_execute_parallel(reg, parallel_calls, 3);
    printf("  Executed %d parallel calls:\n", batch.count);
    for (int i = 0; i < batch.count; i++) {
        printf("    Call %d: %s -> %s\n", i,
               batch.calls[i].tool_name, batch.calls[i].result);
    }

    printf("\n--- Removing a tool ---\n");
    printf("  Before removal: %d tools\n", tool_registry_count(reg));
    tool_registry_remove(reg, "file_reader");
    printf("  After removing file_reader: %d tools\n", tool_registry_count(reg));

    tool_entry_t *found = tool_registry_find(reg, "calculator");
    if (found) {
        printf("\nFound tool: %s - %s\n", found->definition.name, found->definition.description);
    }

    char names_buf[1024];
    tool_get_names_list(reg, names_buf, sizeof(names_buf));
    printf("\nRemaining tools: %s\n", names_buf);

    tool_registry_destroy(reg);
    printf("\nExample completed successfully.\n");
    return 0;
}
