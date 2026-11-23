/* Minimal stub for the P2 energy firmware (validation app removed).
 * The energy benchmark in main.c USER CODE 2 calls the ds_cnn_s API directly. */
#include <stddef.h>
#include "app_x-cube-ai.h"

void MX_X_CUBE_AI_Init(void) {}
void MX_X_CUBE_AI_Process(void) {}

/* The project still links with -Wl,--wrap=malloc,--wrap=free (left over from the
 * validation app, which used them for heap monitoring). Forward to the real ones. */
void *__real_malloc(size_t size);
void  __real_free(void *ptr);
void *__wrap_malloc(size_t size) { return __real_malloc(size); }
void  __wrap_free(void *ptr)     { __real_free(ptr); }
