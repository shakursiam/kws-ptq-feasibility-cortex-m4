/* Minimal stub for the P2 energy firmware (validation app removed).
 * The energy benchmark in main.c USER CODE 2 calls the ds_cnn_s API directly. */
#ifndef __APP_AI_H
#define __APP_AI_H
#ifdef __cplusplus
extern "C" {
#endif
#include "ai_platform.h"
void MX_X_CUBE_AI_Init(void);
void MX_X_CUBE_AI_Process(void);
#ifdef __cplusplus
}
#endif
#endif /* __APP_AI_H */
