// P4 (Reviewer 1, comments 2 and 9): the same DS-CNN-S .tflite run with TensorFlow Lite Micro
// on the Nucleo-F446RE, with the SAME clock setup and SAME measurement phases as the
// X-CUBE-AI energy firmware (5+10 s busy-wait | flash | 15 s WFI sleep | flash | 200 inferences).
//
// Results are written to g_res and read back over ST-LINK (no UART traffic during measurement).
#include <string.h>
#include "stm32f4xx_hal.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/cortex_m_generic/debug_log_callback.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"
#include "model_data.h"

#ifndef ARENA_KB
#define ARENA_KB 100
#endif

struct Result {
  uint32_t done;        // 0xC0FFEE = burst finished, 0xDEAD0000|stage = failed at stage
  uint32_t burst_ms;    // SysTick length of the 200-inference burst
  uint32_t stage;       // last stage reached (see below)
  uint32_t status;      // TfLiteStatus of the failing call
  uint32_t arena_used;  // interpreter.arena_used_bytes()
  uint32_t arena_size;
  uint32_t in_type, out_type;
  int8_t out_i8[12];
  float out_f[12];
  char log[384];        // TFLM error messages
};
extern "C" { volatile Result g_res; }

alignas(16) static uint8_t tensor_arena[ARENA_KB * 1024];

static void log_cb(const char* s) {
  size_t n = strlen((const char*)g_res.log);
  size_t m = strlen(s);
  if (n + m + 1 < sizeof(g_res.log)) memcpy((char*)g_res.log + n, s, m + 1);
}

extern "C" void SystemClock_Config(void);
extern "C" void Error_Handler(void);

static void led(int on) { HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, on ? GPIO_PIN_SET : GPIO_PIN_RESET); }
static void flash(int n, uint32_t ms) { for (int i = 0; i < n; i++) { led(1); HAL_Delay(ms); led(0); HAL_Delay(ms); } }
static void fail(uint32_t stage, uint32_t st) {
  g_res.status = st; g_res.done = 0xDEAD0000u | stage;
  while (1) flash(1, 50);   // fast blink = failure
}

int main(void) {
  HAL_Init();
  SystemClock_Config();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  GPIO_InitTypeDef g = {0};
  g.Pin = GPIO_PIN_5; g.Mode = GPIO_MODE_OUTPUT_PP; g.Pull = GPIO_NOPULL; g.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOA, &g);
  led(0);
  HAL_DBGMCU_DisableDBGSleepMode();
  memset((void*)&g_res, 0, sizeof(g_res));
  RegisterDebugLogCallback(log_cb);
  g_res.arena_size = sizeof(tensor_arena);

  // ---- set-up (before the measurement phases, so a failure is visible immediately) ----
  g_res.stage = 1;
  const tflite::Model* model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) fail(1, model->version());
  static tflite::MicroMutableOpResolver<6> resolver;
  resolver.AddConv2D(); resolver.AddDepthwiseConv2D(); resolver.AddMean();
  resolver.AddFullyConnected(); resolver.AddSoftmax(); resolver.AddDequantize();
  g_res.stage = 2;
  static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, sizeof(tensor_arena));
  TfLiteStatus st = interpreter.AllocateTensors();
  if (st != kTfLiteOk) fail(2, st);
  g_res.stage = 3;
  g_res.arena_used = interpreter.arena_used_bytes();
  TfLiteTensor* in = interpreter.input(0);
  TfLiteTensor* out = interpreter.output(0);
  g_res.in_type = in->type; g_res.out_type = out->type;
  memset(in->data.raw, 0, in->bytes);              // same all-zero input bytes as the X-CUBE-AI firmware
  st = interpreter.Invoke();
  if (st != kTfLiteOk) fail(3, st);
  g_res.stage = 4;
  for (int i = 0; i < 12; i++) {
    if (out->type == kTfLiteInt8) g_res.out_i8[i] = out->data.int8[i];
    else if (out->type == kTfLiteFloat32) g_res.out_f[i] = out->data.f[i];
  }

  // ---- measurement phases: identical to the X-CUBE-AI energy firmware ----
  HAL_Delay(5000);
  HAL_Delay(10000);                                // busy-wait idle
  flash(1, 100);
  { uint32_t t0 = HAL_GetTick();
    while ((HAL_GetTick() - t0) < 15000U) HAL_PWR_EnterSLEEPMode(PWR_MAINREGULATOR_ON, PWR_SLEEPENTRY_WFI); }
  led(1); HAL_Delay(100); led(0);                  // single flash: sleep -> active

  uint32_t t0 = HAL_GetTick();
  for (int i = 0; i < 200; i++) interpreter.Invoke();
  g_res.burst_ms = HAL_GetTick() - t0;
  g_res.stage = 5;
  g_res.done = 0xC0FFEE;

  flash(2, 100);
  while (1) { HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5); HAL_Delay(1000); }
}

extern "C" void SystemClock_Config(void) {
  RCC_OscInitTypeDef o = {0};
  RCC_ClkInitTypeDef c = {0};
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);
  o.OscillatorType = RCC_OSCILLATORTYPE_HSI; o.HSIState = RCC_HSI_ON;
  o.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  o.PLL.PLLState = RCC_PLL_ON; o.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  o.PLL.PLLM = 8; o.PLL.PLLN = 180; o.PLL.PLLP = 2; o.PLL.PLLQ = 2; o.PLL.PLLR = 2;
  if (HAL_RCC_OscConfig(&o) != HAL_OK) Error_Handler();
  if (HAL_PWREx_EnableOverDrive() != HAL_OK) Error_Handler();
  c.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
  c.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK; c.AHBCLKDivider = RCC_SYSCLK_DIV1;
  c.APB1CLKDivider = RCC_HCLK_DIV4; c.APB2CLKDivider = RCC_HCLK_DIV2;
  if (HAL_RCC_ClockConfig(&c, FLASH_LATENCY_5) != HAL_OK) Error_Handler();
}

extern "C" void Error_Handler(void) { __disable_irq(); while (1) {} }
