/* P4: on-device time of the MFCC front end (CMSIS-DSP arm_mfcc_f32), same clock setup as the
 * inference firmware. Computes the full 49-frame feature map of a 1 s, 16 kHz clip,
 * for 40 and for 10 coefficients, and times it with the DWT cycle counter. */
#include <string.h>
#include "stm32f4xx_hal.h"
#include "arm_math.h"
#include "mfcc_tables.h"

#define REPS 20
typedef struct {
  uint32_t done, reps;
  uint32_t cyc40, cyc10;      /* cycles for 49 frames, mean over REPS */
  uint32_t cyc_frame40;       /* cycles for one frame (40 coefficients) */
  float    checksum;
} Result;
volatile Result g_mfcc;

static float audio[16000];
static float frame[MFCC_FFT], tmp[MFCC_FFT + 2], out[MFCC_MEL];
static arm_mfcc_instance_f32 m40, m10;

void SystemClock_Config(void);
void Error_Handler(void) { __disable_irq(); while (1) {} }

static uint32_t run49(arm_mfcc_instance_f32 *m, int n) {
  uint32_t t0 = DWT->CYCCNT;
  for (int f = 0; f < MFCC_NFRAMES; f++) {
    memcpy(frame, &audio[f * MFCC_STEP], MFCC_FRAME * sizeof(float));
    memset(&frame[MFCC_FRAME], 0, (MFCC_FFT - MFCC_FRAME) * sizeof(float));
    arm_mfcc_f32(m, frame, out, tmp);
    g_mfcc.checksum += out[n - 1];
  }
  return DWT->CYCCNT - t0;
}

int main(void) {
  HAL_Init();
  SystemClock_Config();
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0; DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

  uint32_t x = 12345;                                   /* deterministic pseudo-noise "speech" */
  for (int i = 0; i < 16000; i++) { x = x * 1664525u + 1013904223u; audio[i] = ((int32_t)x >> 8) * (1.0f / 8388608.0f); }

  if (arm_mfcc_init_f32(&m40, MFCC_FFT, MFCC_MEL, 40, dct40, mel_pos, mel_len, mel_coef, win_coef) != ARM_MATH_SUCCESS) Error_Handler();
  if (arm_mfcc_init_f32(&m10, MFCC_FFT, MFCC_MEL, 10, dct10, mel_pos, mel_len, mel_coef, win_coef) != ARM_MATH_SUCCESS) Error_Handler();

  uint64_t s40 = 0, s10 = 0;
  run49(&m40, 40);                                      /* warm caches */
  for (int r = 0; r < REPS; r++) { s40 += run49(&m40, 40); s10 += run49(&m10, 10); }
  uint32_t t0 = DWT->CYCCNT;
  memcpy(frame, audio, MFCC_FRAME * sizeof(float)); memset(&frame[MFCC_FRAME], 0, (MFCC_FFT - MFCC_FRAME) * sizeof(float));
  arm_mfcc_f32(&m40, frame, out, tmp);
  g_mfcc.cyc_frame40 = DWT->CYCCNT - t0;
  g_mfcc.cyc40 = (uint32_t)(s40 / REPS); g_mfcc.cyc10 = (uint32_t)(s10 / REPS);
  g_mfcc.reps = REPS;
  g_mfcc.done = 0xC0FFEE;
  while (1) {}
}

void SystemClock_Config(void) {
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
