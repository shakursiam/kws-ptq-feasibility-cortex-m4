/*
 * P2 firmware change for kws_ptq_bench/Core/Src/main.c
 * ----------------------------------------------------
 * Replace ONLY these lines of your current USER CODE 2 block:
 *
 *     // Give the board 5 seconds to boot up and stabilize before starting baseline
 *     HAL_Delay(5000);
 *
 *     / * --- 1. IDLE BASELINE PERIOD (~10 seconds, LED off) --- * /
 *     HAL_Delay(10000);
 *
 *     // Single flash marks the idle -> active transition. ...
 *     HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_SET);
 *     HAL_Delay(100);
 *     HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_RESET);
 *
 * with the block below. Everything else (LED init, network init, 200-inference
 * loop, end blink) stays exactly as it is.
 *
 * Resulting trace:  5 s boot | 10 s BUSY idle | flash | 15 s WFI-SLEEP idle | flash | 200 inferences
 *                             (~122 mW)                  (lower)                       (~149 mW)
 */

    // Keep debug logic from holding clocks on during Sleep (measure without a debug session anyway)
    HAL_DBGMCU_DisableDBGSleepMode();

    // Give the board 5 seconds to boot up and stabilize before starting baseline
    HAL_Delay(5000);

    /* --- 1a. BUSY-WAIT IDLE BASELINE (10 s, core running at 180 MHz) --- */
    HAL_Delay(10000);

    // Single flash: busy -> sleep transition
    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_SET);
    HAL_Delay(100);
    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_RESET);

    /* --- 1b. WFI SLEEP IDLE (15 s). Core sleeps; SysTick wakes it every 1 ms,
     *         loop re-checks time and goes straight back to sleep. --- */
    {
        uint32_t t0 = HAL_GetTick();
        while ((HAL_GetTick() - t0) < 15000U)
        {
            HAL_PWR_EnterSLEEPMode(PWR_MAINREGULATOR_ON, PWR_SLEEPENTRY_WFI);
        }
    }

    // Single flash: sleep -> active transition
    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_SET);
    HAL_Delay(100);
    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_RESET);

/* ---- (your existing "2. ACTIVE INFERENCE BURST" code continues unchanged below) ---- */
