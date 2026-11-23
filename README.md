# Quantization as a Deployability Gate for Keyword Spotting on a Cortex-M4 MCU

Code, models, firmware, raw measurements, and analysis scripts for a controlled, physically
measured comparison of the four TensorFlow Lite post-training quantization (PTQ) options
(FP32, float16, dynamic-range INT8, full-integer INT8) for DS-CNN-S keyword spotting on an
**STM32 Nucleo-F446RE** (Cortex-M4, 180 MHz, 128 KiB SRAM).

**Main finding.** On a memory-constrained MCU, the choice of PTQ scheme acts as a *binary
deployability gate* rather than a smooth accuracy–efficiency trade-off. With the 40-MFCC input,
FP32 and dynamic-range INT8 exceed the SRAM budget and float16 cannot be processed by the
toolchain; only full-integer INT8 runs. With a 10-MFCC input the gate opens, yet full-integer
INT8 remains 3.9× faster and 3.6× more energy-efficient. TensorFlow Lite Micro confirms that
float16 and dynamic-range INT8 cannot be executed and that X-CUBE-AI is the faster runtime.

---

## Key results

### PTQ variants on the Nucleo-F446RE (X-CUBE-AI 10.2.1 / ST Edge AI Core v2.2.0)

| Input | Variant | Acc. (%) | Flash (KiB) | RAM (KiB) | Latency (ms) | Total energy (mJ) |
|---|---|---|---|---|---|---|
| 49×40 | FP32 | 92.18 | 110.45 | 256.70 ✗ | – | – |
| 49×40 | float16 | 92.23 | analysis aborts (`INTERNAL ERROR: 'FLOAT16'`) | | – | – |
| 49×40 | DR-INT8 | 92.57 | 107.21 | 194.46 ✗ | – | – |
| 49×40 | **Full-INT8** | **92.25** | **62.26** | **69.53** | **158.72** | **22.45 ± 0.03** |
| 49×10 | FP32 | 93.70 | 110.44 | 69.20 | 175.87 | 23.35 ± 0.02 |
| 49×10 | float16 | 93.70 | analysis aborts | | – | – |
| 49×10 | DR-INT8 | 93.58 | 107.20 | 55.71 | 174.75 | 24.37 ± 0.01 |
| 49×10 | **Full-INT8** | **93.27** | **62.27** | **23.98** | **45.03** | **6.40 ± 0.004** |

✗ exceeds the 128 KiB SRAM. Accuracy: desktop TFLite interpreter on the 4,888-clip test split
(on-device Validate-on-target for 49×40 full-INT8 on all 4,888 clips: 92.27 %, top-1 identical
to the reference on every clip). Energy: total energy per inference, `P_active × t_lat`, mean ± sd
of three runs.

### Runtime comparison (same `.tflite` files, same board, clock, and power protocol)

| Model | Runtime | Latency (ms) | Cycles/MAC | E_tot (mJ) | Image (KiB) | RAM (KiB) |
|---|---|---|---|---|---|---|
| 49×40 INT8 | X-CUBE-AI | 156.6 | 2.65 | 22.45 | 81.1 | 69.5 |
| 49×40 INT8 | TFLM + CMSIS-NN | 270.8 | 4.57 | 38.15 | 151.5 | 69.2 |
| 49×40 INT8 | TFLM reference | 1674.4 | 28.3 | 213.5 | 107.1 | 69.1 |
| 49×10 INT8 | X-CUBE-AI | 45.0 | 3.04 | 6.40 | 81.1 | 24.0 |
| 49×10 INT8 | TFLM + CMSIS-NN | 69.2 | 4.67 | 9.99 | 151.5 | 22.3 |
| 49×10 INT8 | TFLM reference | 402.2 | 27.2 | 52.41 | 107.1 | 22.2 |
| 49×10 FP32 | X-CUBE-AI | 175.9 | 11.6 | 23.35 | 129.3 | 69.2 |
| 49×10 FP32 | TFLM (reference float) | 399.6 | 26.3 | 54.51 | 201.3 | 64.5 |

float16 under TFLM fails in `DEQUANTIZE` (only int8/int16/uint8 inputs supported);
dynamic-range INT8 is rejected (`Hybrid models are not supported on TFLite Micro`).

### Energy and real-time feasibility (49×40 full-INT8)

| Quantity | Value |
|---|---|
| Busy-wait idle (`HAL_Delay`, 180 MHz) | 124.26 ± 0.37 mW |
| WFI Sleep idle | 42.99 ± 0.17 mW |
| Active (inference) | 143.31 ± 0.21 mW |
| Marginal energy `(P_active − P_idle) · t_lat` | 2.98 ± 0.03 mJ |
| Total energy `P_active · t_lat` | 22.45 ± 0.03 mJ |
| Average power with WFI between inferences (T_hop = 200 ms / 500 ms / 1 s) | 121.6 / 74.4 / 58.7 mW |
| On-device MFCC (CMSIS-DSP `arm_mfcc_f32`), one 20 ms frame / 49 frames | 1.04 ms / 50.9 ms |
| Minimum hop `T_hop ≥ t_lat / 0.948` (49×40 / 49×10, X-CUBE-AI) | 165 ms / 48 ms |

Accuracy statistics (paired bootstrap, 10,000 resamples; exact McNemar; Bonferroni over 6 pairs):
full-INT8 vs FP32 (49×40): +0.06 pp, 95 % CI [−0.41, 0.53], p = 0.87. Official Speech Commands v2
test set (4,890 clips): 91.47 / 91.49 / 91.80 / 91.31 % for FP32 / float16 / DR-INT8 / full-INT8.

---

## Repository structure

```
models/
  mfcc40/                 4 .tflite variants + FP32 Keras checkpoint (49x40 input, main model)
  mfcc10/                 4 .tflite variants + FP32 Keras checkpoint (49x10 input)
training/
  Speech_Command.ipynb    data split, balancing, MFCC front end, training, PTQ export (49x40)
  train_10mfcc.py         same recipe with NUM_MFCC = 10, export and evaluation
evaluation/
  p1_eval.py              rebuilds the exact test split, evaluates all variants, bootstrap/McNemar
  bootstrap_ci.py         paired bootstrap CIs and exact McNemar test
  count_macs.py           MAC counter for .tflite files
  evaluate_and_save_cm.py original evaluation / confusion-matrix script
firmware/
  xcubeai/                STM32CubeIDE main.c for the energy firmware (busy-wait | WFI | 200 inferences)
  tflm/                   TFLM firmware, make-free TFLM build script, measurement driver, MFCC benchmark
power_measurement/
  esp32s3_ina238_logger/  ESP32-S3 Arduino sketch reading the INA238 (52 Hz, averaged samples)
  p2_capture.py           serial -> CSV capture
  p2_analyze.py           phase segmentation, P_idle / P_sleep / P_active, energy, average power
  p3_*                    deploy-and-measure helpers for generated X-CUBE-AI models
results/                  raw data and reports (see docs/EXPERIMENTS.md)
docs/EXPERIMENTS.md       what each experiment does and where its outputs are
```

## Hardware setup

- **Target:** STM32 Nucleo-F446RE, HSI → PLL at 180 MHz, flash latency 5, ART accelerator on.
- **Power monitor:** INA238 breakout, 15 mΩ shunt, placed on the Nucleo **IDD jumper (JP6)**:
  IN+ to the supply-side pin (3.3 V with the jumper removed), IN− (and VBUS) to the MCU-side pin.
  This measures the STM32 supply only, excluding the ST-LINK.
- **Logger:** ESP32-S3 (SDA = GPIO 8, SCL = GPIO 9, INA238 at 0x40), common ground with the Nucleo.
  Continuous shunt+bus conversions, VSHCT = 1052 µs, VBUSCT = 150 µs, AVG = 16 → ~52 Hz.
- **Firmware phases per capture:** 5 s boot + 10 s busy-wait idle | LED flash | 15 s WFI Sleep |
  LED flash | 200 back-to-back inferences (SysTick-timed) | end blink. Windows are trimmed by
  0.5 s; the active window is anchored on the Sleep plateau and the firmware-timed burst length.

## Software versions

| Component | Version |
|---|---|
| TensorFlow (training, PTQ, desktop evaluation) | 2.21, Python 3.13 |
| STM32CubeIDE / GNU Arm toolchain | 2.2.0 / GCC 14.3.rel1 |
| X-CUBE-AI / ST Edge AI Core | 10.2.1 / v2.2.0 (optimization: balanced) |
| TensorFlow Lite Micro | `tflite-micro` commit `38a8f0f` (Nov 2026), release flags (`-Os` core, `-O2` kernels, `-DNDEBUG`) |
| CMSIS-DSP (MFCC benchmark) | from STM32Cube_FW_F4 V1.28.3 |
| ESP32 Arduino core | 3.0.7 |

## Reproducing the experiments

The dataset is not included. Download
[Speech Commands v0.02](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz) and,
for the official test set,
[speech_commands_test_set_v0.02](http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz).

Several scripts contain absolute paths to local tool installations; these appear as `<HOME>`,
`<DRIVE>`, or `<SCRATCH>` placeholders and must be set to your own paths (constants at the top of
each script).

1. **Training and PTQ export (49×40):** run `training/Speech_Command.ipynb`.
2. **Accuracy statistics:**
   `python evaluation/p1_eval.py --project <dataset_and_models_dir> --official <official_test_dir> --out results/P1`
3. **49×10 model:**
   `python training/train_10mfcc.py --project <dataset_dir> --official <official_test_dir> --out results/P3`
4. **Static analysis:** `stedgeai analyze --target stm32f4 -m <model.tflite>` (and `generate` for
   flash/RAM totals, with the Arm GCC toolchain on `PATH`).
5. **Energy (X-CUBE-AI):** flash the CubeIDE project with `firmware/xcubeai/main.c`, flash
   `power_measurement/esp32s3_ina238_logger` to the ESP32-S3, then
   `python power_measurement/p2_capture.py --port COMx --seconds 75 --out run1.csv` (press RESET) and
   `python power_measurement/p2_analyze.py "run*.csv" --tlat-ms <t> --anchored --plot`.
6. **TFLM runtime comparison:**
   `python firmware/tflm/tflm_build.py --kernels cmsis_nn|ref --model <m.tflite> --name <n> --out results/P4`
   then `python firmware/tflm/p4_measure.py results/P4/<n>.elf results/P4 <n>`.
7. **MFCC timing:** `python firmware/tflm/mfcc_build.py results/P4`, flash `mfcc_bench.elf`, read `g_mfcc` over ST-LINK.

## Notes and limitations

- One network (DS-CNN-S) at two input sizes, one board, one training run per input size
  (test-set uncertainty quantified with a paired bootstrap on two test sets; seed variation not measured).
- Busy-wait idle power depends on the firmware build (124–126 mW for X-CUBE-AI builds,
  160–163 mW for TFLM builds); WFI Sleep power is 43–45 mW in every build. Compare runtimes by
  **total** energy.
- Boards were reset through the ST-LINK hardware reset; debug-in-sleep is disabled in firmware.

## Citation

If you use this repository, please cite the accompanying paper (citation to be added on publication).

## License

Code is released under the MIT License (see `LICENSE`). The Speech Commands dataset is distributed
by its authors under CC BY 4.0 and is not included here.
