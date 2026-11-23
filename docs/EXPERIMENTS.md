# Experiments and result files

| ID | Experiment | Key outputs (in `results/`) |
|---|---|---|
| P0 | Static analysis, Validate-on-target, original energy firmware, float16 failure | `P0/analyze_*.txt`, `P0/*_c_info.json`, `P0/full_int8_generate_report.txt`, `P0/validate_on_target_full_int8.txt`, `P0/validate_on_target_full_testset_4888.txt`, `P0/analyze_fp16_error.txt` |
| P1 | Exact test split rebuilt; accuracy, paired bootstrap, McNemar; official test set; full-test-set on-device validation | `P1/log_paper_split.txt`, `P1/log_full.txt`, `P1/predictions_*.npz`, `P1/full_50min.txt` |
| P2 | Energy with busy-wait / WFI Sleep / active phases, firmware-timed latency (49×40 full-INT8) | `P2/int8_40mfcc_run{1,2,3}.csv`, `P2/int8_40mfcc_check_timer.csv`, `P2/int8_40mfcc_analysis_anchored.txt`, `P2/power_trace_paper.png` |
| P3 | 49×10 DS-CNN-S: training, PTQ export, accuracy statistics, Analyze/Generate, deployment, energy | `P3/log_train.txt`, `P3/predictions_*.npz`, `P3/analyze/`, `P3/generate/`, `P3/energy/` |
| P4 | TFLM (reference, CMSIS-NN) for all variants; float16/DR failures; output check; MFCC timing | `P4/*_result.json`, `P4/*_run*.csv`, `P4/*_analysis.txt`, `P4/*_size.txt`, `P4/desktop_zero_input_outputs.json`, `P4/mfcc_timing.txt` |

## Measurement protocol

1. Flash firmware; reset the board (ST-LINK hardware reset or power cycle).
2. Firmware: 5 s boot + 10 s busy-wait idle → LED flash → 15 s WFI Sleep → LED flash →
   200 back-to-back inferences (SysTick-timed; result stored in a global variable) → double flash.
3. The ESP32-S3 streams INA238 samples (~52 Hz) to the host; three captures per configuration.
4. `p2_analyze.py --anchored`: the Sleep plateau is located from the trace; busy window = 10 s
   before it, active window = firmware-timed burst after it; 0.5 s trimmed at each edge.
5. Reported: P_idle (busy-wait), P_sleep, P_active, marginal energy, total energy, average power
   for several hop periods, mean ± sd over three runs.

## Mapping to reviewer comments

| Comment | Experiment |
|---|---|
| 1 — gate predictable; add a model/board where it opens | P3 (49×10 model) |
| 2 — MACs, CMSIS-NN/TFLM comparison, real-time feasibility | `evaluation/count_macs.py`, P4, MFCC timing |
| 3 — total energy, idle state | P2 (busy-wait and WFI Sleep phases) |
| 4 — seeds or bootstrap CIs | P1 (paired bootstrap, McNemar, Bonferroni) |
| 7 — on-device accuracy on the full test set | P1 `full_50min.txt` / P0 `validate_on_target_full_testset_4888.txt` |
| 8 — standard split | P1 (official test set) |
| 9 — float16 on TFLM | P4 (`s10_fp16_cmsis_result.json`) |
