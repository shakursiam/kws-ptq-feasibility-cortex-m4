#!/bin/bash
# P3: build + flash one generated model into a copy of the energy firmware, record 3 INA238 runs,
# read the firmware burst timer. Usage: p3_deploy_measure.sh <variant> <generated_dir> <results_dir>
set -e
V=$1; GEN=$2; RES=$3
SCR="<SCRATCH>"
G=/c/ST/STM32CubeIDE_2.2.0/STM32CubeIDE/plugins/com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.14.3.rel1.win32_1.0.100.202602081740/tools/bin
M=/c/ST/STM32CubeIDE_2.2.0/STM32CubeIDE/plugins/com.st.stm32cube.ide.mcu.externaltools.make.win32_2.2.200.202604021615/tools/bin
export PATH="$G:$M:$PATH"
B="$SCR/p3build_$V"
[ -d "$B" ] || cp -r "$SCR/buildtest/kws_ptq_bench" "$B"
cp <HOME>/kws_ptq_bench/Core/Src/main.c "$B/Core/Src/main.c"
cp "$GEN"/ds_cnn_s*.c "$GEN"/ds_cnn_s*.h "$B/X-CUBE-AI/App/"
cd "$B/Debug"; make -j8 all 2>&1 | grep -E " error|Finished building target"
arm-none-eabi-size kws_ptq_bench.elf | tee "$RES/size_$V.txt"
cp kws_ptq_bench.elf "$RES/fw_$V.elf"
ADDR=$(arm-none-eabi-nm kws_ptq_bench.elf | awk '$3=="g_burst_done"{print $1}')
python "$(cygpath -w /g/My\ Drive/Research_2026/Conference_August/TinyML/Revised_Paper/scripts/p3_capture_runs.py)" "$(cygpath -w kws_ptq_bench.elf)" "$RES" "$V" "0x$ADDR"
