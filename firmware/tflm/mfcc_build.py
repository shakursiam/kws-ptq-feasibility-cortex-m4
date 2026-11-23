"""Build mfcc_bench.c (HAL + CMSIS-DSP from STM32Cube_FW_F4 V1.28.3) for the Nucleo-F446RE."""
import concurrent.futures as cf, os, subprocess, sys
from pathlib import Path
from tflm_build import GCC, PROJ, HAL_INC, HAL_DEF, ARCH

HERE = Path(__file__).resolve().parent
DSP = Path("<HOME>/STM32Cube/Repository/STM32Cube_FW_F4_V1.28.3/Drivers/CMSIS/DSP")
out = Path(sys.argv[1]); obj = out / "obj_mfcc"; obj.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(HERE / "gen_mfcc_tables.py"), str(obj / "mfcc_tables.h")], check=True)
INC = HAL_INC + [f"-I{DSP}/Include", f"-I{DSP}/PrivateInclude", f"-I{obj}"]
DEF = HAL_DEF + ["-DARM_MATH_CM4", "-D__FPU_PRESENT=1U"]
groups = ["BasicMathFunctions", "CommonTables", "ComplexMathFunctions", "FastMathFunctions", "MatrixFunctions",
          "StatisticsFunctions", "SupportFunctions", "TransformFunctions"]
srcs = [DSP / "Source" / g / f"{g}.c" for g in groups]
srcs += sorted((PROJ / "Drivers/STM32F4xx_HAL_Driver/Src").glob("*.c"))
srcs += [PROJ / "Core/Src" / f for f in ("system_stm32f4xx.c", "stm32f4xx_it.c", "stm32f4xx_hal_msp.c", "syscalls.c", "sysmem.c")]
srcs += [PROJ / "Core/Startup/startup_stm32f446retx.s", HERE / "mfcc_bench.c"]


def cc(s):
    o = obj / (s.stem + ".o")
    x = ["-x", "assembler-with-cpp"] if s.suffix == ".s" else ["-std=gnu11"]
    r = subprocess.run([str(GCC / "arm-none-eabi-gcc"), *ARCH, *x, "-O2", "-ffunction-sections", "-fdata-sections",
                        *DEF, *INC, "-c", str(s), "-o", str(o)], capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"FAILED {s}\n{r.stderr[-2000:]}")
    return o


with cf.ThreadPoolExecutor(os.cpu_count()) as ex:
    objs = list(ex.map(cc, srcs))
elf = out / "mfcc_bench.elf"
r = subprocess.run([str(GCC / "arm-none-eabi-gcc"), *ARCH, f"-T{PROJ}/STM32F446RETX_FLASH.ld", "--specs=nano.specs",
                    "--specs=nosys.specs", "-Wl,--gc-sections", *map(str, objs), "-lm", "-o", str(elf)],
                   capture_output=True, text=True)
if r.returncode:
    sys.exit(r.stderr[-3000:])
print(subprocess.run([str(GCC / "arm-none-eabi-size"), str(elf)], capture_output=True, text=True).stdout)
nm = subprocess.run([str(GCC / "arm-none-eabi-nm"), str(elf)], capture_output=True, text=True).stdout
print("g_mfcc @ 0x" + [l.split()[0] for l in nm.splitlines() if l.endswith(" g_mfcc")][0])
