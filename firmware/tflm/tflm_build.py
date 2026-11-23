#!/usr/bin/env python3
"""
tflm_build.py -- build the P4 TFLM firmware for the Nucleo-F446RE without `make`.

Mirrors what tensorflow/lite/micro/tools/make/Makefile does for
  TARGET=cortex_m_generic TARGET_ARCH=cortex-m4+fp BUILD_TYPE=release_with_logs
  [OPTIMIZED_KERNEL_DIR=cmsis_nn]
(-Os core / -O2 kernels, -DNDEBUG, KERNELS_OPTIMIZED_FOR_SPEED, same source lists and
kernel specialisation), then links with the Nucleo HAL/startup/linker script of the
X-CUBE-AI energy project so that clocks, flash wait states and caches are identical.

Usage:
  python tflm_build.py --kernels cmsis_nn|ref --model x.tflite --name s40_int8_cmsis --out <dir> [--arena-kb 100]
"""
import argparse, concurrent.futures as cf, hashlib, os, re, subprocess, sys
from pathlib import Path

TFLM = Path("<HOME>/tflm_p4/tflite-micro")
DL = TFLM / "tensorflow/lite/micro/tools/make/downloads"
PROJ = Path("<HOME>/kws_ptq_bench")
GCC = Path("C:/ST/STM32CubeIDE_2.2.0/STM32CubeIDE/plugins/"
           "com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.14.3.rel1.win32_1.0.100.202602081740/tools/bin")
HERE = Path(__file__).resolve().parent
CACHE = Path("<HOME>/tflm_p4/obj")

ARCH = ["-mcpu=cortex-m4", "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard", "-mthumb", "-mlittle-endian"]
COMMON = ["-fno-unwind-tables", "-fno-asynchronous-unwind-tables", "-ffunction-sections", "-fdata-sections",
          "-fmessage-length=0", "-DTFLM_REMOVE_VIRTUAL_DELETE", "-DTF_LITE_DISABLE_X86_NEON",
          "-DFLATBUFFERS_LOCALE_INDEPENDENT=0", "-DFLATBUFFERS_PREFER_PRINTF=1",
          "-DKERNELS_OPTIMIZED_FOR_SPEED", "-DNDEBUG",
          "-DTF_LITE_MCU_DEBUG_LOG", "-Wno-type-limits", "-fomit-frame-pointer", "-DCPU_M4=1",
          "-funsigned-char", '-DCMSIS_DEVICE_ARM_CORTEX_M_XX_HEADER_FILE="ARMCM4.h"', "-DARMCM4",
          "-DARM_NN_ENABLE_F16=0", "-DARM_NN_ENABLE_F32=0"] + ARCH
CXX = ["-std=c++17", "-fno-rtti", "-fno-exceptions", "-fno-threadsafe-statics"]
CC = ["-std=c17"]
TFLM_INC = [f"-I{TFLM}", f"-I{DL}", f"-I{DL}/flatbuffers/include", f"-I{DL}/kissfft",
            f"-I{DL}/cmsis", f"-I{DL}/cmsis_nn", f"-I{DL}/cmsis/CMSIS/Core/Include",
            f"-I{DL}/cmsis_nn/Include", f"-I{DL}/cmsis/Cortex_DFP/Device/ARMCM4/Include"]
HAL_INC = [f"-I{PROJ}/Core/Inc", f"-I{PROJ}/Drivers/STM32F4xx_HAL_Driver/Inc",
           f"-I{PROJ}/Drivers/CMSIS/Device/ST/STM32F4xx/Include", f"-I{PROJ}/Drivers/CMSIS/Include"]
HAL_DEF = ["-DUSE_HAL_DRIVER", "-DSTM32F446xx"]


def make_var(text, name):
    m = re.search(rf"^{name} :=\s*\\\n((?:.*\\\n)*.*)$", text, re.M)
    return [l.strip().rstrip("\\").strip().replace("$(TENSORFLOW_ROOT)", "") for l in m.group(1).splitlines()
            if l.strip().rstrip("\\").strip()]


def glob_wild(entries):
    out = []
    for e in entries:
        w = re.match(r"\$\(wildcard (.*)\)", e)
        out += sorted(str(p.relative_to(TFLM)).replace("\\", "/") for p in TFLM.glob(w.group(1))) if w else [e]
    return out


def specialize(files, d):
    d = TFLM / d
    return [str((d / Path(f).name).relative_to(TFLM)).replace("\\", "/") if (d / Path(f).name).exists() else f
            for f in files]


def library_sources(kernels):
    src = (TFLM / "tensorflow/lite/micro/tools/make/sources.inc").read_text()
    base = glob_wild(make_var(src, "MICROLITE_CC_BASE_SRCS"))
    tests = set(make_var(src, "MICROLITE_TEST_SRCS"))
    base = [f for f in base if f not in tests and not f.endswith("_test.cc") and "benchmark" not in f]
    base = specialize(base, "tensorflow/lite/micro/cortex_m_generic")
    kern = make_var(src, "MICROLITE_CC_KERNEL_SRCS")
    core = [(f, "-Os") for f in base]
    ks = [(f, "-O2") for f in (specialize(kern, "tensorflow/lite/micro/kernels/cmsis_nn") if kernels == "cmsis_nn" else kern)]
    third = []
    if kernels == "cmsis_nn":
        third = [(str(p.relative_to(TFLM)).replace("\\", "/"), "-O2") for p in sorted((DL / "cmsis_nn/Source").rglob("*.c"))
                 if not p.name.endswith(("_f16.c", "_f32.c")) and p.name != "arm_nntables_flt.c"]
    return core + ks + third


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"FAILED: {' '.join(map(str, cmd))[:300]}\n{r.stderr[-3000:]}")
    return r


def compile_one(src, opt, inc, extra, objdir):
    src = Path(src); obj = objdir / (hashlib.md5(str(src).encode()).hexdigest()[:10] + "_" + src.stem + ".o")
    if obj.exists() and obj.stat().st_mtime > src.stat().st_mtime:
        return obj
    if src.suffix == ".cc":
        tool, std = "arm-none-eabi-g++", CXX
    elif src.suffix in (".s", ".S"):
        tool, std = "arm-none-eabi-gcc", ["-x", "assembler-with-cpp"]
    else:
        tool, std = "arm-none-eabi-gcc", CC
    run([str(GCC / tool), *std, *COMMON, opt, *inc, *extra, "-c", str(src), "-o", str(obj)])
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", choices=["cmsis_nn", "ref"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arena-kb", type=int, default=100)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # 1. TFLM library (cached per kernel set)
    libdir = CACHE / a.kernels; libdir.mkdir(parents=True, exist_ok=True)
    srcs = library_sources(a.kernels)
    with cf.ThreadPoolExecutor(os.cpu_count()) as ex:
        kdef = ["-DCMSIS_NN"] if a.kernels == "cmsis_nn" else []
        objs = list(ex.map(lambda s: compile_one(TFLM / s[0], s[1], TFLM_INC, kdef, libdir), srcs))
    print(f"TFLM library ({a.kernels}): {len(objs)} objects", flush=True)

    # 2. firmware: HAL + startup + model + main (per build)
    fw = out / f"obj_{a.name}"; fw.mkdir(exist_ok=True)
    data = Path(a.model).read_bytes()
    (fw / "model_data.h").write_text("#pragma once\nextern const unsigned char g_model[];\nextern const unsigned int g_model_len;\n")
    body = ",".join(str(b) for b in data)
    (fw / "model_data.cc").write_text('#include "model_data.h"\nalignas(16) const unsigned char g_model[] = {'
                                      + body + f"}};\nconst unsigned int g_model_len = {len(data)};\n")
    hal = sorted((PROJ / "Drivers/STM32F4xx_HAL_Driver/Src").glob("*.c"))
    core = [PROJ / "Core/Src" / f for f in ("system_stm32f4xx.c", "stm32f4xx_it.c", "stm32f4xx_hal_msp.c", "syscalls.c", "sysmem.c")]
    fw_srcs = [(s, "-Os", HAL_INC, HAL_DEF) for s in hal + core + [PROJ / "Core/Startup/startup_stm32f446retx.s"]]
    fw_srcs += [(fw / "model_data.cc", "-Os", [f"-I{fw}"], []),
                (HERE / "main_tflm.cc", "-Os", [f"-I{fw}", *HAL_INC, f"-I{TFLM}", f"-I{DL}/flatbuffers/include"],
                 HAL_DEF + [f"-DARENA_KB={a.arena_kb}"])]
    for f in fw.glob("main_tflm*.o"):
        f.unlink()
    with cf.ThreadPoolExecutor(os.cpu_count()) as ex:
        fobjs = list(ex.map(lambda s: compile_one(s[0], s[1], s[2], s[3], fw), fw_srcs))

    elf = out / f"{a.name}.elf"
    run([str(GCC / "arm-none-eabi-g++"), *ARCH, f"-T{PROJ}/STM32F446RETX_FLASH.ld", "--specs=nano.specs", "--specs=nosys.specs",
         "-Wl,--gc-sections", f"-Wl,-Map={out / (a.name + '.map')}", "-static",
         *map(str, fobjs), *map(str, objs), "-Wl,--start-group", "-lc", "-lm", "-lstdc++", "-lsupc++", "-Wl,--end-group",
         "-o", str(elf)])
    sz = run([str(GCC / "arm-none-eabi-size"), str(elf)]).stdout
    print(sz, flush=True)
    (out / f"{a.name}_size.txt").write_text(sz)
    nm = run([str(GCC / "arm-none-eabi-nm"), str(elf)]).stdout
    addr = [l.split()[0] for l in nm.splitlines() if l.endswith(" g_res")][0]
    (out / f"{a.name}_gres_addr.txt").write_text("0x" + addr)
    print(f"g_res @ 0x{addr}")


if __name__ == "__main__":
    main()
