"""
p4_measure.py -- flash a TFLM firmware, read its status block (g_res) over ST-LINK, and if the
model set up and ran, record 3 INA238 runs exactly as for X-CUBE-AI, then read the burst timer.

Usage: python p4_measure.py <elf> <results_dir> <name> [--runs 3]
"""
import argparse, json, re, struct, subprocess, time
from pathlib import Path
import serial

P = (r"C:\ST\STM32CubeIDE_2.2.0\STM32CubeIDE\plugins\com.st.stm32cube.ide.mcu.externaltools."
     r"cubeprogrammer.win32_2.2.500.202603051304\tools\bin\STM32_Programmer_CLI.exe")
GRES_BYTES = 8 * 4 + 12 + 12 * 4 + 384
TYPES = {1: "float32", 9: "int8", 10: "float16"}


def st(*args):
    return subprocess.run([P, "-c", "port=SWD", *args], capture_output=True, text=True).stdout


def read_gres(addr):
    out = st("mode=HOTPLUG", "-r32", hex(addr), str(GRES_BYTES))
    words = []
    for l in out.splitlines():
        m = re.match(r"\s*0x[0-9A-Fa-f]{8}\s*:\s*((?:[0-9A-Fa-f]{8}\s*)+)$", l)
        if m:
            words += [int(w, 16) for w in m.group(1).split()]
    raw = struct.pack(f"<{len(words)}I", *words)[:GRES_BYTES]
    if len(raw) < GRES_BYTES:   # SWD read fails while the core is in WFI sleep (debug-in-sleep is off)
        return {"done": 0, "burst_ms": 0, "stage": -1}
    u = struct.unpack_from("<8I", raw, 0)
    r = dict(zip(["done", "burst_ms", "stage", "status", "arena_used", "arena_size", "in_type", "out_type"], u))
    r["out_i8"] = list(struct.unpack_from("<12b", raw, 32))
    r["out_f"] = [round(x, 5) for x in struct.unpack_from("<12f", raw, 44)]
    r["log"] = raw[92:].split(b"\0", 1)[0].decode(errors="replace")
    r["done_hex"] = hex(r["done"])
    r["in_type"] = TYPES.get(r["in_type"], r["in_type"]); r["out_type"] = TYPES.get(r["out_type"], r["out_type"])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("elf"); ap.add_argument("res"); ap.add_argument("name")
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    res = Path(a.res)
    addr = int((res / f"{a.name}_gres_addr.txt").read_text().strip(), 16)
    out = st("mode=UR", "-w", a.elf, "-v", "-hardRst")
    print("flash:", "verified" if "verified successfully" in out else out[-400:], flush=True)
    time.sleep(3)
    r = read_gres(addr)
    print(f"set-up: stage={r['stage']} done={r['done_hex']} status={r['status']} arena_used={r['arena_used']} "
          f"in={r['in_type']} out={r['out_type']}\n  log: {r['log']!r}", flush=True)
    if r["stage"] < 4:
        json.dump(r, open(res / f"{a.name}_result.json", "w"), indent=1)
        print("MODEL DID NOT RUN -- no power measurement")
        return
    if a.runs == 0:
        return
    # timing pass: let the firmware finish once, so each capture can be sized to the burst
    t0 = time.time()
    while time.time() - t0 < 600:
        time.sleep(10)
        r = read_gres(addr)
        if r["done"] == 0xC0FFEE:
            break
    burst_s = r["burst_ms"] / 1000
    cap = 31 + burst_s + 8
    print(f"timing pass: burst {burst_s:.2f} s -> capture {cap:.0f} s per run", flush=True)
    for k in range(1, a.runs + 1):
        s = serial.Serial("COM5", 115200, timeout=1); s.reset_input_buffer()
        st("mode=UR", "-hardRst")
        n, t0 = 0, time.time()
        with open(res / f"{a.name}_run{k}.csv", "w", newline="") as f:
            f.write("t_us,vbus_V,vshunt_uV,current_mA,power_mW\n")
            while time.time() - t0 < cap:
                l = s.readline().decode(errors="ignore").strip()
                if l.count(",") == 4 and not l.startswith(("#", "t_us")):
                    f.write(l + "\n"); n += 1
        s.close(); print(f"run{k}: {n} samples", flush=True)
    r = read_gres(addr)
    r["t_lat_ms"] = r["burst_ms"] / 200
    json.dump(r, open(res / f"{a.name}_result.json", "w"), indent=1)
    print(f"burst: {r['burst_ms']} ms -> t_lat = {r['t_lat_ms']:.3f} ms  (done={r['done_hex']})  arena_used={r['arena_used']}")


if __name__ == "__main__":
    main()
