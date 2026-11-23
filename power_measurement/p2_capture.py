#!/usr/bin/env python3
"""
p2_capture.py -- record the ESP32-S3/INA238 stream to a CSV file.

Usage (one run):
    python p2_capture.py --port COM5 --seconds 75 --out ../results/P2/int8_40mfcc_run1.csv
Start it, wait for "RECORDING", then press the black RESET button on the Nucleo.
"""
import argparse, sys, time
import serial  # pip install pyserial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="ESP32-S3 COM port, e.g. COM5")
    ap.add_argument("--seconds", type=float, default=75)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ser = serial.Serial(a.port, 115200, timeout=1)
    time.sleep(0.5)
    ser.reset_input_buffer()
    n, bad, last_print = 0, 0, 0.0
    with open(a.out, "w", newline="") as f:
        f.write("t_us,vbus_V,vshunt_uV,current_mA,power_mW\n")
        print(f"RECORDING {a.seconds:.0f} s -> {a.out}\n>>> press RESET on the Nucleo now <<<", flush=True)
        t0 = time.time()
        while time.time() - t0 < a.seconds:
            line = ser.readline().decode(errors="ignore").strip()
            if not line or line.startswith("#") or line.startswith("t_us"):
                if line.startswith("#"):
                    print(line)
                continue
            parts = line.split(",")
            if len(parts) != 5:
                bad += 1
                continue
            f.write(line + "\n")
            n += 1
            el = time.time() - t0
            if el - last_print >= 1.0:
                last_print = el
                print(f"\r  t={el:5.1f}s  samples={n:5d}  P={float(parts[4]):7.2f} mW  "
                      f"I={float(parts[3]):6.2f} mA  Vbus={float(parts[1]):.3f} V   ", end="", flush=True)
    ser.close()
    dur = time.time() - t0
    print(f"\nDone: {n} samples in {dur:.1f} s ({n/dur:.1f} Hz), {bad} malformed lines.")
    if n / dur < 30:
        print("WARNING: sample rate looks low - is the ESP32 logger running?", file=sys.stderr)


if __name__ == "__main__":
    main()
