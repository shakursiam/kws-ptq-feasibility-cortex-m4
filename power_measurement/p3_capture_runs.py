"""Flash elf, then 3x (ST-LINK hard reset + 76 s INA238 capture), then read g_burst_done/g_burst_ms."""
import serial, subprocess, sys, time
P = r"C:\ST\STM32CubeIDE_2.2.0\STM32CubeIDE\plugins\com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer.win32_2.2.500.202603051304\tools\bin\STM32_Programmer_CLI.exe"
elf, res, v, addr = sys.argv[1:5]
r = subprocess.run([P, "-c", "port=SWD", "mode=UR", "-w", elf, "-v"], capture_output=True, text=True)
print("flash:", "verified" if "verified successfully" in r.stdout else r.stdout[-400:], flush=True)
for k in (1, 2, 3):
    s = serial.Serial("COM5", 115200, timeout=1); s.reset_input_buffer()
    subprocess.run([P, "-c", "port=SWD", "mode=UR", "-hardRst"], capture_output=True)
    n, t0 = 0, time.time()
    with open(f"{res}/{v}_run{k}.csv", "w", newline="") as f:
        f.write("t_us,vbus_V,vshunt_uV,current_mA,power_mW\n")
        while time.time() - t0 < 76:
            l = s.readline().decode(errors="ignore").strip()
            if l.count(",") == 4 and not l.startswith(("#", "t_us")):
                f.write(l + "\n"); n += 1
    s.close(); print(f"run{k}: {n} samples", flush=True)
r = subprocess.run([P, "-c", "port=SWD", "mode=HOTPLUG", "-r32", addr, "8"], capture_output=True, text=True)
line = [l for l in r.stdout.splitlines() if l.strip().lower().startswith(addr.lower())]
print("burst timer:", line, flush=True)
if line:
    done, ms = [int(x, 16) for x in line[0].split(":")[1].split()[:2]]
    open(f"{res}/{v}_burst.txt", "w").write(f"done=0x{done:X} burst_ms={ms} t_lat_ms={ms/200:.3f}\n")
    print(f"t_lat = {ms/200:.3f} ms ({'ok' if done == 0xC0FFEE else 'NOT DONE'})")
