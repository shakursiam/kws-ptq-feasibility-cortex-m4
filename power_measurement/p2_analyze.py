#!/usr/bin/env python3
"""
p2_analyze.py -- Reviewer 1 comment 3: P_busy, P_sleep, P_active, E_marg, E_tot, P_avg.

Expected trace per run (P2 firmware):
    5 s boot + 10 s BUSY idle | flash | 15 s WFI SLEEP | flash | 200 inferences | end blink
The three plateaus are found automatically (1-D k-means on a median-filtered trace),
0.5 s is trimmed from each window edge (same as the paper), and the active-window
length gives an independent check of t_lat (= duration / 200).

Usage:
    python p2_analyze.py ../results/P2/int8_40mfcc_run*.csv --tlat-ms 158.72 --n-inf 200 --plot
"""
import argparse, glob
import numpy as np

TRIM_S = 0.5


def load(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    t = (d["t_us"] - d["t_us"][0]) / 1e6
    return t, d["power_mW"], d["current_mA"], d["vbus_V"]


def medfilt(x, k=9):
    h = k // 2
    xp = np.pad(x, h, mode="edge")
    return np.array([np.median(xp[i:i + k]) for i in range(len(x))])


def kmeans3(x):
    c = np.percentile(x, [5, 50, 95]).astype(float)
    for _ in range(100):
        lab = np.argmin(np.abs(x[:, None] - c[None]), axis=1)
        new = np.array([x[lab == j].mean() if np.any(lab == j) else c[j] for j in range(3)])
        if np.allclose(new, c):
            break
        c = new
    order = np.argsort(c)                       # 0 = sleep, 1 = busy, 2 = active
    return np.argsort(order)[lab], c[order]


def runs(lab):
    """contiguous (label, start, end_exclusive) runs"""
    out, s = [], 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[s]:
            out.append((lab[s], s, i)); s = i
    return out


def window_mean(t, p, s, e):
    m = (t >= t[s] + TRIM_S) & (t <= t[e - 1] - TRIM_S)
    return p[m].mean(), p[m].std(), int(m.sum())


def idx(t, a, b):
    """(start, end_exclusive) sample indices covering time interval [a, b]"""
    return int(np.searchsorted(t, a)), int(np.searchsorted(t, b, side="right"))


def analyze(path, n_inf, burst_s=None, flash_s=0.1, busy_s=10.0):
    t, p, i, v = load(path)
    pf = medfilt(p)
    if burst_s is None:
        # level-based segmentation (fine when active power is well above busy idle)
        lab, _ = kmeans3(pf)
        R = runs(lab)
        dur = lambda r: t[r[2] - 1] - t[r[1]]
        act = max((r for r in R if r[0] == 2), key=dur)
        slp = [r for r in R if r[0] == 0 and r[2] <= act[1] + 5 and dur(r) >= 3][-1]
        bsy = [r for r in R if r[0] == 1 and r[2] <= slp[1] + 5 and dur(r) >= 3][-1]
    else:
        # timing-anchored segmentation: the WFI-sleep plateau is the unambiguous anchor.
        # busy  = the busy_s seconds before the busy->sleep LED flash
        # active = starts after the sleep->active LED flash, lasts burst_s (firmware SysTick timer)
        lo = np.percentile(pf, 5); thr = lo + 0.5 * (np.median(pf) - lo)
        R = runs((pf < thr).astype(int))
        s0, s1 = max((r for r in R if r[0] == 1), key=lambda r: r[2] - r[1])[1:]
        t_s0, t_s1 = t[s0], t[s1 - 1]
        slp = (0,) + (s0, s1)
        bsy = (1,) + idx(t, t_s0 - flash_s - busy_s, t_s0 - flash_s)
        act = (2,) + idx(t, t_s1 + flash_s, t_s1 + flash_s + burst_s)
    res = {"file": path, "fs": (len(t) - 1) / (t[-1] - t[0])}
    for name, (_, s, e) in (("busy", bsy), ("sleep", slp), ("active", act)):
        m, sd, n = window_mean(t, p, s, e)
        res[name] = m
        res[name + "_sd"] = sd
        res[name + "_n"] = n
        res[name + "_dur"] = t[e - 1] - t[s]
        res[name + "_span"] = (t[s], t[e - 1])
    res["vbus"] = v[(t >= res["active_span"][0]) & (t <= res["active_span"][1])].mean()
    res["tlat_from_trace_ms"] = res["active_dur"] / n_inf * 1e3
    res["t"], res["p"] = t, p
    return res


def ms(vals):
    a = np.asarray(vals, float)
    return a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--tlat-ms", type=float, default=158.72, help="Validate-on-target latency")
    ap.add_argument("--n-inf", type=int, default=200)
    ap.add_argument("--hops", type=float, nargs="*", default=[200, 500, 1000], help="T_hop in ms")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--anchored", action="store_true",
                    help="timing-anchored windows: active = n_inf * t_lat after the sleep->active flash")
    a = ap.parse_args()
    files = sorted(sum((glob.glob(c) for c in a.csv), []))
    tl = a.tlat_ms / 1e3

    rs = [analyze(f, a.n_inf, burst_s=(a.n_inf * tl if a.anchored else None)) for f in files]
    print(f"{'run':<28}{'fs Hz':>6}{'P_busy':>9}{'P_sleep':>9}{'P_act':>9}"
          f"{'busy s':>8}{'sleep s':>8}{'act s':>8}{'t_lat(trace) ms':>17}")
    for r in rs:
        print(f"{r['file'].split('/')[-1].split(chr(92))[-1]:<28}{r['fs']:6.1f}{r['busy']:9.2f}{r['sleep']:9.2f}"
              f"{r['active']:9.2f}{r['busy_dur']:8.1f}{r['sleep_dur']:8.1f}{r['active_dur']:8.1f}"
              f"{r['tlat_from_trace_ms']:17.1f}")
        for k in ("busy", "sleep", "active"):
            if r[k + "_dur"] < {"busy": 8, "sleep": 12, "active": 25}[k]:
                print(f"   WARNING: {k} window only {r[k+'_dur']:.1f} s - check the trace (--plot)")

    pb, ps, pa = ms([r["busy"] for r in rs]), ms([r["sleep"] for r in rs]), ms([r["active"] for r in rs])
    marg_b = ms([(r["active"] - r["busy"]) * tl for r in rs])
    marg_s = ms([(r["active"] - r["sleep"]) * tl for r in rs])
    etot = ms([r["active"] * tl for r in rs])
    print(f"\nN runs = {len(rs)}   (mean +/- std across runs), t_lat = {a.tlat_ms} ms, "
          f"Vbus = {np.mean([r['vbus'] for r in rs]):.3f} V")
    print(f"P_busy   (HAL_Delay, 180 MHz) = {pb[0]:8.2f} +/- {pb[1]:.2f} mW")
    print(f"P_sleep  (WFI Sleep, 180 MHz) = {ps[0]:8.2f} +/- {ps[1]:.2f} mW")
    print(f"P_active (inference)          = {pa[0]:8.2f} +/- {pa[1]:.2f} mW")
    print(f"E_marg vs busy  (Pa-Pb)*t     = {marg_b[0]:8.3f} +/- {marg_b[1]:.3f} mJ")
    print(f"E_marg vs sleep (Pa-Ps)*t     = {marg_s[0]:8.3f} +/- {marg_s[1]:.3f} mJ")
    print(f"E_tot           Pa*t          = {etot[0]:8.3f} +/- {etot[1]:.3f} mJ")
    print(f"t_lat from active-window length = {np.mean([r['tlat_from_trace_ms'] for r in rs]):.1f} ms "
          f"(includes ~1 sample of edge blur; Validate-on-target is the reference)")
    print(f"\n{'T_hop ms':>9}{'P_avg busy-idle mW':>21}{'P_avg WFI-idle mW':>20}{'saving':>9}")
    for h in a.hops:
        T = h / 1e3
        if T < tl:
            print(f"{h:9.0f}   infeasible (T_hop < t_lat)"); continue
        pab = (etot[0] + pb[0] * (T - tl)) / T
        pas = (etot[0] + ps[0] * (T - tl)) / T
        print(f"{h:9.0f}{pab:21.2f}{pas:20.2f}{(1 - pas / pab) * 100:8.1f}%")

    if a.plot:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        r = rs[0]
        fig, ax = plt.subplots(figsize=(7, 2.6))
        ax.plot(r["t"], r["p"], lw=0.6, color="0.3")
        for k, c in (("busy", "tab:orange"), ("sleep", "tab:blue"), ("active", "tab:red")):
            s, e = r[k + "_span"]
            ax.axvspan(s + TRIM_S, e - TRIM_S, color=c, alpha=0.15)
            ax.hlines(r[k], s + TRIM_S, e - TRIM_S, color=c, lw=1.5, label=f"{k}: {r[k]:.1f} mW")
        ax.set_xlabel("time (s)"); ax.set_ylabel("power (mW)"); ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        out = files[0].rsplit(".", 1)[0] + "_trace"
        fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf")
        print(f"\nplot: {out}.png / .pdf")


if __name__ == "__main__":
    main()
