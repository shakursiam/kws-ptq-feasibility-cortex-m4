#!/usr/bin/env python3
"""
bootstrap_ci.py -- Reviewer 1, comment 4.

(a) Runs every .tflite variant on the SAME test set and saves per-clip predictions.
(b) Per-variant accuracy with 95% percentile-bootstrap CI.
(c) PAIRED bootstrap CI of the accuracy difference between every variant pair
    (same resampled clips for both models -> much tighter than two separate CIs).
(d) Exact McNemar test for each pair (are the disagreements one-sided?).

Usage:
    python bootstrap_ci.py --x x_test.npy --y y_test.npy \
        --models fp32=ds_cnn_s_fp32.tflite fp16=ds_cnn_s_fp16.tflite \
                 dr_int8=ds_cnn_s_dr_int8.tflite full_int8=ds_cnn_s_full_int8.tflite
    # x_test.npy: float32 MFCC features, shape (N, 49, 40, 1) -- the same ones used for Table I
    # y_test.npy: integer labels, shape (N,)

    # Multi-seed mode: pass a CSV of accuracies (rows=seeds, cols=variants) instead
    python bootstrap_ci.py --seed-csv seed_accuracies.csv
"""
import argparse
import itertools
import numpy as np

B = 10_000
RNG = np.random.default_rng(0)


def run_tflite(path, x):
    import tensorflow as tf
    itp = tf.lite.Interpreter(model_path=path)
    itp.allocate_tensors()
    inp, out = itp.get_input_details()[0], itp.get_output_details()[0]
    preds = np.empty(len(x), dtype=np.int64)
    for i, xi in enumerate(x):
        xi = xi[None].astype(np.float32)
        if inp["dtype"] in (np.int8, np.uint8):          # full-integer model
            s, z = inp["quantization"]
            xi = np.clip(np.round(xi / s + z), -128, 127).astype(inp["dtype"])
        itp.set_tensor(inp["index"], xi)
        itp.invoke()
        preds[i] = int(np.argmax(itp.get_tensor(out["index"])[0]))
    return preds


def mcnemar_exact(c_a, c_b):
    """c_a: correct by A only, c_b: correct by B only. Two-sided exact binomial p."""
    from math import comb
    n, k = c_a + c_b, min(c_a, c_b)
    if n == 0:
        return 1.0
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x"); ap.add_argument("--y")
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--preds", help="optional .npz with saved predictions (skip inference)")
    ap.add_argument("--seed-csv")
    a = ap.parse_args()

    if a.seed_csv:
        import csv
        rows = list(csv.reader(open(a.seed_csv)))
        head, vals = rows[0], np.array(rows[1:], dtype=float)
        print(f"{len(vals)} seeds")
        for j, h in enumerate(head):
            print(f"{h:<10} mean {vals[:, j].mean():.2f}  std {vals[:, j].std(ddof=1):.2f}"
                  f"  min {vals[:, j].min():.2f}  max {vals[:, j].max():.2f}")
        return

    y = np.load(a.y).astype(np.int64)
    if a.preds:
        P = dict(np.load(a.preds))
    else:
        x = np.load(a.x)
        P = {}
        for spec in a.models:
            name, path = spec.split("=", 1)
            print(f"running {name} ...")
            P[name] = run_tflite(path, x)
        np.savez("predictions.npz", **P)
        print("saved predictions.npz (reuse with --preds)")

    report(P, y)


def boot_means(v, chunk=500):
    """B bootstrap means of vector v; the same RNG seed gives the same resamples for every call."""
    rng = np.random.default_rng(0)
    N, out = len(v), []
    for s in range(0, B, chunk):
        idx = rng.integers(0, N, size=(min(chunk, B - s), N))
        out.append(v[idx].mean(1))
    return np.concatenate(out) * 100


def report(P, y):
    N = len(y)
    C = {k: (v == y).astype(np.float64) for k, v in P.items()}

    print(f"\nN = {N}, bootstrap B = {B}")
    print(f"{'variant':<10} {'acc %':>7}  {'correct':>9}  {'95% CI':>17}")
    for k, c in C.items():
        bs = boot_means(c)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"{k:<10} {c.mean()*100:7.2f}  {int(c.sum()):>4}/{N:<4}  [{lo:6.2f}, {hi:6.2f}]")

    print(f"\n{'pair (A - B)':<22} {'diff pp':>8} {'95% paired CI':>18} {'A-only':>7} {'B-only':>7} {'McNemar p':>10}")
    for ka, kb in itertools.combinations(C, 2):
        d = (C[ka] - C[kb])
        bs = boot_means(d)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        a_only = int(((C[ka] == 1) & (C[kb] == 0)).sum())
        b_only = int(((C[ka] == 0) & (C[kb] == 1)).sum())
        print(f"{ka+' - '+kb:<22} {d.mean()*100:8.2f}  [{lo:6.2f}, {hi:6.2f}] {a_only:7d} {b_only:7d}"
              f" {mcnemar_exact(a_only, b_only):10.3f}")
    print("\nIf a paired CI contains 0 (and McNemar p > 0.05), do NOT rank those two variants.")


if __name__ == "__main__":
    main()
