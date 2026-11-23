#!/usr/bin/env python3
"""
count_macs.py -- count multiply-accumulates (MACs) per inference of a .tflite model.

Reviewer 1, comment 2: "Please report the MACs per inference".
Cross-check the result against the "MACC" line in the X-CUBE-AI Analyze report.

Usage:
    python count_macs.py ds_cnn_s_full_int8.tflite [--latency-ms 158.72 --clock-mhz 180]

Counts CONV_2D, DEPTHWISE_CONV_2D and FULLY_CONNECTED (these are >99% of the work
in DS-CNN). Pooling / softmax / add are listed separately as "other" element ops.
"""
import argparse
import numpy as np
import tensorflow as tf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tflite")
    ap.add_argument("--latency-ms", type=float, default=None)
    ap.add_argument("--clock-mhz", type=float, default=180.0)
    a = ap.parse_args()

    itp = tf.lite.Interpreter(model_path=a.tflite)
    itp.allocate_tensors()
    shapes = {t["index"]: tuple(t["shape"]) for t in itp.get_tensor_details()}
    ops = itp._get_ops_details()  # available in TF >= 2.5

    total = 0
    print(f"{'#':>3}  {'op':<20} {'output':<18} {'weights':<18} {'MACs':>12}")
    for op in ops:
        name = op["op_name"]
        ins, outs = list(op["inputs"]), list(op["outputs"])
        if name not in ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"):
            continue
        out = shapes[outs[0]]
        w = shapes[ins[1]]
        if name == "CONV_2D":            # w: [Cout, kh, kw, Cin]; out: [1, H, W, Cout]
            macs = int(np.prod(out[1:])) * w[1] * w[2] * w[3]
        elif name == "DEPTHWISE_CONV_2D":  # w: [1, kh, kw, Cout]
            macs = int(np.prod(out[1:])) * w[1] * w[2]
        else:                              # FC  w: [Cout, Cin]
            macs = w[0] * w[1]
        total += macs
        print(f"{op['index']:>3}  {name:<20} {str(out):<18} {str(w):<18} {macs:>12,}")

    print(f"\nTotal MACs per inference: {total:,}  ({total/1e6:.2f} M)")
    if a.latency_ms:
        cycles = a.latency_ms * 1e-3 * a.clock_mhz * 1e6
        print(f"Cycles per inference    : {cycles:,.0f}")
        print(f"Cycles per MAC          : {cycles/total:.2f}")
        print(f"Throughput              : {total/(a.latency_ms*1e-3)/1e6:.1f} MMAC/s")


if __name__ == "__main__":
    main()
