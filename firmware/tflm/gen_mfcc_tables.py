"""Generate CMSIS-DSP arm_mfcc tables that reproduce the paper's TensorFlow front end
(640-sample periodic Hann window zero-padded to a 1024-point FFT, 40 mel bands 20-4000 Hz,
log(+1e-6), DCT-II as in tf.signal.mfccs_from_log_mel_spectrograms) for 40 and 10 coefficients."""
import sys
import numpy as np
import tensorflow as tf

out = sys.argv[1]
N_FFT, FRAME, MEL = 1024, 640, 40
W = tf.signal.linear_to_mel_weight_matrix(MEL, N_FFT // 2 + 1, 16000, 20.0, 4000.0).numpy()  # (513, 40)
pos, lens, coefs = [], [], []
for j in range(MEL):
    nz = np.nonzero(W[:, j])[0]
    a, b = int(nz[0]), int(nz[-1]) + 1
    pos.append(a); lens.append(b - a); coefs += W[a:b, j].tolist()
win = np.zeros(N_FFT, np.float32)
win[:FRAME] = tf.signal.hann_window(FRAME, periodic=True).numpy()
D = tf.signal.mfccs_from_log_mel_spectrograms(tf.eye(MEL)).numpy().T  # (40 out, 40 in): y = D @ logmel


def arr(name, typ, v, fmt):
    return f"static const {typ} {name}[{len(v)}] = {{" + ",".join(fmt(x) for x in v) + "};\n"


def f32(x):
    t = f"{float(x):.9g}"
    return (t if ("." in t or "e" in t) else t + ".0") + "f"
s = "#pragma once\n#include <stdint.h>\n"
s += f"#define MFCC_FFT {N_FFT}\n#define MFCC_FRAME {FRAME}\n#define MFCC_MEL {MEL}\n#define MFCC_STEP 320\n#define MFCC_NFRAMES 49\n"
s += arr("mel_pos", "uint32_t", pos, str) + arr("mel_len", "uint32_t", lens, str)
s += arr("mel_coef", "float", coefs, f32) + arr("win_coef", "float", win, f32)
s += arr("dct40", "float", D[:40].ravel(), f32) + arr("dct10", "float", D[:10].ravel(), f32)
open(out, "w").write(s)
print(f"{len(coefs)} mel coefs, filters {min(lens)}-{max(lens)} bins")
