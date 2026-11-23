#!/usr/bin/env python3
"""
P1 -- Reviewer 1 comments 4, 7, 8.

1. Rebuilds the paper's 4,888-clip test split EXACTLY as evaluate_and_save_cm.py does
   (same file lists, same glob order, same seed-42 shuffles, same MFCC front end).
2. Runs all four .tflite variants on it -> per-clip predictions -> bootstrap CIs + McNemar.
3. Same for the OFFICIAL Speech Commands v2 12-class test set
   (speech_commands_test_set_v0.02, folder per class incl. _silence_ / _unknown_).
4. Saves x_test.npy / y_test_onehot.npy for X-CUBE-AI "Validate on target" with custom data.

Usage:
    python p1_eval.py --project "<DRIVE>/icece2026_tinyml" --official <extracted_dir> --out <results/P1>
"""
import argparse, glob, os, random, sys, time
from pathlib import Path
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))
from bootstrap_ci import report, run_tflite  # noqa: E402

TARGET_WORDS = {'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'}
LABELS = ['_silence_', '_unknown_', 'yes', 'no', 'up', 'down',
          'left', 'right', 'on', 'off', 'stop', 'go']
L2I = {l: i for i, l in enumerate(LABELS)}
VARIANTS = {"fp32": "ds_cnn_s_fp32.tflite", "fp16": "ds_cnn_s_fp16.tflite",
            "dr_int8": "ds_cnn_s_dr_int8.tflite", "full_int8": "ds_cnn_s_full_int8.tflite"}


# ---- front end: identical to evaluate_and_save_cm.py ---------------------------------
def load_wav(path):
    audio_binary = tf.io.read_file(path)
    waveform, _ = tf.audio.decode_wav(audio_binary, desired_channels=1, desired_samples=16000)
    return tf.squeeze(waveform, axis=-1)


def compute_mfcc(waveform):
    stft = tf.signal.stft(waveform, frame_length=640, frame_step=320, fft_length=1024)
    spectrogram = tf.abs(stft)
    mel = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=40, num_spectrogram_bins=spectrogram.shape[-1],
        sample_rate=16000, lower_edge_hertz=20.0, upper_edge_hertz=4000.0)
    log_mel = tf.math.log(tf.tensordot(spectrogram, mel, 1) + 1e-6)
    return tf.signal.mfccs_from_log_mel_spectrograms(log_mel)[..., :40]


def features(samples):
    ds = tf.data.Dataset.from_tensor_slices(([s[0] for s in samples], [L2I[s[1]] for s in samples]))
    ds = ds.map(lambda p, l: (tf.expand_dims(compute_mfcc(load_wav(p)), -1), l),
                num_parallel_calls=tf.data.AUTOTUNE).batch(256)
    xs, ys = [], []
    for bx, by in ds:
        xs.append(bx.numpy()); ys.append(by.numpy())
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.int64)


# ---- the paper's split: copied logic from evaluate_and_save_cm.py ---------------------
def paper_test_samples(project):
    os.chdir(project)
    D = Path("./speech_commands_v2")
    val_files = set((D / "validation_list.txt").read_text().splitlines())
    test_files = set((D / "testing_list.txt").read_text().splitlines())
    train, val, test = [], [], []
    for wav_path in glob.glob(str(D / "*/*.wav")):
        rel = Path(wav_path).relative_to(D).as_posix()
        if "_background_noise_" in rel or "_silence_generated_" in rel:
            continue
        word = rel.split("/")[0]
        e = (wav_path, word if word in TARGET_WORDS else "_unknown_")
        (test if rel in test_files else val if rel in val_files else train).append(e)

    def split(s):
        return [x for x in s if x[1] != '_unknown_'], [x for x in s if x[1] == '_unknown_']

    def avg(t):
        c = {}
        for _, l in t:
            c[l] = c.get(l, 0) + 1
        return int(round(sum(c.values()) / len(c)))

    tr_t, tr_u = split(train); va_t, va_u = split(val); te_t, te_u = split(test)
    tr_a, va_a, te_a = avg(tr_t), avg(va_t), avg(te_t)
    random.seed(42); np.random.seed(42)
    random.shuffle(tr_u); random.shuffle(va_u); random.shuffle(te_u)
    sil = D / "_silence_generated_" / "test"
    test_bal = te_t + te_u[:te_a] + [(str(sil / f"silence_{i:05d}.wav"), "_silence_") for i in range(te_a)]
    # reproduce the RNG stream of the original script (train/val shuffles happen before test)
    tr_bal = tr_t + tr_u[:tr_a] + [("", "_silence_")] * tr_a
    va_bal = va_t + va_u[:va_a] + [("", "_silence_")] * va_a
    random.shuffle(tr_bal); random.shuffle(va_bal); random.shuffle(test_bal)
    return test_bal


def official_samples(root):
    root = Path(root)
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        lab = d.name if d.name in L2I else None
        if lab is None:
            continue
        out += [(str(f), lab) for f in sorted(d.glob("*.wav"))]
    return out


def evaluate(name, x, y, project, out):
    print(f"\n==================== {name} ====================", flush=True)
    P = {}
    for k, f in VARIANTS.items():
        t = time.time()
        P[k] = run_tflite(str(Path(project) / f), x)
        print(f"  {k:<10} acc {np.mean(P[k] == y)*100:6.2f}%  ({time.time()-t:.0f}s)", flush=True)
    np.savez(out / f"predictions_{name}.npz", y=y, **P)
    report(P, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--official")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    official = Path(a.official).resolve() if a.official else None

    t = time.time()
    s = paper_test_samples(a.project)
    from collections import Counter
    print(f"paper test split: {len(s)} clips  {dict(Counter(l for _, l in s))}", flush=True)
    x, y = features(s)
    print(f"features {x.shape} in {time.time()-t:.0f}s", flush=True)
    np.save(out / "x_test.npy", x)
    np.save(out / "y_test.npy", y)
    np.save(out / "y_test_onehot.npy", np.eye(12, dtype=np.float32)[y])
    evaluate("paper_split", x, y, a.project, out)

    if official:
        s = official_samples(official)
        print(f"\nofficial test set: {len(s)} clips  {dict(Counter(l for _, l in s))}", flush=True)
        x, y = features(s)
        np.save(out / "x_official.npy", x); np.save(out / "y_official.npy", y)
        evaluate("official", x, y, a.project, out)


if __name__ == "__main__":
    main()
