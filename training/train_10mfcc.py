#!/usr/bin/env python3
"""
P3 -- Reviewer 1 comment 1: a second model where the memory gate opens.

Retrains DS-CNN-S exactly as Speech_Command.ipynb does (same split, same balancing,
same front end, same architecture, same optimiser/callbacks), changing ONE thing:
NUM_MFCC 40 -> 10 (input 49x10, the Hello Edge DS-CNN-S setting). Then exports the
same four PTQ variants and evaluates them on the paper split and the official test set.

Usage:
    python p3_train_10mfcc.py --project "<DRIVE>/icece2026_tinyml" \
        --official <official_test/extracted> --out ../results/P3
"""
import argparse, glob, os, random, sys, time
from pathlib import Path
from collections import Counter
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

sys.path.insert(0, str(Path(__file__).parent))
from bootstrap_ci import report, run_tflite  # noqa: E402

NUM_MFCC = 10
TARGET_WORDS = {'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'}
LABELS = ['_silence_', '_unknown_', 'yes', 'no', 'up', 'down',
          'left', 'right', 'on', 'off', 'stop', 'go']
L2I = {l: i for i, l in enumerate(LABELS)}


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
    return tf.signal.mfccs_from_log_mel_spectrograms(log_mel)[..., :NUM_MFCC]


def features(samples):
    ds = tf.data.Dataset.from_tensor_slices(([s[0] for s in samples], [L2I[s[1]] for s in samples]))
    ds = ds.map(lambda p, l: (tf.expand_dims(compute_mfcc(load_wav(p)), -1), l),
                num_parallel_calls=tf.data.AUTOTUNE).batch(256)
    xs, ys = [], []
    for bx, by in ds:
        xs.append(bx.numpy()); ys.append(by.numpy())
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.int64)


def balanced_splits(project):
    """Same file lists, glob order, seed-42 shuffles and silence clips as the notebook."""
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
        c = Counter(l for _, l in t)
        return int(round(sum(c.values()) / len(c)))

    tr_t, tr_u = split(train); va_t, va_u = split(val); te_t, te_u = split(test)
    tr_a, va_a, te_a = avg(tr_t), avg(va_t), avg(te_t)
    random.seed(42); np.random.seed(42)
    random.shuffle(tr_u); random.shuffle(va_u); random.shuffle(te_u)
    sil = D / "_silence_generated_"
    S = lambda split_dir, n: [(str(sil / split_dir / f"silence_{i:05d}.wav"), "_silence_") for i in range(n)]
    tr = tr_t + tr_u[:tr_a] + S("train", tr_a)
    va = va_t + va_u[:va_a] + S("val", va_a)
    te = te_t + te_u[:te_a] + S("test", te_a)
    random.shuffle(tr); random.shuffle(va); random.shuffle(te)
    return tr, va, te


def official_samples(root):
    out = []
    for d in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        if d.name in L2I:
            out += [(str(f), d.name) for f in sorted(d.glob("*.wav"))]
    return out


def build_ds_cnn_s(input_shape, num_classes=12):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (10, 4), strides=(2, 2), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    for _ in range(4):
        x = layers.DepthwiseConv2D((3, 3), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
        x = layers.Conv2D(64, (1, 1), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs, name="DS-CNN-S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--official")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()
    out = Path(a.out).resolve(); (out / "models").mkdir(parents=True, exist_ok=True)
    official = Path(a.official).resolve() if a.official else None
    tf.keras.utils.set_random_seed(42)

    t = time.time()
    tr, va, te = balanced_splits(a.project)
    for n, s in (("train", tr), ("val", va), ("test", te)):
        print(f"{n}: {len(s)} clips {dict(Counter(l for _, l in s))}", flush=True)
    xtr, ytr = features(tr); xva, yva = features(va); xte, yte = features(te)
    print(f"features {xtr.shape} {xva.shape} {xte.shape} in {time.time()-t:.0f}s", flush=True)
    np.save(out / "x_test_10mfcc.npy", xte); np.save(out / "y_test.npy", yte)
    np.save(out / "y_test_onehot.npy", np.eye(12, dtype=np.float32)[yte])

    train_ds = (tf.data.Dataset.from_tensor_slices((xtr, ytr))
                .shuffle(len(xtr), seed=42, reshuffle_each_iteration=True).batch(64).prefetch(2))
    val_ds = tf.data.Dataset.from_tensor_slices((xva, yva)).batch(64)

    model = build_ds_cnn_s((49, NUM_MFCC, 1))
    model.summary(print_fn=lambda s: print(s, flush=True))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cb = [tf.keras.callbacks.ModelCheckpoint(str(out / "models/ds_cnn_s10_best.keras"),
                                             monitor="val_accuracy", save_best_only=True, verbose=1),
          tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8,
                                           restore_best_weights=True, verbose=1),
          tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=4, verbose=1)]
    t = time.time()
    model.fit(train_ds, validation_data=val_ds, epochs=a.epochs, callbacks=cb, verbose=2)
    print(f"training {time.time()-t:.0f}s", flush=True)
    model.save(out / "models/ds_cnn_s10_fp32.keras")

    def conv(**kw):
        c = tf.lite.TFLiteConverter.from_keras_model(model)
        for k, v in kw.items():
            setattr(c, k, v)
        return c

    def rep():
        for f, _ in train_ds.unbatch().batch(1).take(300):
            yield [tf.cast(f, tf.float32)]

    c = conv(); fp32 = c.convert()
    c = conv(optimizations=[tf.lite.Optimize.DEFAULT]); c.target_spec.supported_types = [tf.float16]; fp16 = c.convert()
    c = conv(optimizations=[tf.lite.Optimize.DEFAULT]); dr = c.convert()
    c = conv(optimizations=[tf.lite.Optimize.DEFAULT], representative_dataset=rep,
             inference_input_type=tf.int8, inference_output_type=tf.int8)
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]; full = c.convert()
    V = {}
    for k, b in (("fp32", fp32), ("fp16", fp16), ("dr_int8", dr), ("full_int8", full)):
        p = out / "models" / f"ds_cnn_s10_{k}.tflite"; p.write_bytes(b); V[k] = p
        print(f"{p.name}: {len(b):,} B", flush=True)

    def evaluate(name, x, y):
        print(f"\n==================== {name} ====================", flush=True)
        P = {k: run_tflite(str(p), x) for k, p in V.items()}
        for k in P:
            print(f"  {k:<10} acc {np.mean(P[k] == y)*100:6.2f}%", flush=True)
        np.savez(out / f"predictions_{name}.npz", y=y, **P)
        report(P, y)

    evaluate("paper_split", xte, yte)
    if official:
        s = official_samples(official)
        xo, yo = features(s)
        evaluate("official", xo, yo)


if __name__ == "__main__":
    main()
