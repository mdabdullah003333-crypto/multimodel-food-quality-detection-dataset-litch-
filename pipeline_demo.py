"""End-to-end PIPELINE WALKTHROUGH (thesis demo cell / Cell 6).
Prints the actual numbers flowing through every methodology layer for
three real test samples (one per class): raw sensor readings -> moving
average -> adaptive Kalman -> baseline tracking -> min-max -> visual Fv
-> joint matrix -> int8 model -> freshness index -> Mahalanobis
'Chemical Masking' -> final 3-class decision.
Reuses fv_cache.npy + int8 flatbuffer + SavedModel + train_ladd_tf.py.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")


def pick_base():
    for base in ["/workspace", "/kaggle/working"]:
        if os.path.isdir(os.path.join(base, "data", "images", "Fresh")):
            return base
    raise FileNotFoundError("dataset not found: expected <base>/data/images/Fresh")


BASE = pick_base()
DATA = f"{BASE}/data"
sys.path.insert(0, BASE)

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import train_ladd_tf as tl
tl.DATA = DATA

CLASSES = ["Fresh", "Rotten", "Formalin-Mixed"]
np.set_printoptions(precision=3, suppress=True, floatmode="fixed")


def hdr(t):
    print(f"\n{'=' * 64}\n{t}\n{'=' * 64}")


def int8_score_vec(inter, qi, qo, X):
    qscale, qzp = qi["quantization"]
    oscale, ozp = qo["quantization"]
    out = []
    for x in X:
        inter.set_tensor(qi["index"], np.clip(np.round(x[None] / qscale + qzp),
                                              -128, 127).astype(np.int8))
        inter.invoke()
        out.append((inter.get_tensor(qo["index"]).astype(np.float32) - ozp) * oscale)
    return np.array(out, np.float32).ravel()


def main():
    df = tl.load_manifest()
    y = df["label"].to_numpy()
    tr, tmp = train_test_split(np.arange(len(df)), test_size=0.30, stratify=y, random_state=42)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=42)

    demo = [int(np.where(y[te] == c)[0][0]) for c in range(3)]
    rows = [te[i] for i in demo]

    hdr("STAGE 1 | SYNCHRONIZED INGESTION (CSV row paired with its image)")
    print(f"dataset pairs: {len(df)}/3000 | classes: "
          + ", ".join(f"{CLASSES[c]}={(y == c).sum()}" for c in range(3)))
    for r in rows:
        print(f"  row {r:4d}  {CLASSES[y[r]]:13s} image={os.path.basename(df['path'][r])}")

    hdr("STAGE 2 | RAW TELEMETRY (7 sensor channels, straight from LADD CSV)")
    print("  " + "  ".join(f"{s[:6]:>7s}" for s in tl.SENSORS))
    for r in rows:
        vec = df[tl.SENSORS].to_numpy(float)[r]
        print(f"  {CLASSES[y[r]][:13]:13s} " + "  ".join(f"{v:7.2f}" for v in vec))

    hdr("STAGE 2b | MOVING AVERAGE (w=7) — window noise smoothing")
    raw = df[tl.SENSORS].to_numpy(float).copy()
    ma = tl.moving_average(raw, w=7)
    r = rows[2]
    print(f"  example ({CLASSES[y[r]]}, row {r}), Formaldehyde:")
    print(f"    raw values  rows {r}-{r+3}: {raw[r:r+4, 6]}")
    print(f"    after MA(7)        : {ma[r:r+4, 6]}")

    hdr("STAGE 3 | ADAPTIVE KALMAN FILTER + BASELINE TRACKING")
    kal = raw.copy()
    for j in range(kal.shape[1]):
        kal[:, j] = tl.adaptive_kalman(ma[:, j])
        kal[:, j] -= np.median(kal[:100, j])
    lab_raw = np.load(f"{BASE}/fv_cache.npy")
    fv = lab_raw
    print("  Kalman gain adapts to innovation (r scales with 10-sample window);")
    print("  baseline = median of first 100 measurements per channel (drift removal)")
    print(f"  example ({CLASSES[y[r]]}, row {r}), Formaldehyde: raw={raw[r,6]:.3f} -> MA={ma[r,6]:.3f} -> Kalman+baseline={kal[r,6]:.3f}")
    print(f"  baseline (first-100 medians): {np.median(kal[:100], axis=0)}")

    hdr("STAGE 4 | MIN-MAX SCALING (fit on TRAIN split, applied to all)")
    senm = tl.minmax_scale(kal[tr], kal)[1].astype(np.float32)
    for r in rows:
        print(f"  {CLASSES[y[r]][:13]:13s} {senm[r]}")

    hdr("STAGE 5 | VISUAL FEATURE VECTOR Fv (MobileNetV2, ImageNet, 1280-d)")
    print(f"  Fv shape per sample: {fv.shape[1]} dims | cache: {fv.shape}")
    for r in rows:
        print(f"  {CLASSES[y[r]][:13]:13s} Fv[:5]={fv[r][:5]}  |Fv|={np.linalg.norm(fv[r]):.2f}")

    hdr("STAGE 6 | JOINT FEATURE MATRIX (late-fusion input = Fv | telemetry)")
    for r in rows:
        joint = np.hstack([fv[r], senm[r]])
        print(f"  {CLASSES[y[r]][:13]:13s} v[joint]={joint.shape[0]} dims | "
              f"telemetry part={joint[1280:]}")

    hdr("STAGE 7 | HYBRID MODEL INFERENCE (int8 TFLite, batch=1 — as on ESP32-S3)")
    inter = tf.lite.Interpreter(f"{BASE}/hybrid_freshness_int8.tflite")
    inter.allocate_tensors()
    qi, qo = inter.get_input_details()[0], inter.get_output_details()[0]
    joint_tr = np.hstack([fv[tr], senm[tr]]).astype(np.float32)
    joint_te = np.hstack([fv[te], senm[te]]).astype(np.float32)
    scores_te = int8_score_vec(inter, qi, qo, joint_te[[i for i in demo]])
    for k, r in enumerate(rows):
        print(f"  {CLASSES[y[te][demo[k]]][:13]:13s} freshness index = {scores_te[k]:.2f} / 100")

    hdr("STAGE 8 | MAHALANOBIS ANOMALY — 'CHEMICAL MASKING' CHECK")
    gas = [4, 5, 6]
    s_all_te = int8_score_vec(inter, qi, qo, joint_te)
    s_all_tr = int8_score_vec(inter, qi, qo, joint_tr)
    vec_te = np.column_stack([s_all_te, senm[te][:, gas]])
    vec_tr = np.column_stack([s_all_tr, senm[tr][:, gas]])
    fr_vec = vec_tr[y[tr] == 0]
    mu, cov = fr_vec.mean(0), np.cov(fr_vec.T) + 1e-6 * np.eye(fr_vec.shape[1])
    cinv = np.linalg.inv(cov)
    md = np.einsum("ij,jk,ik->i", vec_te - mu, cinv, vec_te - mu)
    thr = np.quantile(np.einsum("ij,jk,ik->i", fr_vec - mu, cinv, fr_vec - mu), 0.99)
    print(f"  anomaly vector = [visual_score | CH2O | CO2 | Ethanol]  (gas idx {gas})")
    print(f"  Fresh-state mean mu = {mu}")
    print(f"  threshold (99% quantile of train-Fresh distances) = {thr:.2f}")
    for k, r in enumerate(rows):
        v = vec_te[demo[k]]
        d = md[demo[k]]
        print(f"  {CLASSES[y[te][demo[k]]][:13]:13s} vector={v}  distance={d:10.2f}  {'> THR' if d > thr else '<= THR'}")

    hdr("STAGE 9 | FINAL DECISION (masking->Formalin | score>=70->Fresh | else Rotten)")
    print(f"  {'truth':13s} {'index':>7s} {'dist':>9s} {'alert':>6s}  decision        correct?")
    n_ok = 0
    for k, r in enumerate(rows):
        s = s_all_te[demo[k]]
        d = md[demo[k]]
        masking = (s >= 70.0) and (senm[te][demo[k], 6] > 0.5) and (d > thr)
        pred = 2 if masking else (0 if s >= 70.0 else 1)
        ok = pred == y[te][demo[k]]
        n_ok += ok
        print(f"  {CLASSES[y[te][demo[k]]][:13]:13s} {s:7.2f} {d:9.2f} {str(masking):>6s}  {CLASSES[pred]:13s}  {'YES' if ok else 'NO'}")
    print(f"\n  demo samples correct: {n_ok}/3  (full test-set accuracy: run Cell 3 -> 98.9%)")


main()
