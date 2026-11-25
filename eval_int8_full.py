"""Full 3-class decision pipeline on the QUANTIZED int8 TFLite model.

Proves the proposal claim quantized accuracy (> 95 percent) end-to-end:
  int8 interpreter -> freshness index -> Mahalanobis 'Chemical Masking'
  -> combined decision -> 3-class accuracy / recall / precision.

Standalone: reuses the training artifacts (fv_cache.npy, int8 flatbuffer,
SavedModel) and train_ladd_tf.py preprocessing; runs both in the sandbox
(/workspace) and on Kaggle (/kaggle/working).
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", message="tf.lite.Interpreter is deprecated.*")


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
from sklearn.metrics import confusion_matrix

import train_ladd_tf as tl
tl.DATA = DATA

CLASSES = ["Fresh", "Rotten", "Formalin-Mixed"]


def int8_scores(joint):
    inter = tf.lite.Interpreter(f"{BASE}/hybrid_freshness_int8.tflite")
    inter.allocate_tensors()
    qi, qo = inter.get_input_details()[0], inter.get_output_details()[0]
    qscale, qzp = qi["quantization"]
    oscale, ozp = qo["quantization"]
    out = np.empty(len(joint), np.float32)
    t0 = time.perf_counter()
    for i, x in enumerate(joint):
        inter.set_tensor(qi["index"], np.clip(np.round(x[None] / qscale + qzp),
                                              -128, 127).astype(np.int8))
        inter.invoke()
        out[i] = ((inter.get_tensor(qo["index"]).astype(np.float32) - ozp) * oscale).ravel()[0]
    return out, (time.perf_counter() - t0) * 1e3 / len(joint)


def main():
    df = tl.load_manifest()
    y = df["label"].to_numpy()
    gid = df["freshness_gt"].to_numpy()
    tr, tmp = train_test_split(np.arange(len(df)), test_size=0.30,
                               stratify=y, random_state=42)
    va, te = train_test_split(tmp, test_size=0.50,
                              stratify=y[tmp], random_state=42)
    print(f"[split] train {len(tr)}  val {len(va)}  test {len(te)}")

    fv = np.load(f"{BASE}/fv_cache.npy")
    if len(fv) != len(df):
        raise RuntimeError(f"fv_cache size {len(fv)} != manifest {len(df)}; rerun training cell first")
    sen_raw = tl.telemetry_preprocess(df)
    senm = tl.minmax_scale(sen_raw[tr], sen_raw)[1].astype(np.float32)
    joint = np.hstack([fv, senm]).astype(np.float32)

    print("[1] running FULL int8 inference (interpreter, batch=1) ...")
    scores_i8, lat_ms = int8_scores(joint)
    s_tr, s_te = scores_i8[tr], scores_i8[te]

    print("[2] refitting Mahalanobis 'Chemical Masking' on int8 scores ...")
    gas = [4, 5, 6]
    fr = np.column_stack([s_tr, senm[tr][:, gas]])[y[tr] == 0]
    mu, cov = fr.mean(0), np.cov(fr.T) + 1e-6 * np.eye(fr.shape[1])
    cinv = np.linalg.inv(cov)
    md = lambda X: np.einsum("ij,jk,ik->i", X - mu, cinv, X - mu)
    thr = np.quantile(md(fr), 0.99)
    te_m = np.column_stack([s_te, senm[te][:, gas]])
    masking = (s_te >= 70.0) & (senm[te][:, 6] > 0.5) & (md(te_m) > thr)
    print(f"    anomaly threshold (99% quantile, int8) = {thr:.2f}")

    print("[3] combined decision rule (masking->Formalin, score>=70->Fresh, else Rotten)")
    pred = np.where(masking, 2, np.where(s_te >= 70.0, 0, 1)).astype(int)
    cm = confusion_matrix(y[te], pred, labels=[0, 1, 2])
    acc3 = cm.diagonal().sum() / cm.sum()
    fr_mask = np.isin(y[te], [0, 1])
    acc_frvrot = (pred[fr_mask] == y[te][fr_mask]).mean()
    recall = cm[2, 2] / cm[2].sum()
    prec = cm[2, 2] / max(cm[:, 2].sum(), 1)

    print("\n================ INT8 QUANTIZED ACCURACY REPORT ================")
    print("confusion matrix (rows=truth, cols=pred; order Fresh/Rotten/Formalin):")
    for r, name in enumerate(CLASSES):
        print(f"  {name:14s} {cm[r]}")
    print(f"\n  3-class accuracy (FULL-PIPELINE, int8): {cm.diagonal().sum()}/{cm.sum()} = {acc3:.1%}")
    for r, name in enumerate(CLASSES):
        print(f"  {name:14s} per-class accuracy: {cm[r, r] / cm[r].sum():.1%}")
    print(f"  Fresh-vs-Rotten accuracy (threshold 70): {acc_frvrot:.1%}")
    print(f"  Chemical Masking: recall {cm[2, 2]}/{cm[2].sum()} = {recall:.1%}, "
          f"precision {prec:.1%}, false alerts {cm[0, 2] + cm[1, 2]}")
    print(f"  int8 latency: {lat_ms:.2f} ms/sample (interpreter, CPU batch=1)")

    print("\n[4] float32 cross-check (SavedModel) ...")
    try:
        serve = tf.saved_model.load(f"{BASE}/hybrid_sm").signatures["serving_default"]
        s_te_f = np.asarray(serve(tf.constant(joint[te]))["output_0"]).ravel()
        agree = (np.abs(s_te_f - scores_i8[te]) <= 1.0).mean()
        print(f"    float32 vs int8 score agreement (|diff| <= 1 pt): {agree:.1%}")
        print(f"    mean |diff|: {np.abs(s_te_f - scores_i8[te]).mean():.3f}")
    except Exception as e:
        print("    (float32 cross-check skipped:", e, ")")

    ok = acc3 >= 0.95
    print("\n================ PROPOSAL CLAIM (>95% quantized accuracy) ================")
    print(f"  {'PROVEN' if ok else 'NOT PROVEN'}: int8 full-pipeline accuracy = {acc3:.1%}"
          f"  (recall {recall:.1%}, precision {prec:.1%}, false alerts {cm[0, 2] + cm[1, 2]})")

    lines = [
        "INT8 QUANTIZED FULL-PIPELINE EVALUATION",
        f"confusion matrix: {cm.tolist()}",
        f"3-class accuracy: {cm.diagonal().sum()}/{cm.sum()} = {acc3:.1%}",
        f"Fresh-vs-Rotten: {acc_frvrot:.1%}",
        f"masking recall {recall:.1%}, precision {prec:.1%}, false alerts {cm[0, 2] + cm[1, 2]}",
        f"int8 threshold {thr:.2f}, latency {lat_ms:.2f} ms/sample",
        f"claim >95% quantized accuracy: {'PROVEN' if ok else 'NOT PROVEN'}",
    ]
    rep = f"{BASE}/int8_accuracy_report.txt"
    open(rep, "w").write("\n".join(lines) + "\n")
    print(f"\nreport saved -> {rep}")


if __name__ == "__main__":
    main()
