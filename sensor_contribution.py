"""Sensor-contribution analysis (thesis Cell 4):
  A. Ablation study  -> visual-only vs sensor-only vs fusion (same seed/split)
  B. Per-class sensor distributions -> fig7_sensor_features (300 DPI, PNG+PDF)

No retraining of MobileNetV2 needed: reuses fv_cache.npy + the same
preprocessing/split helpers from train_ladd_tf.py. Runs on
/workspace and /kaggle/working.
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
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

import train_ladd_tf as tl
tl.DATA = DATA

CLASSES = ["Fresh", "Rotten", "Formalin-Mixed"]
SENSORS = tl.SENSORS


def build_branch(inp_dim, wide):
    inp = layers.Input((inp_dim,), name="joint_features")
    if wide:
        h = layers.Dense(256, activation="relu")(inp)
        h = layers.Dropout(0.2)(h)
        h = layers.Dense(64, activation="relu")(h)
    else:
        h = layers.Dense(64, activation="relu")(inp)
    h = layers.Dense(64, activation="relu")(h)
    h = layers.Dropout(0.2)(h)
    out = layers.Dense(1, activation="sigmoid", name="freshness_raw")(h)
    idx = layers.Rescaling(100.0, name="freshness_index")(out)
    return keras.Model(inp, idx, name=f"branch_{inp_dim}")


def full_pipeline_acc(y, scores, senm, with_masking):
    if with_masking:
        gas = [4, 5, 6]
        vec = np.column_stack([scores, senm[:, gas]])
        fr_vec = vec[y == 0]
        mu, cov = fr_vec.mean(0), np.cov(fr_vec.T) + 1e-6 * np.eye(fr_vec.shape[1])
        cinv = np.linalg.inv(cov)
        md = np.einsum("ij,jk,ik->i", vec - mu, cinv, vec - mu)
        thr = np.quantile(md[y == 0], 0.99)
        masking = (scores >= 70.0) & (senm[:, 6] > 0.5) & (md > thr)
    else:
        masking = np.zeros(len(scores), bool)
    pred = np.where(masking, 2, np.where(scores >= 70.0, 0, 1)).astype(int)
    cm = np.zeros((3, 3), int)
    for c in range(3):
        for p in range(3):
            cm[c, p] = int(((y == c) & (pred == p)).sum())
    acc = cm.diagonal().sum() / cm.sum()
    recall = cm[2, 2] / cm[2].sum()
    false_alerts = int(cm[0, 2] + cm[1, 2])
    return acc, recall, false_alerts, cm


def main():
    df = tl.load_manifest()
    y = df["label"].to_numpy()
    gid = df["freshness_gt"].to_numpy()
    tr, tmp = train_test_split(np.arange(len(df)), test_size=0.30, stratify=y, random_state=42)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=42)

    fv = np.load(f"{BASE}/fv_cache.npy")
    if len(fv) != len(df):
        raise RuntimeError(f"fv_cache size {len(fv)} != manifest {len(df)}; rerun training cells first")
    sen_raw = tl.telemetry_preprocess(df)
    senm = tl.minmax_scale(sen_raw[tr], sen_raw)[1].astype(np.float32)

    x_fv = fv.astype(np.float32)
    x_sen = senm.astype(np.float32)
    x_fuse = np.hstack([fv, senm]).astype(np.float32)

    def run(name, xtr, xte, inp_dim, wide, with_masking):
        tf.random.set_seed(42)
        np.random.seed(42)
        model = build_branch(inp_dim, wide)
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
        model.fit(xtr, gid[tr], validation_split=0.0, epochs=60, batch_size=128, verbose=0)
        s = model.predict(xte, verbose=0).ravel()
        mae = np.abs(s - gid[te]).mean()
        fr2 = np.isin(y[te], [0, 1])
        acc2 = ((s >= 70) == (y[te] == 0))[fr2].mean()
        acc3, recall, fp, cm = full_pipeline_acc(y[te], s, senm[te], with_masking)
        print(f"  {name:12s} MAE={mae:5.2f}  Fresh-vs-Rotten={acc2:6.1%}  "
              f"Formalin recall={recall:6.1%}  3-class acc={acc3:6.1%}  false alerts={fp}")
        return mae, acc2, recall, acc3, cm

    print("[A] ABLATION STUDY (same split/seed, test set n=%d)" % len(te))
    res = {}
    res["sensor-only"] = run("sensor-only", x_sen[tr], x_sen[te], 7, False, True)
    res["visual-only"] = run("visual-only", x_fv[tr], x_fv[te], 1280, True, False)
    res["fusion (ours)"] = run("fusion (ours)", x_fuse[tr], x_fuse[te], 1287, False, True)

    print("\n  3-class confusion (fusion):")
    for r, n in enumerate(CLASSES):
        print(f"    {n:14s} {res['fusion (ours)'][4][r]}")

    print("\n[B] per-class sensor distributions figure ...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2))
    axes = axes.ravel()
    colors = ["#2e7d32", "#c62828", "#6a1b9a"]
    for j, feat in enumerate(SENSORS):
        ax = axes[j]
        vals = [df.loc[df["label"] == c, feat].to_numpy(float) for c in range(3)]
        bp = ax.boxplot(vals, tick_labels=["Fresh", "Rotten", "Formalin"], patch_artist=True,
                        widths=.55, medianprops=dict(color="black", lw=1.2),
                        flierprops=dict(marker=".", ms=2.5, alpha=.4))
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col); patch.set_alpha(.65)
        ax.set_title(feat, fontsize=10)
        ax.tick_params(axis="x", labelsize=7.5, rotation=30)
        ax.grid(axis="y", alpha=.3, ls=":")
    axes[7].axis("off")
    axes[7].text(.05, .78, "Sensor value ranges per class\n(raw LADD CSV columns)",
                 fontsize=11, va="top")
    axes[7].text(.05, .45,
                 "Key separations:\n"
                 "  Formaldehyde: Fresh/Rotten ~0.05 ppm\n"
                 "                  Formalin ~2-5 ppm\n"
                 "  Ethanol + CO2: Rotten explodes\n"
                 "  Moisture/Firmness: Fresh high",
                 fontsize=9, va="top", color="#333333")
    fig.suptitle("Telemetry (gas/physical) sensor distributions by class - LADD dataset", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .96])
    figdir = os.path.join(BASE, "thesis_figures")
    os.makedirs(figdir, exist_ok=True)
    for ext, dpi in (("png", 300), ("pdf", 300)):
        fig.savefig(os.path.join(figdir, f"fig7_sensor_features.{ext}"), dpi=dpi)
    plt.close(fig)
    print(f"  saved -> {figdir}/fig7_sensor_features.png/.pdf")

    lines = [
        "SENSOR-CONTRIBUTION ABLATION (test set, seed 42)",
        "variant        MAE   Fresh-vsRotten  FormalinRecall  3class  falseAlerts",
    ]
    for k, (mae, acc2, recall, acc3, cm) in res.items():
        lines.append(f"{k:14s} {mae:5.2f}   {acc2:6.1%}          {recall:6.1%}       {acc3:6.1%}   {cm[0,2]+cm[1,2]}")
    rep = os.path.join(BASE, "sensor_ablation_report.txt")
    open(rep, "w").write("\n".join(lines) + "\n")
    print(f"\nreport saved -> {rep}")


main()
