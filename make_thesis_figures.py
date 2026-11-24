"""Generate 6 publication-grade thesis figures (300 DPI, PNG + PDF)."""
import os
import re
import warnings
warnings.filterwarnings("ignore", message="tf.lite.Interpreter is deprecated.*")
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
import importlib.util

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
    "legend.fontsize": 10, "font.family": "serif",
})

OUT = "/workspace/thesis_figures"
os.makedirs(OUT, exist_ok=True)
CLASSES = ["Fresh", "Rotten", "Formalin-Mixed"]
COLORS = {"Fresh": "#2e7d32", "Rotten": "#b71c1c", "Formalin-Mixed": "#6a1b9a"}

spec = importlib.util.spec_from_file_location("tl", "/workspace/train_ladd_tf.py")
tl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tl)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png")
    fig.savefig(f"{OUT}/{name}.pdf")
    plt.close(fig)
    print("saved", name)


def load_all():
    model = tf.saved_model.load("/workspace/hybrid_sm")
    serve = model.signatures["serving_default"]
    df = tl.load_manifest()
    y = df["label"].to_numpy()
    gid = df["freshness_gt"].to_numpy()
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=.30, stratify=y, random_state=42)
    va, te = train_test_split(tmp, test_size=.50, stratify=y[tmp], random_state=42)
    sen_raw = tl.telemetry_preprocess(df)
    senm = tl.minmax_scale(sen_raw[tr], sen_raw)[1].astype(np.float32)
    joint = lambda rows: np.hstack([np.load("/workspace/fv_cache.npy")[rows], senm[rows]]).astype(np.float32)
    predict = lambda rows: np.asarray(serve(tf.constant(joint(rows)))["output_0"]).ravel()
    inter, qi, qo = tflite_interpreter()
    int8_predict = lambda x: tflite_one(inter, qi, qo, x)
    return df, y, gid, tr, va, te, sen_raw, senm, predict, int8_predict


def tflite_interpreter():
    inter = tf.lite.Interpreter("/workspace/hybrid_freshness_int8.tflite")
    inter.allocate_tensors()
    return inter, inter.get_input_details()[0], inter.get_output_details()[0]


def tflite_one(inter, qi, qo, x1d):
    scale, zp = qi["quantization"]
    inter.set_tensor(qi["index"], np.clip(np.round(x1d[None] / scale + zp), -128, 127).astype(np.int8))
    inter.invoke()
    so, sz = qo["quantization"]
    return ((inter.get_tensor(qo["index"]).astype(np.float32) - sz) * so).ravel()[0]


def anomaly(saved, y_tr, sen_tr, scores_te, sen_te):
    gas = [4, 5, 6]
    fr = np.column_stack([saved, sen_tr[:, gas]])[y_tr == 0]
    mu, cov = fr.mean(0), np.cov(fr.T) + 1e-6 * np.eye(fr.shape[1])
    cinv = np.linalg.inv(cov)
    md = lambda X: np.einsum("ij,jk,ik->i", X - mu, cinv, X - mu)
    thr = np.quantile(md(fr), 0.99)
    te = np.column_stack([scores_te, sen_te[:, gas]])
    masking = (scores_te >= 70) & (sen_te[:, 6] > 0.5) & (md(te) > thr)
    return md, thr, masking


def fig1():
    hist = {"loss": [], "mae": [], "val_loss": [], "val_mae": []}
    for line in open("/workspace/tf_train_log.txt"):
        m = re.match(r"Epoch (\d+)/60\n.*?loss: ([\d.]+) - mae: ([\d.]+) - val_loss: ([\d.]+) - val_mae: ([\d.]+)", line + "\n", re.S)
        if not m:
            m = re.search(r"loss: ([\d.]+) - mae: ([\d.]+) - val_loss: ([\d.]+) - val_mae: ([\d.]+)", line)
            if m:
                hist["loss"].append(float(m.group(1))); hist["mae"].append(float(m.group(2)))
                hist["val_loss"].append(float(m.group(3))); hist["val_mae"].append(float(m.group(4)))
    ep = np.arange(1, len(hist["loss"]) + 1)
    fig, ax1 = plt.subplots(figsize=(6.2, 4.2))
    ax1.plot(ep, hist["loss"], color="#1565c0", lw=1.8, label="Train MSE")
    ax1.plot(ep, hist["val_loss"], color="#c62828", lw=1.8, ls="--", label="Val MSE")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("MSE loss")
    ax2 = ax1.twinx()
    ax2.plot(ep, hist["mae"], color="#2e7d32", lw=1.4, alpha=.85, label="Train MAE")
    ax2.plot(ep, hist["val_mae"], color="#ef6c00", lw=1.4, ls="--", alpha=.85, label="Val MAE")
    ax2.set_ylabel("MAE (freshness index pts)")
    l1, la1 = ax1.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, la1 + la2, loc="center right", framealpha=.9)
    ax1.set_title("Hybrid model training dynamics (60 epochs)")
    ax1.grid(alpha=.3)
    save(fig, "fig1_training_loss")


def fig2(df):
    raw = df[tl.SENSORS].to_numpy(float)[:1200]
    proc = tl.telemetry_preprocess(df)[:1200]
    feats = ["Ethanol", "CO2", "Formaldehyde"]
    j = [tl.SENSORS.index(f) for f in feats]
    fig, axes = plt.subplots(len(feats), 1, figsize=(7.0, 6.4), sharex=True)
    for ax, f, k in zip(axes, feats, j):
        ax.plot(raw[:, k], color="#9e9e9e", lw=.8, label="Raw telemetry")
        ax.plot(proc[:, k], color="#00695c", lw=1.6, label="Moving avg + adaptive Kalman + baseline")
        ax.set_ylabel(f)
        ax.grid(alpha=.3)
    axes[0].legend(loc="upper left", framealpha=.9)
    axes[-1].set_xlabel("Measurement sequence (t, dataset-instrument order)")
    fig.suptitle("Adaptive Kalman filtering and baseline tracking of gas telemetry", y=0.995)
    fig.tight_layout()
    save(fig, "fig2_kalman_filtering")


def fig3(y, gid, te, predict):
    s = predict(te)
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.plot([0, 100], [0, 100], "k--", lw=1.2, label="$y = x$ (ideal)")
    for c, name in enumerate(CLASSES):
        m = y[te] == c
        ax.scatter(gid[te][m], s[m], s=22, alpha=.75, color=COLORS[name],
                   edgecolors="white", linewidths=.3, label=name)
    err = np.abs(s - gid[te])
    ax.set_xlabel("Ground-truth freshness index"); ax.set_ylabel("Predicted freshness index")
    ax.set_xlim(0, 104); ax.set_ylim(-2, 104)
    ax.set_title(f"Predicted vs. ground-truth freshness index (test, n={len(te)})\noverall MAE = {err.mean():.2f}")
    ax.legend(framealpha=.9); ax.grid(alpha=.3)
    save(fig, "fig3_pred_vs_gt")


def fig4_real(y, te, senm, md_fn, thr, scores_te):
    d_f = md_fn(np.column_stack([scores_te, senm[te][:, [4, 5, 6]]])[y[te] == 0])
    d_fo = md_fn(np.column_stack([scores_te, senm[te][:, [4, 5, 6]]])[y[te] == 2])
    d_r = md_fn(np.column_stack([scores_te, senm[te][:, [4, 5, 6]]])[y[te] == 1])
    lg = lambda d: np.log10(np.maximum(d, 1e-3))
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    bins = np.linspace(-3, np.ceil(lg(max(d_r.max(), d_fo.max()))) + 0.5, 70)
    for d, name, alpha in [(d_f, "Fresh", 0.65), (d_fo, "Formalin-Mixed", 0.65), (d_r, "Rotten", 0.45)]:
        ax.hist(lg(d), bins=bins, alpha=alpha, color=COLORS[name], label=name, edgecolor="white")
    ax.axvline(np.log10(thr), color="black", lw=1.6, ls="--")
    ax.annotate(f"anomaly threshold\n$D_{{mah}}$ = {thr:.2f} (log\u2081\u2080 = {np.log10(thr):.2f})",
                xy=(np.log10(thr), ax.get_ylim()[1] * .75),
                xytext=(np.log10(thr) + 0.8, ax.get_ylim()[1] * .75), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=1.0))
    ax.set_xlabel("log\u2081\u2080 Mahalanobis distance to Fresh gas-state distribution")
    ax.set_ylabel("Test samples")
    ax.set_title("Mahalanobis distance distribution, Fresh vs. Formalin-Mixed (test)")
    ax.legend(framealpha=.9); ax.grid(alpha=.3)
    save(fig, "fig4_mahalanobis_threshold")
    return d_f, d_fo, d_r


def fig5(y, te, masking, scores_te):
    pred = np.where(masking, 2, np.where(scores_te >= 70, 0, 1))
    cm = np.zeros((3, 3), int)
    for tc in range(3):
        for pc in range(3):
            cm[tc, pc] = ((y[te] == tc) & (pred == pc)).sum()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
    for ax, mat, title, fmt in [
            (axes[0], cm, "Raw counts (test, n=448)", "d"),
            (axes[1], cm / cm.sum(1, keepdims=True), "Normalized (row-wise recall)", ".2%")]:
        im = ax.imshow(mat, cmap="Greens", vmin=0, vmax=1 if title.startswith("Norm") else None)
        for r in range(3):
            for c in range(3):
                ax.text(c, r, format(mat[r, c], fmt), ha="center", va="center",
                        fontsize=11, color="white" if mat[r, c] > .55 * (1 if title.startswith("Norm") else cm.max()) else "black")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(CLASSES, rotation=20, ha="right"); ax.set_yticklabels(CLASSES)
        ax.set_xlabel("Predicted (fusion + masking alert)"); ax.set_ylabel("True class")
        ax.set_title(title)
    fig.suptitle("Classification and Chemical-Masking alert performance", y=1.0)
    fig.tight_layout()
    save(fig, "fig5_confusion_matrix")
    return cm


def fig6(saved_scores, int8_scores, y_te):
    size_fp = os.path.getsize("/workspace/hybrid_freshness.keras") / 1e6
    size_int = os.path.getsize("/workspace/hybrid_freshness_int8.tflite") / 1e3
    diffs = np.abs(saved_scores - int8_scores)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
    bars = axes[0].bar(["Keras float32", "TFLite int8"], [size_fp, size_int / 1e3],
                       color=["#1565c0", "#2e7d32"], width=.55)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Artifact size (MB, log scale)")
    for b, v in zip(bars, [size_fp, size_int / 1e3]):
        axes[0].text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:.2f} MB" if v > 0.1 else f"{size_int:.1f} KB",
                     ha="center", fontsize=10)
    red = (1 - (size_int / 1e3) / size_fp) * 100
    axes[0].set_title(f"Model size: {red:.1f}% reduction")
    axes[0].grid(axis="y", alpha=.3)

    axes[1].scatter(saved_scores, int8_scores, s=22, alpha=.75, c="#6a1b9a", edgecolors="white", linewidths=.3)
    lims = [min(saved_scores.min(), int8_scores.min()) - 2, max(saved_scores.max(), int8_scores.max()) + 2]
    axes[1].plot(lims, lims, "k--", lw=1.2)
    axes[1].set_xlabel("Float32 freshness index"); axes[1].set_ylabel("Int8 freshness index")
    axes[1].set_title(f"Output fidelity: |diff| mean = {diffs.mean():.2f}, max = {diffs.max():.2f} (0\u2013100)")
    axes[1].grid(alpha=.3)
    fig.suptitle("Float32 vs. Int8 quantization impact", y=1.0)
    fig.tight_layout()
    save(fig, "fig6_quantization_impact")


def main():
    df, y, gid, tr, va, te, sen_raw, senm, predict, int8_one = load_all()
    print("computing float32 scores (train + test) ...")
    scores_te = predict(te)
    scores_tr = predict(tr)

    gas = [4, 5, 6]
    fr = np.column_stack([scores_tr, senm[tr][:, gas]])[y[tr] == 0]
    mu, cov = fr.mean(0), np.cov(fr.T) + 1e-6 * np.eye(fr.shape[1])
    cinv = np.linalg.inv(cov)
    md = lambda X: np.einsum("ij,jk,ik->i", X - mu, cinv, X - mu)
    thr = np.quantile(md(fr), 0.99)
    tte = np.column_stack([scores_te, senm[te][:, gas]])
    masking = (scores_te >= 70) & (senm[te][:, 6] > 0.5) & (md(tte) > thr)
    print(f"threshold = {thr:.2f}; masking alerts: Fresh {masking[y[te]==0].sum()}, "
          f"Rotten {masking[y[te]==1].sum()}, Formalin {masking[y[te]==2].sum()}")

    fig1()
    fig2(df)
    fig3(y, gid, te, predict)
    fig4_real(y, te, senm, md, thr, scores_te)
    cm = fig5(y, te, masking, scores_te)
    print("class matrix:", cm.tolist())

    print("computing int8 scores for fidelity figure ...")
    joint_te = np.hstack([np.load("/workspace/fv_cache.npy")[te], senm[te]]).astype(np.float32)
    int8_scores = np.array([int8_one(x) for x in joint_te[:150]])
    fig6(scores_te[:150], int8_scores, y[te][:150])
    print("done.")


if __name__ == "__main__":
    main()
