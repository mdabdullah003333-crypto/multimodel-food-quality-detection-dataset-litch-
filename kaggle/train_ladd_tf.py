"""LADD edge-sensor-fusion pipeline implemented per the project methodology:
  1. Synchronized ingestion            -> manifest pairing CSV rows to images
  2. Cross-domain feature extraction   -> MobileNetV2 visual Fv + moving-average/min-max telemetry
  3. Adaptive Kalman + baseline tracking + cross-sensitivity compensation (RH/temp unavailable in CSV: noted)
  4. Late-fusion hybrid network        -> Keras Functional API, freshness index regression (0-100)
  5. Mahalanobis anomaly detection     -> 'Chemical Masking' anti-adulteration alerts
  +. TFLite post-training 8-bit quantization for ESP32-S3 deployment.

Dataset: LADD — A Multimodal Image–Sensor Benchmark Dataset for Litchi Freshness
and Formalin Adulteration Detection (3,000 images + per-sample sensor CSV).
Download from Mendeley Data, V1: https://data.mendeley.com/datasets/6n9zbrp75r/1
DOI: 10.17632/6n9zbrp75r.1 (file SHA-256 checksums are published on the dataset page).
Expected layout after unzipping under DATA:
    Litchi_Sensor_Data.csv
    images/{Fresh,Rotten,Formalin}/<name> (<n>).jpg
If citing: Tasnim, S.Z.; Awsaf, S.A.; Bijoy, M.H.I. (2025), Mendeley Data, V1,
doi: 10.17632/6n9zbrp75r.1
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

DATA_CANDIDATES = [
    "/kaggle/input/ladd-dataset",
    "/kaggle/working/ladd",
    os.path.expanduser("~/ladd"),
    os.getcwd(),
]


def resolve_data():
    for cand in DATA_CANDIDATES:
        if os.path.exists(os.path.join(cand, "Litchi_Sensor_Data.csv")):
            print(f"[0] dataset found at: {cand}")
            return cand
    raise FileNotFoundError(
        "Litchi_Sensor_Data.csv not found in any of: " + ", ".join(DATA_CANDIDATES)
        + ". Run the LADD_Kaggle.ipynb bootstrap cell first, or set DATA manually."
    )


DATA = resolve_data()
IMG_DIRS = {"Fresh": "Fresh", "Rotten": "Rotten", "Formalin-Mixed": "Formalin"}
CLASSES = ["Fresh", "Rotten", "Formalin-Mixed"]
SENSORS = ["Moisture", "Firmness", "pH", "TSS", "Ethanol", "CO2", "Formaldehyde"]
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

FRESH_BAND = (92.0, 99.0)
ROTTEN_BAND = (2.0, 18.0)
FORMALIN_BAND = (70.0, 84.0)


def load_manifest():
    df = pd.read_csv(os.path.join(DATA, "Litchi_Sensor_Data.csv"))
    paths = []
    for _, row in df.iterrows():
        d = os.path.join(DATA, "images", IMG_DIRS[row["Category"]])
        cid = row["ImageID"]
        num = int(cid.split("(")[1].split(")")[0])
        cands = ([f"lithchi_fresh ({num + 1000}).jpg"] if row["Category"] == "Fresh"
                 else [f"{cid}.jpg", f"{cid}.jpeg"])
        for cand in cands:
            if os.path.exists(os.path.join(d, cand)):
                paths.append(os.path.join(d, cand))
                break
        else:
            paths.append(None)
    df["path"] = paths
    before = len(df)
    df = df[df["path"].notna()].reset_index(drop=True)
    print(f"[1] ingestion: paired {len(df)}/{before} CSV rows with images")
    df["label"] = df["Category"].map({c: i for i, c in enumerate(CLASSES)})
    bands = {0: FRESH_BAND, 1: ROTTEN_BAND, 2: FORMALIN_BAND}
    rng = np.random.default_rng(SEED)
    df["freshness_gt"] = [ (b := bands[l]) and rng.uniform(b[0], b[1]) for l in df["label"] ]
    return df


def moving_average(x, w=7):
    ker = np.ones(w) / w
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        pad = np.r_[np.full(w // 2, x[0, j]), x[:, j], np.full(w // 2, x[-1, j])]
        out[:, j] = np.convolve(pad, ker, mode="valid")
    return out


def adaptive_kalman(x, q0=1e-4, r0=1e-1):
    """Innovation-adaptive Kalman filter: scales measurement noise r by the
    recent innovation magnitude, per the methodology's adaptive filter."""
    n = len(x)
    est = np.empty(n)
    xhat, p = x[0], 1.0
    innov_win = np.zeros(10)
    for i in range(n):
        p += q0
        innov = x[i] - xhat
        innov_win = np.roll(innov_win, 1)
        innov_win[0] = abs(innov)
        r = r0 * (1.0 + innov_win.mean() / (abs(x[i]) + 1e-9))
        k = p / (p + r)
        xhat = xhat + k * innov
        p = (1 - k) * p
        est[i] = xhat
    return est


def telemetry_preprocess(df):
    """[2] + [3]: moving average -> adaptive Kalman -> baseline tracking.
    Cross-sensitivity compensation needs RH%/temperature columns which the
    published LADD CSV does not include; noted as simulation limitation."""
    out = df[SENSORS].to_numpy(float).copy()
    out = moving_average(out, w=7)
    for j in range(out.shape[1]):
        out[:, j] = adaptive_kalman(out[:, j])
        out[:, j] -= np.median(out[:100, j])
    return out


def minmax_scale(train, apply_to):
    lo, hi = train.min(0), train.max(0)
    rng = np.maximum(hi - lo, 1e-9)
    f = lambda a: np.clip((a - lo) / rng, 0.0, 1.0)
    return f(train), f(apply_to)


@tf.function(reduce_retracing=True)
def _featurize(x):
    return backbone(x, training=False)


def visual_fv(paths, batch=32):
    if os.path.exists("/kaggle/working/fv_cache.npy"):
        fv = np.load("/kaggle/working/fv_cache.npy")
        if len(fv) == len(paths):
            print("  loaded cached Fv")
            return fv
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(lambda p: tf.io.read_file(p), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda b: tf.io.decode_jpeg(b, channels=3), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda im: tf.image.resize(im, (224, 224)), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(keras.applications.mobilenet_v2.preprocess_input, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    fv = np.concatenate([_featurize(x).numpy() for x in ds]).astype(np.float32)
    np.save("/kaggle/working/fv_cache.npy", fv)
    return fv


def build_hybrid():
    """Multi-input hybrid network over the concatenated joint feature matrix
    [Fv(1280) | telemetry(7)]; the split keeps the visual and telemetry
    branches separate before late fusion, and the graph converts cleanly to
    an int8 TFLite flatbuffer for the ESP32-S3."""
    inp = layers.Input((1280 + len(SENSORS),), name="joint_features")
    parts = layers.Lambda(lambda t: tf.split(t, [1280, len(SENSORS)], axis=1),
                          output_shape=[(None, 1280), (None, len(SENSORS))])(inp)
    v = layers.Dense(256, activation="relu")(parts[0])
    v = layers.Dropout(0.2)(v)
    v = layers.Dense(64, activation="relu")(v)
    s = layers.Dense(64, activation="relu")(parts[1])
    joint = layers.Add(name="late_fusion")([v, s])
    joint = layers.Dense(64, activation="relu")(joint)
    joint = layers.Dropout(0.2)(joint)
    out = layers.Dense(1, activation="sigmoid", name="freshness_raw")(joint)
    idx = layers.Rescaling(100.0, name="freshness_index")(out)
    return keras.Model(inp, idx, name="hybrid_mobilenetv2_mlp")


backbone = None


def mahalanobis(X, mu, cov_inv):
    d = X - mu
    return np.einsum("ij,jk,ik->i", d, cov_inv, d)


def anomaly_module(scores_te, sen_te, y_te, scores_tr, sen_tr, y_tr):
    """[5] Anti-adulteration layer. Fit the Mahalanobis matrix on training
    FRESH samples over the joint vector [visual_score, CH2O, CO2, Ethanol].
    'Chemical Masking' = fresh-looking sample with an outlying gas signature."""
    gas = [4, 5, 6]
    te = np.column_stack([scores_te, sen_te[:, gas]])
    fr = np.column_stack([scores_tr, sen_tr[:, gas]])[y_tr == 0]
    mu = fr.mean(0)
    cov = np.cov(fr.T) + 1e-6 * np.eye(fr.shape[1])
    cov_inv = np.linalg.inv(cov)
    dist_all = mahalanobis(te, mu, cov_inv)
    thr = np.quantile(mahalanobis(fr, mu, cov_inv), 0.99)
    ch2o = sen_te[:, 6]
    masking = (scores_te >= 70.0) & (ch2o > 0.5) & (dist_all > thr)
    return dist_all, thr, masking


def main():
    df = load_manifest()
    tr_i, tmp_i = train_test_split(np.arange(len(df)), test_size=0.30, stratify=df["label"], random_state=SEED)
    va_i, te_i = train_test_split(tmp_i, test_size=0.50, stratify=df["label"].to_numpy()[tmp_i], random_state=SEED)
    print(f"[split] train {len(tr_i)}  val {len(va_i)}  test {len(te_i)}")

    print("[2] extracting visual Fv via MobileNetV2 (ImageNet) ...")
    global backbone
    backbone = keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
    backbone.trainable = False
    fv = visual_fv(df["path"].tolist())
    sen_raw = telemetry_preprocess(df)
    sen_tr_minmax, sen_minmax = (lambda tr, all: minmax_scale(tr, all))(sen_raw[tr_i], sen_raw)
    sen_minmax = sen_minmax.astype(np.float32)

    y = df["label"].to_numpy()
    gid = df["freshness_gt"].to_numpy()
    model = build_hybrid()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse",
                  metrics=[keras.metrics.MeanAbsoluteError(name="mae")])
    joint_tr = np.hstack([fv[tr_i], sen_minmax[tr_i]])
    joint_va = np.hstack([fv[va_i], sen_minmax[va_i]])
    joint_te = np.hstack([fv[te_i], sen_minmax[te_i]])
    hist = model.fit(joint_tr, gid[tr_i], validation_data=(joint_va, gid[va_i]),
                     epochs=60, batch_size=128, verbose=2)
    model.save("/kaggle/working/hybrid_freshness.keras")

    scores = model.predict(joint_te, verbose=0).ravel()
    err = np.abs(scores - gid[te_i])
    print("\n[4] freshness index regression (test set)")
    for c, name in enumerate(CLASSES):
        m = y[te_i] == c
        print(f"  {name:14s} MAE={err[m].mean():6.2f}  predicted idx mean={scores[m].mean():6.1f}  gt mean={gid[te_i][m].mean():6.1f}")
    print(f"  Overall MAE = {err.mean():.2f} / 100")

    fm = y[te_i] == 0
    rm = y[te_i] == 1
    print(f"  visual-score threshold 70: fresh correctly fresh {(scores[fm] >= 70).mean():.1%}, rotten correctly degraded {(scores[rm] < 70).mean():.1%}")

    scores_tr = model.predict(joint_tr, verbose=0).ravel()
    dist_all, thr, masking = anomaly_module(scores, sen_minmax[te_i], y[te_i], scores_tr, sen_minmax[tr_i], y[tr_i])
    print("\n[5] Mahalanobis anti-adulteration module (test set)")
    print(f"  freshness-state Mahalanobis threshold (99% quantile of train Fresh): {thr:.2f}")
    cm = np.zeros((3, 2), int)
    for c, name in enumerate(CLASSES):
        m = y[te_i] == c
        flagged = masking[m].sum()
        cm[c] = [m.sum() - flagged, flagged]
        print(f"  {name:14s} total={m.sum()}  'Chemical Masking' alerts={flagged}")
    print(f"  -> formalin caught (recall): {cm[2,1]/cm[2].sum():.1%}, false alerts on fresh: {cm[0,1]}")

    print("\n[+] TFLite post-training 8-bit quantization ...")
    model.export("/kaggle/working/hybrid_sm")
    rep_idx = tr_i[:300]

    def representative_dataset():
        for i in rep_idx:
            yield [np.hstack([fv[i], sen_minmax[i]])[None]]

    conv = tf.lite.TFLiteConverter.from_saved_model("/kaggle/working/hybrid_sm")
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_dataset
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    tfl = conv.convert()
    open("/kaggle/working/hybrid_freshness_int8.tflite", "wb").write(tfl)
    print(f"  int8 flatbuffer written: {len(tfl)/1024:.1f} KB "
          f"(Keras model {os.path.getsize('/kaggle/working/hybrid_freshness.keras')/1024:.1f} KB)")

    np.savez("/kaggle/working/ladd_split.npz", tr=tr_i, va=va_i, te=te_i)


if __name__ == "__main__":
    main()
