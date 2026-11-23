# LADD: Multimodal Litchi Freshness & Formalin-Adulteration Detection

Edge sensor-fusion AI pipeline (ESP32-S3 target) built on the **LADD** dataset —
multimodal image + sensor benchmark for litchi freshness and formalin
adulteration detection.

## Dataset

- **Source:** [Mendeley Data — LADD, V1](https://data.mendeley.com/datasets/6n9zbrp75r/1)
- **DOI:** 10.17632/6n9zbrp75r.1
- **Contents:** 3,000 RGB images (1000 Fresh / 1000 Rotten / 1000 Formalin-Mixed)
  + per-sample sensor measurements (Moisture, Firmness, pH, TSS, Ethanol, CO2, Formaldehyde)
- **Cite:** Tasnim, S.Z.; Awsaf, S.A.; Bijoy, M.H.I. (2025), "LADD: A Multimodal
  Image–Sensor Benchmark Dataset for Litchi Freshness and Formalin Adulteration
  Detection", Mendeley Data, V1, doi: 10.17632/6n9zbrp75r.1

### Download & setup

1. Download `LADD: ... .zip` (~131 MB) from the link above and verify the
   published SHA-256 checksum.
2. Unzip so the layout below exists (the zip's `Original_Images.zip` holds the
   three class folders):
   ```
   data/
   ├── Litchi_Sensor_Data.csv
   └── images/
       ├── Fresh/    lithchi_fresh (N).jpg ...
       ├── Rotten/   litchi_rotten (N).jpg ...
       └── Formalin/ litchi_formalin (N).jpg ...
   ```
3. Set `DATA` at the top of `train_ladd_tf.py` to that folder's path.

## Pipeline

1. **Ingestion** — pairs each CSV row to its image file (handles the dataset's
   naming quirks: Fresh images are numbered csv_num + 1000; 18 Rotten files are
   `.jpeg` duplicates of nums 1–18 → 2981/3000 usable pairs).
2. **Feature extraction** — MobileNetV2 (ImageNet) visual features `Fv`
   (1280-d) + moving-average / min-max telemetry preprocessing.
3. **Filtering** — innovation-adaptive Kalman filter + baseline tracking.
4. **Fusion** — hybrid two-branch network (visual Dense branch + telemetry
   Dense branch → late fusion) regressing a freshness index (0–100).
5. **Anomaly detection** — Mahalanobis-distance "Chemical Masking" detector.
6. **Deployment** — TFLite full-int8 post-training quantization
   (`hybrid_freshness_int8.tflite`, ~358 KB) for ESP32-S3 / TFLite Micro.

## Results (test set, n=448)

| Metric | Value |
|---|---|
| Freshness index MAE (overall) | 3.49 / 100 (Fresh 2.10, Rotten 4.18, Formalin 4.21) |
| Visual threshold 70 | Fresh 100% / Rotten 100% correct |
| Chemical-Masking recall (Formalin) | 96.0% — 0 false alerts on Fresh/Rotten |
| Model size | 4.24 MB → 0.37 MB int8 (91.6% ↓, output |diff| mean 0.17) |

Full run log: `tf_train_log.txt`. Figures (300 DPI, PNG + PDF): `thesis_figures/`.

## Reproduce

```bash
pip install tensorflow-cpu pandas scikit-learn numpy pillow matplotlib
python train_ladd_tf.py          # trains + evaluates + writes int8 tflite
python make_thesis_figures.py    # regenerates all 6 figures
```

Training takes ~10 min on CPU (feature extraction dominates; the fusion head
trains in seconds once `fv_cache.npy` exists). On a GPU-free machine, Kaggle /
Colab with a T4 is the fast route.
