"""Build the full work-documentation note (Word .docx) for the LADD pipeline session."""
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

FIGS = "/workspace/thesis_figures"
OUT = "/workspace/LADD_work_documentation.docx"

doc = Document()
st = doc.styles["Normal"]
st.font.name = "Calibri"
st.font.size = Pt(11)


def h(text, lvl=1):
    doc.add_heading(text, level=lvl)


def p(text, bold=False, italic=False, size=None):
    par = doc.add_paragraph()
    run = par.add_run(text)
    run.bold = bold
    run.italic = italic
    if size:
        run.font.size = Pt(size)
    return par


def bullet(text):
    doc.add_paragraph(text, style="List Bullet")


def numli(text):
    doc.add_paragraph(text, style="List Number")


def table(rows, header=True, widths=None):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Light Grid Accent 1"
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            c = t.cell(i, j)
            c.text = str(cell)
            for par in c.paragraphs:
                for r in par.runs:
                    r.font.size = Pt(10)
                    if header and i == 0:
                        r.bold = True
    return t


def figure(png, caption):
    doc.add_picture(png, width=Inches(6.2))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = p(caption, italic=True, size=9)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER


# ---------- Title ----------
title = doc.add_heading("LADD Multimodal Freshness & Formalin-Adulteration Detection", level=0)
sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = sub.add_run("Full Work Documentation Note — Data, Methodology, Training, Evaluation, Quantization & Repository Delivery")
r.italic = True
r.font.size = Pt(12)

# ---------- 1. Overview ----------
h("1. Objective and Scope", 1)
p("This note documents the complete working session that transformed the published LADD dataset "
  "into a trained, evaluated, quantized, deployment-ready multimodal AI pipeline, produced six "
  "publication-grade thesis figures, packaged all assets, and delivered the code and artifacts to "
  "GitHub (draft PR #1, branch vibe/ladd-thesis-figures-3afbd1).")
bullet("Goal 1 — implement the project methodology (5 operational layers) in software on the LADD dataset.")
bullet("Goal 2 — train and evaluate the multimodal (image + gas/thermal telemetry) model.")
bullet("Goal 3 — quantize the model to an ESP32-S3–deployable int8 TFLite artifact.")
bullet("Goal 4 — generate 6 thesis figures (300 DPI, PNG + PDF) and packaging zip.")
bullet("Goal 5 — push code, artifacts, figures and documentation to the connected GitHub repository.")

# ---------- 2. Dataset ----------
h("2. Dataset: LADD (Mendeley Data V1)", 1)
table([
    ["Item", "Details"],
    ["Name", "LADD — A Multimodal Image–Sensor Benchmark Dataset for Litchi Freshness and Formalin Adulteration Detection"],
    ["Source", "https://data.mendeley.com/datasets/6n9zbrp75r/1"],
    ["DOI", "10.17632/6n9zbrp75r.1"],
    ["Published", "15 October 2025 (Version 1)"],
    ["Authors", "Tasnim, S.Z.; Awsaf, S.A.; Bijoy, M.H.I. (Leading University & Daffodil International University)"],
    ["Size / integrity", "Original_Images.zip 136,921,337 B and Litchi_Sensor_Data.csv 342,450 B; both downloaded files matched the published SHA-256 checksums"],
    ["Classes", "Fresh (1000), Rotten (1000), Formalin-Mixed (1000)"],
    ["Image modality", "3000 RGB JPGs, ~1280x719, uniform lighting/background"],
    ["Sensor modality", "7 features per sample: Moisture (%), Firmness (kg/cm2), pH, TSS (Brix), Ethanol (ppm), CO2 (ppm), Formaldehyde (ppm)"],
])
p("Citation for the thesis bibliography:", bold=True)
p("Tasnim, S. Z., Awsaf, S. A., & Bijoy, M. H. I. (2025). LADD: A Multimodal Image–Sensor Benchmark "
  "Dataset for Litchi Freshness and Formalin Adulteration Detection. Mendeley Data, V1. "
  "https://doi.org/10.17632/6n9zbrp75r.1", italic=True)

# ---------- 3. Environment ----------
h("3. Software Environment", 1)
table([
    ["Component", "Version"],
    ["Python", "3.12"],
    ["TensorFlow / Keras", "2.21.0 (tensorflow-cpu)"],
    ["scikit-learn", "1.9.1"],
    ["pandas / numpy", "3.1.1 / 2.5.4"],
    ["Matplotlib", "3.11.2"],
    ["python-docx (this note)", "1.2.0"],
    ["Compute", "CPU-only sandbox (no GPU); PyTorch 2.15.0 was used for an earlier prototype run before the TensorFlow rebuild"],
])

# ---------- 4. Data ingestion ----------
h("4. Data Ingestion and Cleaning", 1)
p("The CSV references images by ImageID; three naming inconsistencies were discovered while pairing:")
numli("Fresh folder: files are numbered csv_num + 1000 (e.g., CSV row 'lithchi_fresh (1)' corresponds to 'lithchi_fresh (1001).jpg'); one stray 'lithchi_fresh (1).jpg' exists and is excluded.")
numli("Rotten folder: 18 files carry a '.jpeg' extension duplicating numbers 1–18 already present as '.jpg'; CSV rows 983–1000 have no files.")
numli("Formalin folder: matches the CSV exactly (1000/1000).")
p("Result: 2981/3000 CSV rows were confidently paired with image files; the 19 ambiguous/missing "
  "rows were dropped. Splits are stratified 70 / 15 / 15 (train 2086, val 447, test 448, seed 42).")

# ---------- 5. Methodology ----------
h("5. Methodology Implementation (5 Layers)", 1)
table([
    ["Layer (methodology)", "Implementation in this session"],
    ["1. Synchronized ingestion", "Manifest pairs each CSV row to its image file (section 4); the device-side t0 synchronization itself is firmware work, outside the model scope"],
    ["2. Cross-domain feature extraction", "Visual Fv: MobileNetV2 (ImageNet pretrained, frozen) global-averaged features, 1280-d, extracted at 224x224. Telemetry: temporal moving average (window 7) then min–max scaling fitted on the train split only"],
    ["3. Adaptive Kalman + baseline tracking", "Innovation-adaptive one-dimensional Kalman filter (measurement-noise r grows with the recent innovation magnitude) + baseline tracking via subtraction of the calibration-block median (first 100 samples/feature)"],
    ["4. Late fusion & DL inference", "Keras Functional hybrid network: split of the concatenated joint feature matrix [Fv(1280) | telemetry(7)] into visual branch (Dense 256 → 64) and telemetry branch (Dense 64), element-wise late fusion (Add), Dense 64, regression head sigmoid x 100 → unified freshness index (0–100)"],
    ["5. Anomaly detection (anti-adulteration)", "Mahalanobis distance over the joint vector [visual score, CH2O, CO2, Ethanol]; covariance fitted on training Fresh samples; threshold = 99% quantile of training Fresh distances = 19.52. 'Chemical Masking' alert fires when the sample looks fresh (score >= 70) AND CH2O > 0.5 ppm AND the Mahalanobis distance exceeds the threshold"],
])
p("Ground truth for regression: the dataset provides only 3-class labels, so a band-based freshness "
  "index proxy was defined — Fresh 92–99, Rotten 2–18, Formalin-Mixed 70–84 (uniform sampling, seed 42). "
  "Formalin-Mixed receives a high visual-freshness band by design: formalin preserves appearance while "
  "the gas signature reveals the adulteration — exactly the contradiction layer 5 must detect.")

# ---------- 6. Training ----------
h("6. Training Setup", 1)
bullet("Optimizer: Adam, lr = 1e-3; loss: MSE on the freshness index; metric: MAE.")
bullet("Epochs: 60, batch size 128; Dropout 0.2 in both branches; cosine-style convergence observed.")
bullet("Reproducibility: fixed seed 42 for numpy / TensorFlow and for all splits.")
bullet("Runtime: ~10 min on CPU; dominant cost is the one-time MobileNetV2 feature extraction over 2981 images (cached to fv_cache.npy thereafter).")
figure(f"{FIGS}/fig1_training_loss.png", "Figure 1. Training and validation dynamics (MSE left axis, MAE right axis) over 60 epochs.")

# ---------- 7. Results ----------
h("7. Results", 1)
h("7.1 Freshness-Index Regression (test set, n = 448)", 2)
table([
    ["Class", "MAE", "Predicted mean", "Ground-truth mean"],
    ["Fresh", "2.10", "96.2", "95.4"],
    ["Rotten", "4.18", "9.6", "10.0"],
    ["Formalin-Mixed", "4.21", "77.4", "76.8"],
    ["Overall", "3.49", "—", "—"],
])
p("Visual-score threshold 70: Fresh classified fresh 100%, Rotten classified degraded 100%.")
figure(f"{FIGS}/fig3_pred_vs_gt.png", "Figure 2. Predicted vs ground-truth freshness index with ideal y = x reference.")

h("7.2 Mahalanobis Anti-Adulteration Module", 2)
table([
    ["True class (test)", "Samples", "Chemical-Masking alerts"],
    ["Fresh", "150", "0"],
    ["Rotten", "148", "0"],
    ["Formalin-Mixed", "150", "144 (96.0% recall)"],
])
figure(f"{FIGS}/fig4_mahalanobis_threshold.png", "Figure 3. Mahalanobis distance distributions (log10 scale) with the anomaly threshold (D = 19.52).")

h("7.3 Classification + Masking Performance", 2)
p("Combined decision rule (masking alert → Formalin; else score >= 70 → Fresh; else Rotten) yields the "
  "confusion matrix [Fresh: 150/150 correct] [Rotten: 148/148 correct] [Formalin-Mixed: 144/150 correct]; "
  "overall accuracy ≈ 98.9% on the test set.")
figure(f"{FIGS}/fig5_confusion_matrix.png", "Figure 4. Confusion matrix with raw counts (left) and row-normalized recall (right).")

h("7.4 Kalman Filtering Effect", 2)
figure(f"{FIGS}/fig2_kalman_filtering.png", "Figure 5. Raw telemetry vs moving-average + adaptive-Kalman + baseline-tracked signals (Ethanol, CO2, Formaldehyde).")

# ---------- 8. Quantization ----------
h("8. Quantization and Edge Deployment (TFLite, ESP32-S3)", 1)
p("Post-training quantization (PTQ) with a 300-sample representative dataset, full integer int8 "
  "weights and activations, int8 input/output — ready for TensorFlow Lite Micro + ESP-NN on the ESP32-S3.")
table([
    ["Metric", "Keras float32", "TFLite int8"],
    ["Artifact size", "4.24 MB", "357.9 KB (91.6% smaller)"],
    ["Output |diff| vs float32", "—", "mean 0.17, max ~0.5–0.8 (0–100 index)"],
])
figure(f"{FIGS}/fig6_quantization_impact.png", "Figure 6. Model-size reduction and int8-vs-float32 output fidelity.")

# ---------- 9. Deliverables ----------
h("9. Deliverables and Packaging", 1)
table([
    ["Deliverable", "Description"],
    ["train_ladd_tf.py", "Full pipeline: ingestion → features → filtering → training → evaluation → quantization"],
    ["make_thesis_figures.py", "Regenerates all 6 thesis figures"],
    ["hybrid_freshness.keras", "Trained float32 fusion head (4.24 MB)"],
    ["hybrid_freshness_int8.tflite", "Deployable int8 flatbuffer (357.9 KB)"],
    ["tf_train_log.txt", "Complete training + evaluation log (reproduction evidence)"],
    ["thesis_figures/ (12 files)", "6 figures x (PNG + PDF) at 300 DPI"],
    ["thesis_assets.zip", "Bundle of code, log, model artifact and figures (sandbox)"],
    ["GitHub draft PR #1", "https://github.com/mdabdullah003333-crypto/multimodel-food-quality-detection-dataset-litch-/pull/1 (branch vibe/ladd-thesis-figures-3afbd1)"],
])

# ---------- 10. Limitations ----------
h("10. Documented Limitations (State Clearly in the Thesis)", 1)
numli("BME680 RH/temperature cross-sensitivity compensation (Layer 3 in the methodology) could not be exercised: the published LADD CSV contains no RH or temperature columns. It should be reported as 'simulated on available channels'.")
numli("The freshness-index ground truth is a synthetic band-based proxy derived from the 3 class labels, not a measured 0–100 sensory score.")
numli("The dataset's uniform lighting/background inflates visual separability; field (open-market) accuracy will be lower, and the anomaly module is a heuristic rule layered on a threshold score.")
numli("The dataset's own CSV↔image numbering inconsistencies (19 unresolved rows) should be reported to the dataset authors.")
numli("Earlier prototype numbers (PyTorch: image-only 99.78%, Mahalanobis 100%) came from a different random state and legacy preprocessing; the TensorFlow pipeline numbers are the canonical set.")

# ---------- 11. Next steps ----------
h("11. Recommended Next Steps", 1)
numli("Merge draft PR #1 after review; then optionally run the pipeline in Kaggle (T4 GPU) to cross-check reproducibility.")
numli("Analyze the 6 missed Formalin samples (borderline CH2O) and consider ROC tuning of the masking thresholds instead of fixed rules.")
numli("Add firmware-side ESP-IDF integration: int8 TFLite inference with ESP-NN, BLE telemetry to the Flutter app.")
numli("Field protocol: collect a small out-of-distribution validation set (market conditions) to honestly measure degradation.")
numli("Contact the LADD authors about the naming inconsistencies; a V2 would strengthen the benchmark.")

# ---------- Footer ----------
doc.add_paragraph()
p("Prepared automatically during the working session; all numeric values are taken from the committed "
  "training log and figure-generation script. Repository: "
  "github.com/mdabdullah003333-crypto/multimodel-food-quality-detection-dataset-litch-", italic=True, size=9)

doc.save(OUT)
print("saved", OUT)
