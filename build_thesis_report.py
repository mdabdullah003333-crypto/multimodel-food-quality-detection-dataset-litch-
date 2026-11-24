"""Build the full DIU-format thesis report (Word .docx) for the
multimodal (image + gas/thermal telemetry) litchi freshness and
formalin-adulteration edge-AI pipeline, embedding the six trained
result figures produced by make_thesis_figures.py.

Formatting follows the DIU thesis guidelines:
- Times New Roman 12 pt (body), double line spacing
- margins: left 4.0 cm, top/bottom/right 2.5 cm
- page numbers 1 cm from bottom center (roman for preliminary, arabic otherwise)
- headings numbered 1.1 / 1.2 / 1.1.1 from the left margin
- tables numbered per chapter with caption ABOVE the table, single-spaced
- figures numbered per chapter with caption BELOW the figure
"""
import re

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Cm, Pt, RGBColor

FIGS = "/workspace/thesis_figures"
OUT = "/workspace/Thesis_Report_Multimodal_Food_Quality_Detection.docx"

FONT = "Times New Roman"

doc = Document()

# ---------------------------------------------------------------- styles
normal = doc.styles["Normal"]
normal.font.name = FONT
normal.font.size = Pt(12)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
normal.paragraph_format.line_spacing = 2.0       # double spacing (guideline)
normal.paragraph_format.space_after = Pt(0)
normal.paragraph_format.space_before = Pt(0)

for name, size in (("Heading 1", 14), ("Heading 2", 12), ("Heading 3", 12)):
    st = doc.styles[name]
    st.font.name = FONT
    st.font.size = Pt(size)
    st.font.bold = True
    st.font.color.rgb = RGBColor(0, 0, 0)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    st.paragraph_format.line_spacing = 2.0
    st.paragraph_format.space_before = Pt(12)
    st.paragraph_format.space_after = Pt(6)

# ---------------------------------------------------------------- pages
A4_W, A4_H = Cm(21.0), Cm(29.7)


def setup_section(sec):
    sec.page_width, sec.page_height = A4_W, A4_H
    sec.left_margin = Cm(4.0)          # guideline: 4.0 cm
    sec.top_margin = Cm(2.5)
    sec.bottom_margin = Cm(2.5)
    sec.right_margin = Cm(2.5)
    sec.footer_distance = Cm(1.0)      # page number 1 cm from bottom


def page_number(sec, fmt=None, start=None, show=True):
    """Attach a centered bottom page-number footer (fldSimple PAGE)."""
    sectPr = sec._sectPr
    if show:
        footer = sec.footer
        footer.is_linked_to_previous = False
        par = footer.paragraphs[0]
        par.text = ""
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.line_spacing = 1.0
        fld = OxmlElement("w:fldSimple")
        fld.set(qn("w:instr"), "PAGE")
        r = OxmlElement("w:r")
        rPr = OxmlElement("w:rPr")
        f = OxmlElement("w:rFonts")
        f.set(qn("w:ascii"), FONT)
        f.set(qn("w:hAnsi"), FONT)
        sz = OxmlElement("w:sz")
        sz.set(qn("w:val"), "24")
        rPr.append(f)
        rPr.append(sz)
        r.append(rPr)
        t = OxmlElement("w:t")
        t.text = "1"
        r.append(t)
        fld.append(r)
        par._p.append(fld)
    else:
        sec.footer.is_linked_to_previous = False
        sec.footer.paragraphs[0].text = ""
    for el in sectPr.findall(qn("w:pgNumType")):
        sectPr.remove(el)
    pgNum = OxmlElement("w:pgNumType")
    if fmt:
        pgNum.set(qn("w:fmt"), fmt)
    if start is not None:
        pgNum.set(qn("w:start"), str(start))
    sectPr.append(pgNum)


def new_section():
    sec = doc.add_section(WD_SECTION.NEW_PAGE)
    setup_section(sec)
    return sec


sec0 = doc.sections[0]
setup_section(sec0)

# ---------------------------------------------------------------- helpers
TOKEN = re.compile(r"(\*\*.+?\*\*|\*.+?\*|⟦.+?⟧|⟪.+?⟫)")


def rich(par, text, size=12, bold=False, italic=False, font=FONT):
    """Add runs honoring **bold**, *italic*, subscript and superscript."""
    for chunk in TOKEN.split(text):
        b, i, sub, sup = bold, italic, False, False
        t = chunk
        if chunk.startswith("**") and chunk.endswith("**"):
            t, b = chunk[2:-2], True
        elif chunk.startswith("*") and chunk.endswith("*"):
            t, i = chunk[1:-1], True
        elif chunk.startswith("⟦") and chunk.endswith("⟧"):
            t, sub = chunk[1:-1], True
        elif chunk.startswith("⟪") and chunk.endswith("⟫"):
            t, sup = chunk[1:-1], True
        r = par.add_run(t)
        r.font.name = font
        r.font.size = Pt(size)
        r.bold = b
        r.italic = i
        if sub:
            r.font.subscript = True
        if sup:
            r.font.superscript = True
    return par


def P(text="", size=12, bold=False, italic=False, single=False, align=None,
      before=0, after=0, indent=None):
    par = doc.add_paragraph()
    rich(par, text, size=size, bold=bold, italic=italic)
    if single:
        par.paragraph_format.line_spacing = 1.0
    if align is not None:
        par.alignment = align
    par.paragraph_format.space_before = Pt(before)
    par.paragraph_format.space_after = Pt(after)
    if indent is not None:
        par.paragraph_format.left_indent = Cm(indent)
    return par


def H(text, lvl=1):
    h = doc.add_heading("", level=lvl)
    rich(h, text, size=14 if lvl == 1 else 12, bold=True)
    h.paragraph_format.keep_with_next = True
    return h


def chapter(ch_no, ch_title, start=None):
    sec = new_section()
    page_number(sec, start=start)
    p1 = doc.add_heading("", level=1)
    rich(p1, f"CHAPTER {ch_no}", size=14, bold=True)
    p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p1.paragraph_format.line_spacing = 2.0
    p2 = doc.add_heading("", level=1)
    rich(p2, ch_title, size=14, bold=True)
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p2.paragraph_format.line_spacing = 2.0
    p2.paragraph_format.space_after = Pt(12)


def eq(number, parts):
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.line_spacing = 1.0
    par.paragraph_format.space_before = Pt(6)
    par.paragraph_format.space_after = Pt(6)
    for kind, text in parts:
        r = par.add_run(text)
        r.font.name = FONT
        r.font.size = Pt(12)
        r.italic = kind == "i"
        r.font.subscript = kind == "s"
        r.font.superscript = kind == "p"
    r = par.add_run("          " + f"({number})")
    r.font.name = FONT
    r.font.size = Pt(12)
    return par


def table(rows, header=True, caption=None, source=None, size=10.5, widths=None):
    if caption:
        cap = P(caption, size=12, bold=True, single=True, after=4)
        cap.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    t.alignment = 1  # center
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            c = t.cell(i, j)
            c.text = ""
            par = c.paragraphs[0]
            rich(par, str(cell), size=size, bold=header and i == 0)
            par.paragraph_format.line_spacing = 1.0     # single-spaced tables
            par.paragraph_format.space_before = Pt(1)
            par.paragraph_format.space_after = Pt(1)
    if widths:
        for j, w in enumerate(widths):
            for row in t.rows:
                row.cells[j].width = Cm(w)
    if source:
        P(source, size=10, italic=True, single=True, before=4, after=8)
    else:
        P(after=8)
    return t


def figure(png, caption, width_cm=13.5):
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.line_spacing = 1.0
    par.add_run().add_picture(png, width=Cm(width_cm))
    cap = P(caption, size=12, single=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=12)
    cap.paragraph_format.keep_together = True


def bullets(items):
    for it in items:
        par = doc.add_paragraph(style="List Bullet")
        rich(par, it)
        par.paragraph_format.line_spacing = 2.0
        par.paragraph_format.space_after = Pt(0)


def numbers(items):
    for it in items:
        par = doc.add_paragraph(style="List Number")
        rich(par, it)
        par.paragraph_format.line_spacing = 2.0
        par.paragraph_format.space_after = Pt(0)


C = WD_ALIGN_PARAGRAPH.CENTER

# =========================================================== COVER (page i)
for _ in range(2):
    P()
P("MULTIMODAL FOOD QUALITY DETECTION AND FORMALIN ADULTERATION ALERTING "
  "USING EDGE SENSOR-FUSION ARTIFICIAL INTELLIGENCE", bold=True, align=C)
P()
P("[Student's Full Name]", bold=True, align=C)
P("[Second Student's Full Name]", bold=True, align=C)
P()
P("[Student ID(s)]", align=C)
P()
P("Bachelor of Science", align=C)
P()
P("Department of Software Engineering", align=C)
P()
P("DAFFODIL INTERNATIONAL UNIVERSITY", align=C)
P("[Month] [Year]", align=C)

# ================================================ DECLARATION (roman from ii)
sec = new_section()
page_number(sec, fmt="lowerRoman", start=2)

H("DECLARATION OF THESIS AND COPYRIGHT", 1)
P("Author's Full Name : [Student's Full Name]", single=True, after=6)
P("Date of Birth : [DD/MM/YYYY]", single=True, after=6)
P("Title : Multimodal Food Quality Detection and Formalin Adulteration Alerting "
  "Using Edge Sensor-Fusion Artificial Intelligence", single=True, after=6)
P("Academic Session : [Session]", single=True, after=12)
P("I declare that this thesis is classified as:")
P("CONFIDENTIAL (Contains confidential information under the Official Secret Act "
  "1997)*", single=True)
P("RESTRICTED (Contains restricted information as specified by the organization "
  "where research was done)*", single=True)
P("OPEN ACCESS I agree that my thesis to be published as online open access (Full "
  "Text)", single=True)
P("I acknowledge that Daffodil International University reserves the following "
  "rights:")
numbers([
    "The Thesis is the Property of Daffodil International University.",
    "The Library of Daffodil International University has the right to make copies "
    "of the thesis for the purpose of research only.",
    "The Library of Daffodil International University has the right to make copies "
    "of the thesis for academic exchange.",
])
P()
P("Certified by:")
P()
P("(Student's Signature)", single=True, after=2)
P("Student ID : [ID]", single=True)
P("Date :", single=True)
P()
P("(Supervisor's Signature)", single=True, after=2)
P("Name of Supervisor : [Supervisor's Full Name]", single=True)
P("Date :", single=True)
P("NOTE: * If the thesis is CONFIDENTIAL or RESTRICTED, please attach a thesis "
  "declaration letter.", size=10, single=True)

H("SUPERVISOR'S DECLARATION", 1)
P("I/We hereby declare that I/We have checked this thesis and in my/our opinion, "
  "this thesis is adequate in terms of scope and quality for the award of the "
  "degree of Bachelor of Science.")
P()
P("(Supervisor's Signature)", single=True, after=2)
P("Full Name : [Supervisor's Full Name]", single=True)
P("Position : [Designation, Department]", single=True)
P("Date :", single=True)

H("STUDENT'S DECLARATION", 1)
P("I hereby declare that the work in this thesis is based on my original work "
  "except for quotations and citations which have been duly acknowledged. I also "
  "declare that it has not been previously or concurrently submitted for any other "
  "degree at Daffodil International University or any other institution.")
P()
P("(Student's Signature)", single=True, after=2)
P("Full Name : [Student's Full Name]", single=True)
P("ID Number : [ID]", single=True)
P("Date :", single=True)

# ------------------------------------------------------------ inner title page
for _ in range(3):
    P()
P("MULTIMODAL FOOD QUALITY DETECTION AND FORMALIN ADULTERATION ALERTING "
  "USING EDGE SENSOR-FUSION ARTIFICIAL INTELLIGENCE", bold=True, align=C, single=True)
P()
P("[Student's Full Name]", bold=True, align=C)
P()
P("Thesis submitted in fulfillment of the requirements for the award of the "
  "degree of Bachelor of Science", align=C, single=True)
P()
P("Department of Software Engineering", align=C)
P("DAFFODIL INTERNATIONAL UNIVERSITY", align=C)
P("[MONTH YEAR]", align=C)

# ------------------------------------------------------------ ACKNOWLEDGEMENTS
H("ACKNOWLEDGEMENTS", 1)
P("First of all, I would like to express my deepest gratitude to Almighty for "
  "giving me the strength and patience to complete this work. I would like to "
  "convey my sincere thanks to my supervisor, [Supervisor's Full Name], for the "
  "continuous guidance, constructive criticism, and encouragement throughout this "
  "thesis.")
P("I am grateful to the Department of Software Engineering, Daffodil International "
  "University, for providing the academic environment and infrastructure required "
  "for this research. I also thank the authors of the LADD dataset — Tasnim, Awsaf, "
  "and Bijoy — for publicly publishing their multimodal image–sensor benchmark, "
  "which forms the empirical foundation of this work.")
P("Finally, I would like to thank my family and my friends for their unconditional "
  "support and motivation during the long period of research, experimentation, and "
  "writing.")

# ---------------------------------------------------------------- DEDICATION
H("DEDICATION", 1)
P("This thesis is dedicated to my beloved parents, whose encouragement and "
  "sacrifice made this journey possible.")

# ---------------------------------------------------------------- ABSTRACT
H("ABSTRACT", 1)
P("Food adulteration with formalin solution is a serious public-health threat in "
  "open food markets, because formaldehyde preserves the visual appearance of "
  "decaying fruit and thereby defeats purely vision-based quality inspection. This "
  "thesis presents the design and evaluation of a multimodal, edge-oriented "
  "food-quality detection pipeline that estimates litchi freshness and alerts "
  "formalin adulteration simultaneously, using the fusion of camera imagery and "
  "gas/thermal telemetry — the two information channels that an ESP32-S3-class "
  "embedded device supplies in the field. The work follows a five-layer "
  "methodology: synchronized data ingestion; cross-domain feature extraction "
  "(MobileNetV2 visual features and moving-average/min-max telemetry "
  "preprocessing); adaptive Kalman filtering with baseline tracking; a late-fusion "
  "hybrid neural network that regresses a unified freshness index (0–100); and a "
  "Mahalanobis distance-based anomaly module, Chemical Masking, which exposes the "
  "visual–chemical contradiction that formalin creates. Experiments on the "
  "publicly available LADD benchmark (2,981 paired image–sensor samples; Fresh, "
  "Rotten, and Formalin-Mixed) achieve an overall freshness-index MAE of 3.49/100 "
  "in the sandbox and 3.40–3.46/100 in two independent Kaggle GPU runs, 100 "
  "percent Fresh-versus-Rotten discrimination, Chemical-Masking recall of "
  "95.3–96.7 percent with 100 percent precision and zero false alerts, and "
  "full-pipeline accuracy of up to 98.9 percent. Post-training 8-bit quantization "
  "compresses the model from 4.24 MB to a 357.9 KB full-integer TFLite flatbuffer "
  "with a negligible mean output difference of 0.17 (0–100 index), making the "
  "pipeline deployable on the ESP32-S3 microcontroller with TensorFlow Lite Micro. "
  "The results demonstrate that low-cost multimodal sensor fusion can both "
  "estimate freshness and defeat appearance-based adulteration entirely at the "
  "edge.")
P()
P("Keywords: food safety; formalin adulteration; litchi; multimodal sensor "
  "fusion; edge AI; ESP32-S3; TensorFlow Lite Micro; Mahalanobis distance; "
  "adaptive Kalman filter.", single=True, italic=True)

# ---------------------------------------------------------------- TOC
H("TABLE OF CONTENT", 1)
toc_par = doc.add_paragraph()
toc_par.paragraph_format.line_spacing = 1.0
fld = OxmlElement("w:fldSimple")
fld.set(qn("w:instr"), r'TOC \o "1-3" \h \z \u')
run = OxmlElement("w:r")
t_el = OxmlElement("w:t")
t_el.text = ("Right-click here and choose 'Update Field' in Microsoft Word to "
             "generate the table of contents with page numbers.")
run.append(t_el)
fld.append(run)
toc_par._p.append(fld)

# ---------------------------------------------------------------- LISTS
H("LIST OF TABLES", 1)
table([
    ["Table", "Title", "Page"],
    ["Table 2.1", "Comparison of reviewed approaches with the proposed pipeline", "[pg]"],
    ["Table 3.1", "LADD dataset composition", "[pg]"],
    ["Table 3.2", "Sensor channels of the LADD benchmark", "[pg]"],
    ["Table 3.3", "Network configuration", "[pg]"],
    ["Table 3.4", "Training hyperparameters and configuration", "[pg]"],
    ["Table 4.1", "Freshness-index regression results (test set, n = 448)", "[pg]"],
    ["Table 4.2", "Chemical-Masking module results (test set, n = 448)", "[pg]"],
    ["Table 4.3", "Quantization impact summary", "[pg]"],
    ["Table 4.4", "Cross-platform reproducibility of the pipeline", "[pg]"],
])

H("LIST OF FIGURES", 1)
table([
    ["Figure", "Title", "Page"],
    ["Figure 3.1", "Adaptive Kalman filtering and baseline tracking of gas telemetry", "[pg]"],
    ["Figure 3.2", "Training and validation dynamics of the fusion network", "[pg]"],
    ["Figure 4.1", "Predicted versus ground-truth freshness index", "[pg]"],
    ["Figure 4.2", "Mahalanobis distance distributions and anomaly threshold", "[pg]"],
    ["Figure 4.3", "Confusion matrix with raw counts and row-normalized recall", "[pg]"],
    ["Figure 4.4", "Quantization impact: model size and output fidelity", "[pg]"],
])

H("LIST OF SYMBOLS", 1)
table([
    ["Symbol", "Description"],
    ["Fv", "Visual feature vector extracted by the CNN backbone (1280-d)"],
    ["w", "Moving-average window width (samples)"],
    ["x-hat", "Kalman filter state estimate"],
    ["q", "Kalman process-variance parameter"],
    ["r", "Kalman measurement-variance parameter"],
    ["K", "Kalman gain"],
    ["gamma-k", "Kalman innovation (measurement residual)"],
    ["eps", "Small constant used for numerical stability"],
    ["mu", "Mean vector of the Fresh reference distribution"],
    ["Sigma", "Covariance matrix of the Fresh reference distribution"],
    ["D2", "Mahalanobis distance (squared)"],
    ["thr", "Anomaly threshold (99th percentile of train-Fresh distances)"],
    ["MAE", "Mean Absolute Error on the 0–100 freshness index"],
    ["MSE", "Mean Squared Error (training loss)"],
])

H("LIST OF ABBREVIATIONS", 1)
table([
    ["Abbreviation", "Full form"],
    ["LADD", "Multimodal Litchi Image–Sensor benchmark dataset (this work's dataset)"],
    ["CNN", "Convolutional Neural Network"],
    ["MCU", "Microcontroller Unit"],
    ["TFLite", "TensorFlow Lite"],
    ["PTQ", "Post-Training Quantization"],
    ["MLP", "Multi-Layer Perceptron"],
    ["BLE", "Bluetooth Low Energy"],
    ["UART", "Universal Asynchronous Receiver-Transmitter"],
    ["I2C", "Inter-Integrated Circuit bus"],
    ["CH2O", "Formaldehyde"],
    ["CO2", "Carbon dioxide"],
    ["VOC", "Volatile Organic Compounds"],
    ["RH", "Relative Humidity"],
    ["TSS", "Total Soluble Solids"],
    ["API", "Application Programming Interface"],
    ["DOI", "Digital Object Identifier"],
])

H("LIST OF APPENDICES", 1)
table([
    ["Appendix", "Title", "Page"],
    ["Appendix A", "Reproduction instructions for the training pipeline", "[pg]"],
    ["Appendix B", "Kaggle (GPU T4) reproduction output summary", "[pg]"],
])

# =========================================================== CHAPTER 1
chapter(1, "INTRODUCTION", start=1)

H("1.1 Background", 2)
P("Food safety remains one of the most pressing public-health challenges in "
  "developing countries, where a large fraction of the daily food trade passes "
  "through crowded open markets. In such environments fruit freshness is judged "
  "almost exclusively by human visual inspection, and sellers face an economic "
  "incentive to disguise decayed produce. The most dangerous widespread practice is "
  "treatment of over-ripened or rotten fruit with formalin — an aqueous solution of "
  "formaldehyde — which hardens tissue and restores a fresh-looking appearance for "
  "days while introducing a toxic, carcinogenic chemical into the food chain.")
P("Purely visual inspection cannot defeat this practice, because formalin "
  "specifically masks the visual symptoms of degradation. Conversely, purely "
  "chemical analysis such as chromatography is laboratory-bound and impractical "
  "for a market stall. A practical detector must therefore look at the food and "
  "smell it at the same time, and must reason about the logical consistency "
  "between the two perceptions. This is precisely the setting of multimodal "
  "sensor fusion: a camera measures the external symptoms, a gas-sensor array "
  "measures the volatile chemicals released by decomposition and adulteration, "
  "and a thermal sensor observes sub-surface decay trends.")
P("At the same time, the emergence of energy-efficient microcontrollers with "
  "native machine-learning acceleration, such as the ESP32-S3 with its ESP-NN "
  "vector instructions, makes it possible to run compact neural networks entirely "
  "at the edge — offline, battery-powered, and without any cloud dependency. This "
  "property is essential in markets where reliable internet connectivity cannot "
  "be assumed.")

H("1.2 Problem Statement", 2)
P("Existing food-quality inspection systems suffer from three coupled "
  "limitations. First, laboratory instrumentation (chromatographs, spectrometers, "
  "high-resolution imaging rigs) is too expensive and slow for market-level "
  "screening. Second, single-modality embedded solutions — camera-only or "
  "sensor-only — are easily defeated: camera-only by formalin and surface "
  "washing, and sensor-only by ambient interference from vehicle smoke, garbage, "
  "and mixed shop odors in an open market. Third, cloud-based inference pipelines "
  "require connectivity and expose consumer purchasing data, and thus do not fit "
  "the deployment environment.")
P("The problem addressed by this thesis is therefore: how can a low-cost, "
  "battery-powered edge device simultaneously (a) estimate a reliable "
  "quantitative freshness index of litchi fruit and (b) detect formalin "
  "adulteration that is deliberately designed to be visually invisible, under "
  "noisy real-world sensor conditions?")

H("1.3 Research Objectives", 2)
P("The specific objectives of this thesis are:")
numbers([
    "To design a five-layer edge sensor-fusion execution paradigm — synchronized "
    "ingestion, cross-domain feature extraction, adaptive state-space noise "
    "filtering, late-fusion deep learning inference, and anomaly-based "
    "anti-adulteration alerting.",
    "To train and evaluate a multimodal (image and gas/thermal telemetry) hybrid "
    "neural network that regresses a unified freshness index (0–100) on the "
    "publicly available LADD benchmark dataset.",
    "To implement a Mahalanobis-distance-based anomaly module (Chemical Masking) "
    "that detects the visual–chemical contradiction produced by formalin "
    "adulteration.",
    "To quantize the trained model to a full-integer TFLite flatbuffer suitable "
    "for ESP32-S3 / TensorFlow Lite Micro deployment, and measure the fidelity "
    "loss.",
    "To verify reproducibility of the full pipeline across heterogeneous "
    "platforms (CPU sandbox and Kaggle GPU T4).",
])

H("1.4 Scope and Limitations of the Work", 2)
P("The software contribution of this thesis covers all model-side layers of the "
  "methodology, validated on the LADD benchmark. The hardware layers — the OV2640 "
  "camera, ZE08-CH2O electrochemical formaldehyde sensor, BME680 environmental "
  "array, and MLX90614 infrared thermometer around an ESP32-S3 core with a "
  "TP4056-powered 3.7 V 2000 mAh LiPo battery, a 3D-printed micro-climate "
  "isolation hood, and a KiCad-designed analog/digital-partitioned PCB — are part "
  "of the overall system design of the project, but the experiments reported here "
  "are simulation-stage experiments on published dataset measurements, which is "
  "the standard evaluation practice before firmware integration.")
P("Three dataset-inherited limitations are documented transparently. The "
  "published LADD CSV contains no relative-humidity or temperature columns, so "
  "the BME680 cross-sensitivity compensation layer can only be specified, not "
  "exercised, at this stage; it is reported as simulated on available channels. "
  "The freshness ground truth is a band-based proxy derived from the three class "
  "labels, because the dataset does not ship measured 0–100 sensory scores. "
  "Finally, the uniform lighting and background of the dataset inflate visual "
  "separability relative to open-market conditions.")

H("1.5 Expected Contributions", 2)
bullets([
    "A faithful, reproducible software implementation of the five-layer edge "
    "sensor-fusion methodology in TensorFlow 2 / Keras Functional API.",
    "A late-fusion hybrid network (visual Dense branch plus telemetry Dense "
    "branch) regressing a unified 0–100 freshness index, with a clean "
    "single-input graph that converts losslessly to int8 TFLite.",
    "A Mahalanobis-based Chemical Masking module exposing formalin adulteration "
    "with 100 percent precision and zero false alerts on non-adulterated produce.",
    "A deployment artifact: 357.9 KB full-int8 TFLite flatbuffer with mean output "
    "deviation 0.17 relative to the float32 model.",
    "Cross-platform reproducibility evidence (CPU sandbox and two Kaggle T4 GPU "
    "runs) supporting the thesis evaluation chapter.",
])

H("1.6 Thesis Organization", 2)
P("The remainder of this report is organized as follows. Chapter 2 reviews "
  "literature on image-based fruit quality assessment, gas-sensor-based "
  "detection, multimodal fusion, anomaly detection, and embedded machine "
  "learning. Chapter 3 details the methodology: dataset, ingestion, "
  "cross-domain feature extraction, adaptive filtering, late fusion, anomaly "
  "detection, quantization, and the training configuration. Chapter 4 reports "
  "the experimental results and discussion, including cross-platform "
  "reproduction. Chapter 5 concludes the thesis and outlines future work. "
  "References and appendices follow the conclusion.")

# =========================================================== CHAPTER 2
chapter(2, "LITERATURE REVIEW")

H("2.1 Introduction", 2)
P("This chapter situates the thesis within five strands of prior work: "
  "computer-vision approaches to fruit quality grading; gas-sensing "
  "(electronic-nose) approaches to food safety; information-fusion techniques "
  "that combine multiple sensing modalities; statistical anomaly detection; and "
  "embedded (TinyML) machine learning on microcontrollers. The chapter closes "
  "with a comparative summary that identifies the research gap the present work "
  "fills.")

H("2.2 Image-Based Fruit Quality Assessment", 2)
P("Deep convolutional neural networks dominate modern fruit-grading and "
  "freshness estimation literature. Transfer learning from ImageNet-pretrained "
  "backbones is the dominant recipe when labeled datasets are small. MobileNetV2 "
  "(Sandler et al., 2018) is specifically designed for resource-constrained "
  "inference: it combines inverted residual blocks with linear bottlenecks to "
  "retain accuracy at a fraction of the computational cost of heavyweight "
  "backbones, and MobileNets (Howard et al., 2017) introduced the "
  "depthwise-separable convolutions that make this efficiency possible. These "
  "properties make MobileNetV2 the natural visual backbone for an on-device "
  "pipeline whose final form is a quantized microcontroller model. A widely "
  "observed caveat in this literature, which is also demonstrated in this "
  "thesis, is that laboratory datasets with uniform backgrounds yield optimistic "
  "accuracy, and appearance alone cannot detect adulterants that are engineered "
  "to preserve appearance.")

H("2.3 Gas-Sensor-Based Food Safety Monitoring", 2)
P("Electronic noses monitor the volatile chemical profile of food: decomposition "
  "releases ethanol and elevated CO2, while formalin adulteration raises ambient "
  "formaldehyde. Electrochemical formaldehyde modules such as the ZE08-CH2O "
  "provide a dedicated, UART-interfaced quantitative channel, while "
  "multi-channel environmental arrays such as the BME680 (VOCs, humidity, "
  "pressure, temperature) supply the context needed to interpret chemical drift. "
  "A persistent theme in this literature is that raw analog gas signals are "
  "unstable in open environments, which motivates the state-space filtering and "
  "baseline-tracking stage adopted in this work (Chapter 3).")

H("2.4 Multimodal Sensor Fusion", 2)
P("Fusion strategies are broadly classified as early fusion (concatenating raw "
  "or derived features before a joint model) and late fusion (processing each "
  "modality in dedicated branches and combining their representations "
  "afterwards). Multimodal food-quality studies consistently report that vision "
  "and chemical sensing are complementary: vision captures external degradation, "
  "while chemical sensing captures the biological processes that cause it and "
  "the adulterants that mask it. This thesis adopts a feature-level late fusion "
  "in which the 1280-dimensional visual feature vector and the seven-dimensional "
  "filtered telemetry profile are joined into a joint feature matrix and "
  "processed by a two-branch network with an element-wise fusion layer.")

H("2.5 Statistical Anomaly Detection", 2)
P("The anti-adulteration layer of this thesis is an application of the "
  "Mahalanobis distance (Mahalanobis, 1936), which measures the distance of a "
  "point from a multivariate distribution while accounting for covariance — the "
  "statistically correct way to answer whether a joint (appearance, gas) "
  "observation is consistent with healthy samples. Mahalanobis-distance novelty "
  "detection remains a standard, computationally cheap one-class method that "
  "suits embedded operation because it requires only a mean vector, a covariance "
  "matrix, a threshold, and a quadratic-form evaluation at inference time.")

H("2.6 Embedded Machine Learning (TinyML)", 2)
P("TensorFlow (Abadi et al., 2016) with the Keras Functional API (Chollet, 2015) "
  "is the dominant training stack, and TensorFlow Lite Micro (David et al., "
  "2021) is the standard runtime for inference on microcontrollers such as the "
  "ESP32-S3. Because microcontrollers lack floating-point throughput and large "
  "memory, 8-bit integer post-training quantization with a small representative "
  "calibration set (Jacob et al., 2018) is the standard compression technique; "
  "it converts weights and activations to int8 with per-tensor or per-channel "
  "scales and zero points. Supporting libraries used in this work — NumPy "
  "(Harris et al., 2020), scikit-learn (Pedregosa et al., 2011), the Adam "
  "optimizer (Kingma and Ba, 2015) and Dropout regularization (Srivastava et "
  "al., 2014) — are cited accordingly in the reference list.")

H("2.7 Research Gap", 2)
P("Table 2.1 summarizes the reviewed approaches against this thesis. No prior "
  "reviewed system simultaneously provides (a) freshness-index regression, (b) "
  "formalin-adulteration alerting based on an explicit visual–chemical "
  "contradiction test, (c) state-space noise filtering for open-market "
  "deployment, and (d) a fully-quantized artifact verified for ESP32-S3-class "
  "deployment — all trained and validated on a public benchmark.")
table([
    ["Approach", "Modality", "Freshness index", "Adulteration alert", "Edge-ready (int8 MCU)"],
    ["Camera-only CNN grading", "Image", "Partial / classification only", "No (defeated by formalin)", "Sometimes"],
    ["Electronic nose / gas panel", "Gas sensors", "No", "Partial (gas thresholds)", "Rarely"],
    ["Lab instrumentation (chromatography)", "Chemical, lab", "Yes (lab only)", "Yes (lab only)", "No"],
    ["Cloud vision + sensor IoT", "Image + gas", "Yes", "Rarely explicit", "No (needs connectivity)"],
    ["Proposed pipeline", "Image + gas + thermal", "Yes (0–100 regression)", "Yes (Mahalanobis Chemical Masking)", "Yes (357.9 KB int8)"],
], caption="Table 2.1 Comparison of reviewed detection approaches with the proposed pipeline")

H("2.8 Summary", 2)
P("The literature establishes efficient visual backbones, chemical sensing "
  "channels, fusion architectures, covariance-aware anomaly detection, and int8 "
  "quantization — but treats them as separate contributions. This thesis "
  "integrates all five into a single, reproducible, deployment-oriented pipeline "
  "and evaluates it on the only available public multimodal litchi benchmark "
  "with both appearance and formalin-adulteration ground-truth classes.")

# =========================================================== CHAPTER 3
chapter(3, "METHODOLOGY")

H("3.1 System Overview", 2)
P("The system operates through a strict, sequential multi-stage sensor-fusion "
  "paradigm structured across five operational layers. On the device side, a "
  "3D-printed micro-climate isolation hood is positioned over the food target to "
  "block external wind currents, and the camera, gas-sensor array, and infrared "
  "thermometer are sampled simultaneously at a common timestamp t0. The two "
  "dominant streams — imagery and gas/thermal telemetry — are then processed by "
  "dedicated domains and fused into a single inference decision, which is exposed "
  "over WiFi/BLE to a mobile interface. Every methodology layer maps to a "
  "concrete software component evaluated in this thesis, as detailed in the "
  "following sections.")

H("3.2 Hardware Platform of the Overall Project", 2)
P("The physical subsystem is organized around an ESP32-S3 dual-core Xtensa "
  "microcontroller whose native ESP-NN vector instructions accelerate quantized "
  "neural arithmetic. Power is supplied by a 3.7 V 2000 mAh LiPo battery "
  "regulated through a TP4056 charging module, with automated deep-sleep and "
  "light-sleep duty cycling to maximize field life. Sensing is performed by an "
  "OV2640 parallel-bus camera (surface texture, blemish, and color degradation), "
  "a ZE08-CH2O electrochemical formaldehyde module over UART, a BME680 "
  "environmental array (VOCs, RH, pressure, temperature) over I2C, and an "
  "MLX90614 non-contact infrared thermometer over I2C (sub-surface decay "
  "trends). The analog sensor lines are routed on a KiCad-designed multi-layer "
  "PCB that partitions delicate analog signals away from the high-frequency "
  "camera bus, and the handheld enclosure and isolation hood are prototyped in "
  "SolidWorks/Fusion360 for 3D printing. Firmware is developed in ESP-IDF / "
  "Arduino IDE, and the mobile interface is a Flutter application receiving BLE "
  "telemetry. The experiments of Chapter 4 are the model-side validation of this "
  "platform, executed on published dataset measurements.")

H("3.3 The LADD Benchmark Dataset", 2)
P("All experiments use the publicly published LADD dataset — a multimodal "
  "image–sensor benchmark for litchi freshness and formalin adulteration "
  "detection (Tasnim, Awsaf, and Bijoy, 2025). The dataset pairs a laboratory "
  "image with a seven-channel sensor measurement for each sample. Table 3.1 "
  "gives the composition and Table 3.2 the sensor channels. Both distributed "
  "files were integrity-checked against the published SHA-256 checksums after "
  "download. The dataset is cited in the bibliography through its Mendeley Data "
  "DOI (10.17632/6n9zbrp75r.1).")
table([
    ["Item", "Details"],
    ["Source", "https://data.mendeley.com/datasets/6n9zbrp75r/1 (DOI 10.17632/6n9zbrp75r.1)"],
    ["Classes", "Fresh (1000), Rotten (1000), Formalin-Mixed (1000)"],
    ["Image modality", "3000 RGB JPGs, ~1280×719 pixels, uniform lighting and background"],
    ["Sensor modality", "7 numeric features per sample (Table 3.2)"],
    ["Usable paired samples in this work", "2981 of 3000 (19 rows unresolvable; Section 3.4)"],
    ["Split", "Stratified 70/15/15 → train 2086, validation 447, test 448 (seed 42)"],
], caption="Table 3.1 LADD dataset composition")
table([
    ["Channel", "Unit / role", "Relevant to"],
    ["Moisture", "%", "Fresh produce retains high moisture (75–82% for Fresh)"],
    ["Firmness", "kg/cm2", "Decay softens tissue"],
    ["pH", "pH units", "Decay lowers pH (4.5–5.0 band for Fresh)"],
    ["TSS (Brix)", "°Brix", "Ripeness indicator"],
    ["Ethanol", "ppm", "Fermentation/decomposition marker (BME680 analog)"],
    ["CO2", "ppm", "Respiration marker (BME680 analog)"],
    ["Formaldehyde", "ppm", "Formalin marker (ZE08-CH2O analog; ~0.05 ppm natural vs 2–5 ppm adulterated)"],
], caption="Table 3.2 Sensor channels of the LADD benchmark")

H("3.4 Layer 1 — Synchronized Ingestion (Data Acquisition)", 2)
P("In the field system, the isolation hood and simultaneous trigger guarantee "
  "that the camera frame, gas sampling window, and infrared reading carry the "
  "same t0. In the dataset experiments, synchronization is realized by manifest "
  "pairing: each CSV row is joined to its image file. Three naming "
  "inconsistencies in the published dataset were discovered and handled "
  "deterministically: (i) Fresh images are numbered csv_num + 1000; (ii) 18 "
  "Rotten files arrive as .jpeg duplicates of numbers 1–18; (iii) 19 CSV rows "
  "have no resolvable image and are dropped. The pipeline therefore pairs "
  "2981/3000 rows and reports the count at ingestion time. Each sample carries "
  "one of three integer labels, mapped to class indices 0, 1, and 2 for Fresh, "
  "Rotten, and Formalin-Mixed respectively.")

H("3.5 Layer 2 — Cross-Domain Feature Extraction", 2)
H("3.5.1 Visual Domain (Fv)", 3)
P("Raw surface frames are preprocessed to 224×224 resolution and encoded by a "
  "frozen MobileNetV2 backbone with ImageNet weights and global average "
  "pooling, producing a compact 1280-dimensional spatial feature vector Fv per "
  "sample. Freezing the backbone keeps the visual encoder identical to the "
  "int8-quantizable encoder design and confines training to the lightweight "
  "fusion head, which is exactly the on-device configuration where the CNN acts "
  "as a fixed quantized feature extractor.")
H("3.5.2 Telemetry Domain", 3)
P("Raw telemetry undergoes temporal moving averaging with window w = 7 "
  "(replicated padding, per channel) to smooth random measurement noise caused "
  "by fluctuating market environments, followed by min–max scaling fitted on "
  "the training split only and clipped to the interval [0, 1] on all splits:")
eq("3.1", [("i", "yt"), ("t", " = (1/w) Σ(xt−i)"), ("t", ",   i = 0 … w−1")])
eq("3.2", [("t", "x' = (x − xmin) / (xmax − xmin)")])

H("3.6 Layer 3 — Adaptive Noise Filtering and Baseline Tracking", 2)
P("To convert raw, unstable gas signals into stable, mathematically reliable "
  "inputs, telemetry is passed through an innovation-adaptive one-dimensional "
  "Kalman filter (Kalman, 1960) plus real-time baseline tracking. The predict "
  "step propagates the state variance with process noise q; the update step "
  "compares the incoming measurement with the current estimate to form the "
  "innovation gamma, scales the measurement noise r adaptively from the recent "
  "innovation magnitude (a 10-sample rolling window), and corrects the state:")
eq("3.3", [("t", "pk|k−1 = pk−1|k−1 + q")])
eq("3.4", [("t", "γk = zk − x̂k−1|k−1")])
eq("3.5", [("t", "rk = r0 · (1 + mean‖γ‖₁₀ / (|zk| + ε))")])
eq("3.6", [("t", "Kk = pk|k−1 / (pk|k−1 + rk);   x̂k = x̂k|k−1 + Kk·γk")])
P("with process noise q = 0.0001 and base measurement noise r0 = 0.1. After "
  "filtering, baseline tracking subtracts the median of the first 100 samples "
  "of each channel (the calibration block) so that the network consumes "
  "deviation-from-baseline signals rather than absolute sensor readings, "
  "isolating the true decomposition gases from persistent environmental "
  "backgrounds. The full environmental cross-sensitivity compensation — live "
  "RH%/temperature correction of the ZE08-CH2O calibration curve from the "
  "BME680 — is specified by the methodology as a firmware-side correction; the "
  "published LADD CSV carries no RH/temperature columns, so this stage is "
  "reported as simulated on available channels. Figure 3.1 visualizes the raw "
  "versus filtered telemetry of the three chemically relevant channels.")
figure(f"{FIGS}/fig2_kalman_filtering.png",
       "Figure 3.1 Adaptive Kalman filtering and baseline tracking: raw gas "
       "telemetry (gray) versus moving-average + adaptive-Kalman + "
       "baseline-tracked signals (teal) for Ethanol, CO2, and Formaldehyde.")

H("3.7 Ground-Truth Freshness Index", 2)
P("The LADD dataset provides three class labels rather than measured 0–100 "
  "freshness scores. To train the regression head, a band-based ground-truth "
  "proxy is defined: Fresh 92–99, Rotten 2–18, and Formalin-Mixed 70–84, "
  "sampled uniformly with a fixed random generator (seed 42). The "
  "Formalin-Mixed band is deliberately high: formalin preserves appearance, so "
  "a vision-dominant score alone is high by design — the contradiction that "
  "the anomaly layer of Section 3.9 must resolve. This proxy is documented as "
  "a relative freshness proxy rather than a laboratory-measured score, and its "
  "band structure is visible in every result figure of Chapter 4.")

H("3.8 Layer 4 — Late-Fusion Hybrid Network (Inference Layer)", 2)
P("The extracted Fv and the filtered telemetry are concatenated into a joint "
  "feature matrix of dimension 1287 and processed by a two-branch hybrid "
  "network built with the Keras Functional API. The visual branch maps the "
  "1280-d vector through Dense(256, ReLU) and Dropout(0.2) to Dense(64, ReLU); "
  "the telemetry branch maps the 7-d vector through Dense(64, ReLU). The two "
  "branch outputs are combined by element-wise addition (named late_fusion), "
  "passed through Dense(64, ReLU) and Dropout(0.2), and a sigmoid neuron "
  "scaled by a Rescaling(100) layer produces the unified freshness index in "
  "[0, 100]. Dropout regularization follows Srivastava et al. (2014), and the "
  "Adam optimizer (Kingma and Ba, 2015) minimizes mean squared error. The "
  "split-based two-branch formulation keeps the graph a single-input network — "
  "mathematically equivalent to two separate inputs — which converts cleanly "
  "to an int8 TFLite flatbuffer for the ESP32-S3.")
table([
    ["Component", "Configuration"],
    ["Optimizer", "Adam, learning rate 0.001"],
    ["Loss / metric", "MSE on the freshness index / MAE"],
    ["Epochs / batch size", "60 / 128"],
    ["Backbone", "MobileNetV2 (ImageNet), frozen, global-avg pooling, 1280-d Fv"],
    ["Visual branch", "Dense 256 → Dropout 0.2 → Dense 64"],
    ["Telemetry branch", "Dense 64"],
    ["Fusion / head", "Add (late_fusion) → Dense 64 → Dropout 0.2 → sigmoid × 100"],
    ["Reproducibility", "seed 42 for NumPy, TensorFlow, and all splits"],
], caption="Table 3.3 Network configuration")
P("Training hyperparameters and preprocessing configuration are summarized in "
  "Table 3.4.")
table([
    ["Hyperparameter", "Value"],
    ["Moving-average window w", "7 (replicated padding)"],
    ["Kalman q / r0", "0.0001 / 0.1, innovation window 10"],
    ["Baseline block", "median of first 100 samples per channel"],
    ["Min–max fitting set", "training split only, clipped to [0, 1]"],
    ["Dropout", "0.2 (both branches and fusion head)"],
    ["Epochs / batch", "60 / 128"],
    ["Fv cache", "fv_cache.npy (one-time extraction, reused afterwards)"],
], caption="Table 3.4 Training hyperparameters and configuration")
P("Figure 3.2 shows the training dynamics over the 60 epochs: an early spike "
  "in squared error in the first epoch followed by a smooth convergence of "
  "both training and validation MAE, indicating stable optimization without "
  "divergence or pathological overfitting within the training schedule.")
figure(f"{FIGS}/fig1_training_loss.png",
       "Figure 3.2 Training and validation dynamics of the fusion network over "
       "60 epochs (MSE on the left axis, MAE on the right axis).")

H("3.9 Layer 5 — Anomaly Detection Module (Anti-Adulteration Layer)", 2)
P("The anti-adulteration layer monitors and identifies logical conflicts "
  "between how the food looks and the gases it releases. A Mahalanobis "
  "distance matrix (Mahalanobis, 1936) is fitted on the training Fresh samples "
  "over the joint vector [visual score, CH2O, CO2, Ethanol]: the mean vector "
  "mu and covariance Sigma of healthy fruit define the reference "
  "gas–appearance state, and the squared distance of any observation x from "
  "this state is")
eq("3.7", [("t", "D2(x) = (x − μ)ᵀ Σ⁻¹ (x − μ)")])
P("The anomaly threshold is set to the 99th percentile of the training-Fresh "
  "distances, so that at most one percent of genuinely fresh samples would "
  "trigger an alert statistically. The Chemical Masking alert fires when all "
  "three conditions hold simultaneously: the visual score is fresh (70 or "
  "higher), the formaldehyde reading exceeds 0.5 ppm, and the Mahalanobis "
  "distance exceeds the threshold. The covariance matrix is regularized by "
  "adding 0.000001 to its diagonal before inversion for numerical stability. "
  "This is exactly the contradiction test the methodology prescribes: a "
  "vendor-treated sample appears fresh but carries an outlying toxic-gas "
  "signature, bypassing the standard prediction and triggering an alert on the "
  "mobile interface.")

H("3.10 Deployment Artifact — Post-Training 8-bit Quantization", 2)
P("For ESP32-S3 deployment the trained network is converted to TensorFlow "
  "Lite with full-integer post-training quantization (Jacob et al., 2018): a "
  "300-sample representative calibration set drawn from the training split "
  "supplies activation ranges, and both weights and activations (including "
  "input and output) are quantized to int8 with TFLITE_BUILTINS_INT8 operator "
  "support only. Because quantized microcontroller inference always runs at "
  "batch size 1, calibration and conversion are verified with batch-1 "
  "representative data. The conversion path exports the model to a SavedModel "
  "before conversion to avoid a Keras-3 multi-input calibration limitation, "
  "and the resulting flatbuffer is validated against the float32 model on the "
  "test set.")

H("3.11 Evaluation Protocol", 2)
P("Evaluation uses the held-out test split (448 samples, stratified). "
  "Reported metrics are: per-class and overall MAE of the freshness index on "
  "the 0–100 scale; Fresh-versus-Rotten discrimination at the visual threshold "
  "70; Chemical-Masking recall, precision, and false alerts per class; the "
  "combined full-pipeline accuracy (Fresh correct + Rotten correct + Formalin "
  "caught); the int8/float32 agreement statistics; and cross-platform "
  "reproducibility of the entire pipeline.")

# =========================================================== CHAPTER 4
chapter(4, "RESULTS AND DISCUSSION")

H("4.1 Experimental Setup", 2)
P("All results are on the held-out test set of 448 samples (Fresh 150, Rotten "
  "148, Formalin-Mixed 150) generated by the fixed stratified split of 2981 "
  "paired samples. The canonical run executes on a CPU sandbox with TensorFlow "
  "2.21; two independent Kaggle runs on NVIDIA T4 GPUs reproduce the pipeline "
  "from manually uploaded dataset copies. All runs use seed 42 and identical "
  "data ordering, so any metric deviation is attributable to floating-point "
  "numerics of the backend, not to methodological changes.")

H("4.2 Freshness-Index Regression", 2)
P("Table 4.1 reports per-class and overall regression quality of the canonical "
  "run. The network predicts each class band tightly (Fresh centered near 96, "
  "Rotten near 9.6, Formalin-Mixed near 77), with an overall MAE of 3.49/100 — "
  "an average freshness-index uncertainty of about 3.5 points on the 100-point "
  "scale. Figure 4.1 shows predicted versus ground-truth values against the "
  "ideal y = x line: the three classes form separated clusters that hug the "
  "diagonal, confirming that the fusion head learned the ordinal structure of "
  "freshness rather than a collapsed mapping.")
table([
    ["Class", "MAE", "Predicted mean", "Ground-truth mean"],
    ["Fresh", "2.10", "96.2", "95.4"],
    ["Rotten", "4.18", "9.6", "10.0"],
    ["Formalin-Mixed", "4.21", "77.4", "76.8"],
    ["Overall", "3.49", "—", "—"],
], caption="Table 4.1 Freshness-index regression results (test set, n = 448; canonical run)")
figure(f"{FIGS}/fig3_pred_vs_gt.png",
       "Figure 4.1 Predicted versus ground-truth freshness index for the three "
       "classes with the ideal y = x reference line.")
P("Decision behavior at the operational threshold is exact: at the "
  "freshness-index cut-off of 70, 100 percent of Fresh test samples are "
  "classified fresh and 100 percent of Rotten samples are classified degraded — "
  "a perfect Fresh-versus-Rotten discrimination of 298/298 samples, reproduced "
  "identically in both GPU runs.")

H("4.3 Mahalanobis Anomaly Module and Chemical Masking", 2)
P("The anomaly module fitted the reference covariance on training Fresh samples "
  "with the 99th-percentile threshold at D = 19.52 for the canonical run (14.71 "
  "and 20.21 across the two GPU reproductions — a model-derived quantity that "
  "tracks small numeric differences between backends and remains conservative "
  "in each run). Figure 4.2 shows the log-scale distance distributions of the "
  "three classes with the threshold marked: Fresh distances concentrate below "
  "the threshold, while adulterated and rotten samples sit orders of magnitude "
  "to the right of the boundary.")
figure(f"{FIGS}/fig4_mahalanobis_threshold.png",
       "Figure 4.2 Mahalanobis distance distributions (log10 scale) with the "
       "anomaly threshold (D = 19.52, canonical run) marked.")
P("Table 4.2 reports the masking outcome. The module identifies 144/150 (96.0 "
  "percent) of Formalin-Mixed test samples as chemically masked while raising "
  "zero alerts on Fresh and zero on Rotten — the desired behavior, where Rotten "
  "is handled by the visual pathway (its appearance is degraded) rather than by "
  "the chemical-contradiction pathway. Masking precision is 100 percent: every "
  "alert is a true adulteration.")
table([
    ["True class (test)", "Samples", "Chemical-Masking alerts"],
    ["Fresh", "150", "0"],
    ["Rotten", "148", "0"],
    ["Formalin-Mixed", "150", "144  (recall 96.0%, precision 100%)"],
], caption="Table 4.2 Chemical-Masking module results (test set, n = 448; canonical run)")

H("4.4 Combined Classification and Full-Pipeline Accuracy", 2)
P("The combined decision rule — masking alert implies Formalin; otherwise score "
  "at least 70 implies Fresh; otherwise Rotten — yields the confusion matrix of "
  "Figure 4.3: 150/150 Fresh correct, 148/148 Rotten correct, and 144/150 "
  "Formalin-Mixed correct in the canonical run, i.e. an overall pipeline "
  "accuracy of approximately 98.9 percent. In the reproduced Kaggle runs the "
  "matrix is [150, 0, 0; 0, 148, 0; 3, 4, 143] and the full-pipeline accuracy "
  "is 441/448 = 98.4 percent with zero false alerts on either non-adulterated "
  "class. The six missed adulterated samples in the canonical run (five in the "
  "reproduced runs) are borderline-formaldehyde cases whose gas signature lies "
  "inside the Fresh envelope — a known limitation of any fixed-quantile "
  "threshold that motivates the ROC-tuning future work of Chapter 5.")
figure(f"{FIGS}/fig5_confusion_matrix.png",
       "Figure 4.3 Confusion matrix of the combined pipeline with raw counts "
       "(left) and row-normalized recall (right).")

H("4.5 Quantization Impact", 2)
P("Post-training int8 quantization compresses the deployment artifact from "
  "4.24 MB (Keras float32) to 357.9 KB — a 91.6 percent reduction — comfortably "
  "inside the flash and RAM envelope of an ESP32-S3 application partition. "
  "Output fidelity is preserved: the mean absolute difference between float32 "
  "and int8 freshness scores is 0.17 on the 0–100 index (maximum below 0.8), "
  "i.e. quantization costs a fraction of a point of precision, negligible "
  "against an overall MAE of 3.49. Figure 4.4 visualizes both the size "
  "reduction and the per-sample fidelity scatter.")
table([
    ["Metric", "Keras float32", "TFLite int8"],
    ["Artifact size", "4.24 MB", "357.9 KB  (91.6% smaller)"],
    ["Output difference vs float32", "—", "mean 0.17, max below 0.8  (0–100 index)"],
    ["Ops / inference type", "float32", "full int8 (input and output int8)"],
], caption="Table 4.3 Quantization impact summary")
figure(f"{FIGS}/fig6_quantization_impact.png",
       "Figure 4.4 Quantization impact: model-size reduction (left) and "
       "int8-versus-float32 output fidelity (right).")

H("4.6 Cross-Platform Reproducibility", 2)
P("To demonstrate that the reported numbers are implementation-independent and "
  "not an accident of one machine, the identical pipeline — same seed, same "
  "code path — was re-executed end-to-end on Kaggle GPU T4 using a manually "
  "uploaded copy of the dataset. Table 4.4 collects the three runs. Every "
  "qualitative conclusion is identical across platforms: perfect Fresh/Rotten "
  "discrimination, 95.3–96.7 percent masking recall, 100 percent masking "
  "precision, zero false alerts, and a byte-identical 357.9 KB int8 flatbuffer. "
  "Run-to-run MAE differences of a few hundredths and threshold shifts of a "
  "few points reflect GPU floating-point numerics, exactly as expected.")
table([
    ["Run", "Platform", "Overall MAE", "Masking recall", "Precision", "False alerts", "Threshold", "Full-pipeline acc."],
    ["Sandbox (canonical)", "CPU", "3.49", "96.0% (144/150)", "100%", "0", "19.52", "98.9%"],
    ["Kaggle run A", "GPU T4", "3.40", "96.7% (145/150)", "100%", "0", "14.71", "98.4%"],
    ["Kaggle run B", "GPU T4", "3.46", "95.3% (143/150)", "100%", "0", "20.21", "98.4%"],
], caption="Table 4.4 Cross-platform reproducibility of the full pipeline")

H("4.7 Discussion", 2)
P("The results support both research claims. First, a lightweight late-fusion "
  "network over frozen visual features and filtered telemetry estimates the "
  "freshness index with a 3.49-point average error and perfectly separates the "
  "edible-looking classes at the operational threshold, using a model that "
  "quantizes to 357.9 KB. Second, the Mahalanobis contradiction test converts "
  "formalin's visual camouflage from a weakness into a detection signal: "
  "precisely because the adulterated fruit looks fresh, its high appearance "
  "score combined with an anomalous gas state is diagnostic, yielding 95–97 "
  "percent detection with zero false alarms.")
P("Three limitations qualify the interpretation, and are restated from Section "
  "1.4 with their result-level consequences. The synthetic band ground truth "
  "means the reported MAE measures band-reconstruction fidelity rather than "
  "agreement with a laboratory freshness scale. The absence of RH/temperature "
  "columns means the cross-sensitivity compensation of Layer 3 is specified "
  "but not exercised on columnar data. The uniform dataset background means "
  "field accuracy will be lower, which argues for an out-of-distribution "
  "market validation set (Chapter 5). None of these limitations weakens the "
  "software-layer design, which is defined and validated on the channels the "
  "benchmark does provide.")

# =========================================================== CHAPTER 5
chapter(5, "CONCLUSION AND FUTURE WORK")

H("5.1 Conclusion", 2)
P("This thesis designed, implemented, and validated a five-layer edge "
  "sensor-fusion paradigm for litchi freshness estimation and "
  "formalin-adulteration alerting. On the public LADD benchmark, the pipeline "
  "regresses a unified 0–100 freshness index with overall MAE 3.49 in the "
  "canonical CPU run, reproduced at 3.40–3.46 in two independent GPU runs; "
  "discriminates Fresh from Rotten at 100 percent; and exposes formalin "
  "adulteration with 95.3–96.7 percent recall, 100 percent precision, and zero "
  "false alerts through the Mahalanobis-based Chemical Masking layer, for a "
  "full-pipeline accuracy of up to 98.9 percent. The trained graph converts to "
  "a 357.9 KB full-int8 TFLite flatbuffer with a mean score deviation of 0.17, "
  "meeting the memory and precision constraints of the ESP32-S3 / TensorFlow "
  "Lite Micro target. All five methodology layers — synchronization-by-manifest "
  "ingestion, cross-domain feature extraction, adaptive Kalman filtering with "
  "baseline tracking, late-fusion regression, and covariance-aware anomaly "
  "detection — are implemented in a reproducible codebase whose results are "
  "stable across hardware platforms.")

H("5.2 Future Work", 2)
numbers([
    "Firmware integration: run the int8 flatbuffer with TensorFlow Lite Micro "
    "and ESP-NN on the ESP32-S3, implement the multi-threaded sensor polling "
    "loop with the t0 simultaneous trigger, and add deep-sleep duty cycling for "
    "field battery life.",
    "Humidity/temperature cross-sensitivity compensation: once field hardware "
    "streams BME680 RH% and temperature, correct the ZE08-CH2O calibration "
    "curve and re-validate the masking thresholds under humid market conditions.",
    "Field validation: collect a small out-of-distribution open-market dataset "
    "(uneven lighting, smoke, mixed odors) to measure honest degradation of "
    "both the visual branch and the anomaly layer.",
    "Threshold tuning: replace the fixed 99th-percentile masking rule with "
    "ROC-based threshold selection, and analyze the borderline-CH2O missed "
    "specimens.",
    "Dataset community feedback: report the CSV–image naming inconsistencies "
    "(19 unresolved rows) to the LADD authors so that a V2 strengthens the "
    "benchmark.",
    "Mobile application: extend the Flutter interface with live dashboards "
    "over BLE and historical trends of the freshness index and gas channels.",
])

# =========================================================== REFERENCES
sec = new_section()
page_number(sec)
hr = doc.add_heading("", level=1)
rich(hr, "REFERENCES", size=14, bold=True)
refs = [
    "Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., Dean, J., Devin, M., "
    "Ghemawat, S., Irving, G., Isard, M., Kudlur, M., Levenberg, J., Monga, R., "
    "Moore, S., Murray, D. G., Steiner, B., Tucker, P., Vasudevan, V., Warden, "
    "P., & Zheng, X. (2016). TensorFlow: A system for large-scale machine "
    "learning. In Proceedings of the 12th USENIX Symposium on Operating Systems "
    "Design and Implementation (OSDI 16), 265–283.",
    "Chollet, F. (2015). Keras [Computer software]. "
    "https://github.com/fchollet/keras",
    "David, R., Duke, J., Jain, A., Reddy, V. J., Jeffries, E., Li, J., "
    "Kreeger, N., Nappier, I., Natraj, M., Peddinti, S., Rhodes, R., & Wang, R. "
    "(2021). TensorFlow Lite Micro: Embedded machine learning for TinyML "
    "systems. Proceedings of Machine Learning and Systems, 3, 800–811.",
    "Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., "
    "Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. "
    "J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M. H., Brett, M., & "
    "Oliphant, T. E. (2020). Array programming with NumPy. Nature, 585, "
    "357–362. https://doi.org/10.1038/s41586-020-2649-2",
    "Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., "
    "Andreetto, M., & Adam, H. (2017). MobileNets: Efficient convolutional "
    "neural networks for mobile vision applications. arXiv preprint "
    "arXiv:1704.04861.",
    "Jacob, B., Kligys, S., Chen, B., Zhu, M., Tang, M., Howard, A., Adam, H., "
    "& Kalenichenko, D. (2018). Quantization and training of neural networks "
    "for efficient integer-arithmetic-only inference. In Proceedings of the "
    "IEEE Conference on Computer Vision and Pattern Recognition (CVPR), "
    "2704–2713. https://doi.org/10.1109/CVPR.2018.00186",
    "Kalman, R. E. (1960). A new approach to linear filtering and prediction "
    "problems. Journal of Basic Engineering, 82(1), 35–45. "
    "https://doi.org/10.1115/1.3662552",
    "Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic "
    "optimization. In Proceedings of the 3rd International Conference on "
    "Learning Representations (ICLR).",
    "Mahalanobis, P. C. (1936). On the generalized distance in statistics. "
    "Proceedings of the National Institute of Sciences of India, 2(1), 49–55.",
    "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., "
    "Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., "
    "Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & "
    "Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. Journal "
    "of Machine Learning Research, 12, 2825–2830.",
    "Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). "
    "MobileNetV2: Inverted residuals and linear bottlenecks. In Proceedings of "
    "the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), "
    "4510–4520. https://doi.org/10.1109/CVPR.2018.00474",
    "Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & "
    "Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural "
    "networks from overfitting. Journal of Machine Learning Research, 15, "
    "1929–1958.",
    "Tasnim, S. Z., Awsaf, S. A., & Bijoy, M. H. I. (2025). LADD: A multimodal "
    "image–sensor benchmark dataset for litchi freshness and formalin "
    "adulteration detection. Mendeley Data, V1. "
    "https://doi.org/10.17632/6n9zbrp75r.1",
]
for r in refs:
    par = P(r, size=12)
    par.paragraph_format.line_spacing = 1.0  # single-spaced reference entries
    par.paragraph_format.space_after = Pt(6)
    par.paragraph_format.left_indent = Cm(1.27)
    par.paragraph_format.first_line_indent = Cm(-1.27)  # hanging indent

# =========================================================== APPENDICES
sec = new_section()
hap = doc.add_heading("", level=1)
rich(hap, "APPENDICES", size=14, bold=True)

H("Appendix A: Reproduction Instructions", 2)
P("The complete pipeline (train_ladd_tf.py) and figure generator "
  "(make_thesis_figures.py) are provided in the project repository. "
  "Environment: Python 3.10 or later with the tensorflow, pandas, "
  "scikit-learn, numpy, and matplotlib packages. The dataset must be "
  "downloaded from the Mendeley Data source (DOI 10.17632/6n9zbrp75r.1), "
  "integrity-checked against the published SHA-256 checksums, and unzipped to "
  "the layout data/Litchi_Sensor_Data.csv plus data/images/{Fresh, Rotten, "
  "Formalin}; the DATA constant at the top of the training script is set to "
  "that folder.")
P("Typical reproduction steps:", italic=True, single=True)
for i, line in enumerate([
    "pip install tensorflow-cpu pandas scikit-learn numpy matplotlib",
    "python train_ladd_tf.py     # trains, evaluates, writes the int8 TFLite artifact",
    "python make_thesis_figures.py   # regenerates all six thesis figures",
]):
    P(f"{i + 1}. {line}", single=True)
P("Training takes about ten minutes on CPU; with the feature cache "
  "(fv_cache.npy) present, fusion-head training completes in seconds. The run "
  "writes hybrid_freshness.keras, hybrid_freshness_int8.tflite (357.9 KB), the "
  "training log, and the split indices.")

H("Appendix B: Kaggle (GPU T4) Reproduction Output Summary", 2)
P("The final Kaggle run (GPU T4) reproduced the whole pipeline from a manually "
  "uploaded dataset copy and reported the following summary, quoted from its "
  "output:")
for line in [
    "[1] ingestion: paired 2981/3000 CSV rows with images",
    "[split] train 2086, val 447, test 448",
    "Epoch 60/60 — loss: 14.1437, mae: 2.8893, val_loss: 21.2324, val_mae: 3.5547",
    "Fresh MAE 2.17 (predicted 96.5, gt 95.4); Rotten MAE 4.24 (8.7 / 10.0); "
    "Formalin-Mixed MAE 3.97 (78.0 / 76.8); Overall MAE 3.46 / 100",
    "Visual-score threshold 70: Fresh 100%, Rotten 100% — Fresh-vs-Rotten "
    "accuracy 100.0% (298/298)",
    "Mahalanobis threshold (99% quantile of train Fresh): 20.21",
    "Chemical Masking: Fresh 0, Rotten 0, Formalin-Mixed 143 → recall 95.3%, "
    "precision 100%, false alerts 0",
    "FULL-PIPELINE ACCURACY: 441/448 = 98.4%",
    "int8 flatbuffer written: 357.9 KB (Keras model 4141.7 KB)",
    "Six figures generated at 300 DPI (PNG + PDF) inside the session output folder.",
]:
    P(line, single=True)

doc.save(OUT)
print("saved", OUT)
