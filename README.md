# 🛰️ SatVerify: Deepfake Satellite Image Detection for Carbon Credit Markets

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/frontend-Streamlit%20v1.51-red.svg)](https://streamlit.io/)
[![Test Suite: 100% Pass](https://img.shields.io/badge/tests-19%2F19%20passed-brightgreen.svg)](tests/)

> **SatVerify** is an end-to-end multi-branch satellite image forensics system engineered to verify the authenticity of satellite imagery in carbon offset projects, detecting "paper forests," synthetic vegetation inpainting, and digital terrain tampering.

---

## 📌 Problem Statement: The "Paper Forest" Threat

Carbon credit markets rely on optical satellite imagery (Sentinel-2, Landsat, PlanetScope) to verify forest growth, canopy cover, and ecological restoration before credits are minted and traded.

Fraudulent actors can manipulate evidence by:
1. **Copy-Pasting Vegetation:** Splicing authentic forest textures onto barren land or deforested clearings.
2. **Generative Synthesis:** Synthesizing artificial tree canopies using GANs or Diffusion models.
3. **Green-Channel Saturation:** Artificially enhancing green spectrum ratios to fake high Normalized Difference Vegetation Index (NDVI).
4. **Patch Repetition:** Tiling small forest patches to fabricate continuous canopy cover.
5. **Boundary Splicing:** Merging imagery from disparate seasons, acquisition sensors, or geographic locations.

SatVerify provides independent, explainable verification to protect carbon market integrity.

---

## 🔬 Core Features & Forensic Branches

SatVerify fuses deterministic physical signal analysis with optional deep learning classifiers:

- **Branch A — ResNet50 Transfer Learning:** Custom binary classification head with Global Average Pooling, Dropout, and Sigmoid output. Operates with strict model honesty: uncalibrated weights are excluded.
- **Branch B — Error Level Analysis (ELA):** Detects differential compression residuals between original and recompressed JPEG representations.
- **Branch C — Fourier Transform Frequency Analysis (2D FFT):** Identifies periodic grid spikes and anomalous high-frequency noise from generative upsampling operators.
- **Branch D — Vision Transformer (ViT):** Patch-level self-attention over $16 \times 16$ spatial tokens to flag long-range semantic discontinuities.
- **Branch E — Local Binary Pattern (LBP) Texture Analysis:** Evaluates micro-texture distributions and regional Shannon entropy to catch cloned canopy tiles and unnatural smoothing.
- **Branch F — Structural Consistency:** Sobel gradient continuity and Excess Green ($2G - R - B$) boundary sharpness analysis.
- **Branch G — Spatial Coherence & Seam Detection:** Step-gradient profiles and cross-channel Pearson correlation ($R, G, B$) to detect rectangular splicing seams.
- **Branch H — Shadow & Solar Geometry:** Segments shadow boundaries, measures directional lighting coherence, and cross-references solar azimuth when acquisition metadata is provided.
- **Branch I — Elevation & Terrain Validation:** Cross-references canopy masks against Digital Elevation Models (DEM) to flag physical impossibilities.
- **Weighted Score Fusion:** Dynamically renormalizes weights across active branches (summing to 100%).
- **Tri-State Decision Engine:** Classifies inputs into `REAL`, `FAKE`, or `INCONCLUSIVE` (uncertainty margin $[0.38, 0.62)$).
- **Publication-Quality PDF Reports:** Generates 2-page audit certificates with analysis IDs, branch metrics, and embedded visual strips.

---

## 🏛️ System Architecture

```
Satellite Image Upload (RGB / NIR)
              │
              ▼
   Image Validation & Preprocessing (Multi-Scale Normalization)
              │
  ┌───────────┼───────────┬───────────┬───────────┐
  ▼           ▼           ▼           ▼           ▼
Branch A    Branch B    Branch C    Branch E    Branch F, G, H, I
(ResNet50)   (ELA)       (FFT)       (LBP)     (Structure, Seams, Shadow)
  │           │           │           │           │
  └───────────┼───────────┴───────────┴───────────┘
              ▼
    Weighted Score Fusion (Dynamic Re-normalization over Active Branches)
              │
              ▼
     Decision Engine (REAL < 0.38 <= INCONCLUSIVE < 0.62 <= FAKE)
              │
              ▼
    Streamlit Web Dashboard & Exportable PDF Audit Report
```

---

## 📂 Project Structure

```text
SatVerify/
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Python package dependencies
├── .gitignore                  # Git ignore rules protecting models and data
├── .env.example                # Configuration template
├── README.md                   # Complete system documentation
├── RECOVERY_REPORT.md          # Asset audit and recovery report
│
├── config/                     # Centralized settings and thresholds
│   ├── __init__.py
│   └── settings.py
│
├── core/                       # Core pipeline and decision logic
│   ├── __init__.py
│   ├── pipeline.py             # Master SatVerifyPipeline orchestrator
│   ├── preprocessing.py        # Image loading and multi-scale normalization
│   ├── score_fusion.py         # Dynamic weighted score fusion
│   └── decision_engine.py      # Tri-state (REAL/FAKE/INCONCLUSIVE) classifier
│
├── models/                     # Deep learning classifiers & loaders
│   ├── __init__.py
│   ├── resnet_classifier.py    # ResNet50 transfer learning binary classifier
│   ├── vit_analyzer.py         # Vision Transformer patch analyzer
│   └── model_loader.py         # Centralized model registry and caching
│
├── forensics/                  # Deterministic mathematical forensic branches
│   ├── __init__.py
│   ├── base.py                 # ForensicResult standard return container
│   ├── ela.py                  # Branch B: Error Level Analysis
│   ├── frequency_analysis.py   # Branch C: 2D Fast Fourier Transform
│   ├── texture_analysis.py     # Branch E: Local Binary Patterns (LBP)
│   ├── structure_analysis.py   # Branch F: Structural consistency & seams
│   ├── geometry_analysis.py    # Branch G: Geometry & color correlation
│   ├── shadow_analysis.py      # Branch H: Shadow & solar geometry
│   └── elevation_analysis.py   # Branch I: DEM elevation & slope validation
│
├── explainability/             # Explainability and chart generators
│   ├── __init__.py
│   ├── gradcam.py              # CNN Gradient-weighted Class Activation Mapping
│   └── visualizations.py       # Comparison panels and score charts
│
├── services/                   # Application services
│   ├── __init__.py
│   ├── image_service.py        # File lifecycle and temporary storage
│   └── report_service.py       # Publication-quality PDF audit generator
│
├── utils/                      # Utilities
│   ├── __init__.py
│   ├── logging_config.py       # Structured logging
│   └── validation.py           # File validation and metadata extraction
│
├── training/                   # Dataset generation, splitting & training
│   ├── __init__.py
│   ├── generate_manipulations.py # Vegetation copy-paste & green boosting
│   ├── prepare_dataset.py      # Split organizer and smoke fixture generator
│   ├── train.py                # ResNet50 training pipeline
│   └── evaluate.py             # Benchmark precision, recall, F1, and AUC
│
├── tests/                      # Automated pytest suite (19 tests, 100% pass)
│   ├── __init__.py
│   ├── test_image_validation.py
│   ├── test_ela.py
│   ├── test_frequency.py
│   ├── test_texture_structure.py
│   ├── test_score_fusion.py
│   ├── test_pipeline.py
│   └── test_report.py
│
├── data/                       # Datasets & synthetic fixtures
│   ├── .gitkeep
│   └── fixtures/               # Preset synthetic real & tampered samples
├── checkpoints/                # Model weights directory
│   └── .gitkeep
├── reports/                    # Generated audit PDF certificates
│   └── .gitkeep
└── docs/                       # Specifications and model cards
    ├── ARCHITECTURE.md
    └── MODEL_CARD.md
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.11, 3.12, or 3.13 (64-bit recommended)
- Windows, macOS, or Linux

### 1. Clone or Open Workspace
```bash
cd SatVerify
```

### 2. Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Core Dependencies
```bash
pip install -r requirements.txt
```

*(Optional: If training neural network classifiers, install `torch` or `tensorflow`)*:
```bash
pip install torch torchvision transformers
# or
pip install tensorflow
```

---

## 🛰️ Launching the Application

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will start locally at:
```text
http://localhost:8501
```

### Quick Demo with Presets
Inside the application, open the **"Quick Demo: Load Preset Fixture"** dropdown to immediately test authentic and tampered satellite samples without uploading external files!

---

## 🧪 Running Automated Tests

Run the complete test suite with `pytest`:

```bash
pytest tests/ -v
```

All 19 tests will execute, validating:
1. Image validation & corrupted file rejection.
2. Error Level Analysis (ELA) array generation & bounded scoring.
3. Fast Fourier Transform (FFT) magnitude harmonics & spike ratios.
4. LBP micro-texture entropy & regional variance.
5. Structural gradient continuity & seam detection.
6. Score fusion weight re-normalization (weights sum to 1.0).
7. Exclusion of unavailable branches (never treated as 0-anomaly).
8. Tri-state decision classification (`REAL`, `FAKE`, `INCONCLUSIVE`).
9. End-to-end pipeline execution on real/fake fixtures.
10. Multi-page PDF report generation.

---

## 📊 Dataset Preparation & Synthetic Manipulations

### Generate Smoke-Test Fixtures
Generate procedural satellite images for offline testing:
```bash
python training/prepare_dataset.py --generate-fixtures --samples 8
```

### Generate Synthetic Manipulations
Use the manipulation engine to create copy-pasted vegetation, green-boosted images, and spliced scenes:
```python
from PIL import Image
from training.generate_manipulations import ManipulationGenerator

gen = ManipulationGenerator()
img = Image.open("path/to/authentic_forest.jpg")

# 1. Vegetation Copy-Paste (paper forest simulation)
tampered_img, meta = gen.copy_paste_vegetation(img, patch_size=(64, 64))

# 2. Artificially Boost Green Channel (fake NDVI enhancement)
boosted_img, meta = gen.boost_green_channel(img, boost_factor=1.5)
```

### Partition Dataset into Train / Validation / Test
```bash
python training/prepare_dataset.py
```

---

## 🧠 Model Training & Evaluation

### Train ResNet50 Transfer Learning Classifier
```bash
python training/train.py --data-dir data --epochs 10 --batch-size 16
```

### Evaluate Pipeline Performance
```bash
python training/evaluate.py --test-dir data/fixtures --output reports/evaluation_results.json
```

Outputs: Precision, Recall, Specificity, F1 Score, Inconclusive Rate, and Confusion Matrix.

---

## ⚖️ Decision Policy & Model Honesty

| Fused Suspicion Score | Classification | Action |
|---|---|---|
| $S < 0.38$ | **REAL (Authentic Terrain)** | High confidence of natural capture. |
| $S \ge 0.62$ | **FAKE (Altered / Synthetic)** | Forensic anomalies flagged for investigation. |
| $0.38 \le S < 0.62$ | **INCONCLUSIVE** | Ambiguous signals; manual audit required. |
| Branch count $< 2$ | **INCONCLUSIVE** | Insufficient evidence to establish decision. |

> **Model Honesty Principle:** Generic ImageNet models without verified satellite deepfake training are never presented as authenticity detectors. SatVerify labels them as `UNAVAILABLE` and transparently relies on deterministic physical forensics.

---

## 📄 Legal & Scientific Disclaimer

SatVerify is an automated image-forensics research tool. A **REAL** classification indicates the absence of detected digital manipulation signatures, not legal or certified proof of biological carbon sequestration. Analysis findings should be corroborated with ground-truth field audits, multispectral indices, and sensor telemetry.

---

## 👥 Contributors & Acknowledgements
- Developed for Hackathon Carbon Credit Verification Challenge.
- Built with Streamlit, NumPy, SciPy, Pillow, Matplotlib, Scikit-Image, and PyTorch/TensorFlow.
