# SatVerify: Project Recovery & Audit Assessment Report

**Date of Audit:** October 2026  
**Auditor / Lead Engineer:** Antigravity AI Engineering  
**Project:** SatVerify — Deepfake Satellite Image Detection for Carbon Credit Markets  
**Status:** Recovery & Architecture Reconstruction in Progress  

---

## 1. Executive Summary

An audit of the repository workspace (`e:\SatVerify`) was conducted to assess surviving assets, identify deleted and incomplete components, and establish a systematic roadmap for reconstructing the full end-to-end multi-branch satellite image forensics system.

The surviving code represented a preliminary monolithic prototype focusing narrowly on a Vision Transformer (ViT) wrapper, basic Error Level Analysis (ELA), and Fast Fourier Transform (FFT) frequency analysis. However, critical components of the hackathon architecture specification—including the primary ResNet50 deep learning classifier, modular forensic feature extractors (LBP texture, structural consistency, geometry, shadow/solar geometry, and elevation/DEM validation), weighted score fusion, calibrated decision engine with INCONCLUSIVE support, Grad-CAM explainability, synthetic manipulation generator, comprehensive automated tests, and PDF report generation—were missing or unbuilt.

All surviving files have been cataloged and backed up to `_backup_surviving_files/`. Reconstruction will preserve and enhance surviving analytical routines while delivering a modular, production-grade, and scientifically honest application.

---

## 2. Surviving Assets Inventory

| File Path | Size (Bytes) | Nature / Role | State / Assessment |
|---|---:|---|---|
| `app.py` | 11,311 | Streamlit App | Monolithic 3-column UI. Hardcoded to a 3-way ensemble (ViT 20%, ELA 40%, FFT 40%). Lacks modular design, advanced branches, PDF report generation, and robust error handling. |
| `forensics.py` | 9,114 | Forensic Signal Utilities | Implements PIL image loading, basic ELA residual amplification, FFT 2D magnitude spectrum, and ELA/FFT variance/spike heuristic scores. Mathematically sound foundation for Branches B & C. |
| `model.py` | 11,367 | ViT Model Wrapper | PyTorch wrapper for HuggingFace `google/vit-base-patch16-224`. Missing ResNet50 primary classifier, fine-tuned weights, and graceful fallbacks when weights are absent. |
| `train.py` | 20,883 | ViT Training Script | PyTorch CLI training script for fine-tuning ViT. Requires an external dataset directory with `REAL` and `FAKE` splits. Lacks synthetic manipulation generation, data augmentation controls, and ResNet50 training. |
| `requirements.txt` | 262 | Dependencies | Lists torch, torchvision, transformers, streamlit, opencv, skimage, Pillow, numpy, matplotlib, scipy. Missing pdf generation (reportlab), etc. |
| `streamlit_stdout.log` | 183 | Runtime Log | Confirms previous local execution on port 8503. |
| `__pycache__/*` | Variable | Bytecode Cache | Python 3.13 bytecode artifacts confirming previous local compilation. |

**Backup Action Taken:** All surviving files have been securely copied to `_backup_surviving_files/` prior to any architectural refactoring.

---

## 3. Missing Components Analysis

The following required architectural modules and files were absent from the repository:

1. **Modular Architecture Structure:**
   - `config/settings.py`: Centralized configuration, thresholds, branch weights, and paths.
   - `core/preprocessing.py`: Multi-format image validation, channel normalization, dimension enforcement.
   - `core/score_fusion.py`: Multi-branch weighted fusion with dynamic re-normalization over available branches.
   - `core/decision_engine.py`: REAL / FAKE / INCONCLUSIVE tri-state classification with uncertainty thresholds.
   - `core/pipeline.py`: Unified pipeline orchestrating validation, parallel/sequential branch execution, score aggregation, and explainability.

2. **Forensic Branches:**
   - **Branch A (ResNet50 Classifier):** Transfer learning CNN with custom binary classification head (GAP, Dropout, Dense ReLU, Sigmoid). Transparent fallback mode when weights are uncalibrated or uninstalled.
   - **Branch D (ViT Analyzer):** Modular Vision Transformer analysis interface with honest reporting of model availability and patch-level anomaly detection.
   - **Branch E (Texture Analysis - LBP):** Local Binary Pattern histograms, regional entropy, and repetition/uniformity heuristics.
   - **Branch F (Structural Consistency):** Canny/Sobel edge discontinuities, Hough line structural alignment, grid consistency, and vegetation boundary analysis.
   - **Branch G (Geometry & Spatial Coherence):** Patch boundary detection, abrupt spectral/gradient transitions, and seam carving/stitching artifacts.
   - **Branch H (Shadow & Solar Geometry):** Dominant shadow orientation estimation, shadow-cast consistency heuristics, and optional solar azimuth metadata validation.
   - **Branch I (Elevation & Terrain Validation):** Geospatial elevation/DEM compatibility hooks, slope-terrain plausibility checks, and graceful degradation when GIS data is omitted.

3. **Explainability & Visualizations:**
   - `explainability/gradcam.py`: Layer activation mapping for CNN models.
   - `explainability/visualizations.py`: Clean, color-mapped visualization generators for all branches and comparative overlays.

4. **Reporting Service:**
   - `services/report_service.py`: Automated professional PDF report generator with Analysis ID, metadata, branch metrics, embedded charts, and scientific disclaimers.

5. **Dataset & Training Utilities:**
   - `training/generate_manipulations.py`: Reproducible synthetic manipulation generator (vegetation copy-paste, splicing, green channel boosting, patch repetition, noise injection) with full metadata tracking.
   - `training/prepare_dataset.py`: Dataset splitting, validation, and fixture generator.
   - `training/train.py`: ResNet50 & multi-branch model training pipeline.
   - `training/evaluate.py`: Evaluation metrics (Precision, Recall, F1, ROC-AUC, Confusion Matrix).

6. **Test Suite:**
   - `tests/`: Automated test suite covering image validation, ELA, FFT, LBP, structural analysis, score fusion, decision engine, pipeline consistency, and report generation.

7. **Documentation:**
   - `README.md`, `docs/ARCHITECTURE.md`, `docs/MODEL_CARD.md`, `.gitignore`, `.env.example`.

---

## 4. Engineering Assumptions & Reconstruction Strategy

1. **Multi-Branch Fusion Weights:**
   As specified in the problem statement and presentation:
   - Error Level Analysis (ELA): 22%
   - Fourier Transform (FFT): 22%
   - Vision Transformer (ViT): 18%
   - Structural Consistency: 16%
   - Texture & Geometry: 12%
   - Optional Shadow & Elevation: 10%
   *Rule:* Any unavailable branch is excluded and remaining weights are re-normalized to sum to 1.0. An unavailable branch is never treated as 0-anomaly evidence.

2. **Tri-State Decision Logic (REAL / FAKE / INCONCLUSIVE):**
   - Heuristic Score $S \in [0.0, 1.0]$: Higher score indicates greater anomaly / manipulation suspicion.
   - $S < \tau_{\text{lower}}$ (default 0.38): **REAL (Authentic Terrain Candidate)**.
   - $S \ge \tau_{\text{upper}}$ (default 0.62): **FAKE (Altered / Synthetic Terrain Candidate)**.
   - $\tau_{\text{lower}} \le S < \tau_{\text{upper}}$ OR insufficient evidence quality: **INCONCLUSIVE (Ambiguous / Conflicting Evidence)**.

3. **Model Honesty Principle:**
   - Untrained or randomly initialized neural network weights must **never** be presented as a trained deepfake detector.
   - If a fine-tuned SatVerify checkpoint is absent, the model status is explicitly displayed as *Pretrained Baseline / Uncalibrated*, and the pipeline leans on deterministic mathematical forensics (ELA, FFT, LBP, Structural) with transparent attribution.

4. **Environment Compatibility:**
   - The primary runtime leverages Anaconda Python 3.13 / Python 3.11 with Streamlit, OpenCV, NumPy, Scipy, Matplotlib, Scikit-Image, Scikit-Learn, and ReportLab.
   - Deep learning components (ResNet50/ViT) are engineered with flexible backends (TensorFlow / PyTorch / NumPy simulation fallback) so the application functions flawlessly across heterogeneous environments.

---

## 5. Reconstruction Roadmap

- **Phase 1: Recovery & Audit** (Complete) — File inspection, backups, recovery report.
- **Phase 2: Core Engineering** — Config, validation, preprocessing, ELA, FFT, LBP, structure, geometry, shadow, elevation, score fusion, decision engine.
- **Phase 3: Machine Learning & Explainability** — ResNet50 classifier, ViT analyzer, Grad-CAM, and visualization suite.
- **Phase 4: Streamlit Web Application** — Modern dark geospatial theme, interactive dashboard, multi-branch inspection, comparative views.
- **Phase 5: Reporting & Dataset Pipeline** — PDF report generation, synthetic manipulation generator, dataset fixtures.
- **Phase 6: Verification & Testing** — Automated pytest suite, syntax validation, end-to-end integration tests.
- **Phase 7: Comprehensive Documentation** — System manual, architecture guide, model card.
