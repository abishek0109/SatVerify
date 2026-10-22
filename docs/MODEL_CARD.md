# SatVerify Model Card & Forensic Calibration Standard

## Model Overview
- **Model Name:** SatVerify Multi-Branch Forensic Pipeline
- **Release Version:** 2.0.0
- **Model Type:** Hybrid Physics-Signal Forensic Ensemble with Optional Deep Learning Transfer Learning
- **Primary Domain:** Optical Satellite Imagery Authentication for Carbon Credit Markets

## Intended Use
- **Primary Use Case:** Screening uploaded satellite scenes in carbon sequestration and afforestation projects to detect digital manipulations (copy-pasting vegetation, generative inpainting, artificial green boosting, and scene splicing).
- **Intended Users:** Carbon market auditors, ESG project evaluators, remote-sensing researchers, compliance officers.
- **Out of Scope:** Legal courtroom verification without ground-truth physical corroboration, automated trade execution, regulatory certification.

## Factors, Metrics, and Evaluation
- **Forensic Branches:**
  - Error Level Analysis (ELA)
  - 2D Fast Fourier Transform (FFT) Magnitude Harmonics
  - Local Binary Pattern (LBP) Micro-Texture Entropy
  - Structural Gradient Disparity & Vegetation Boundary Continuity
  - Cross-Channel Color Correlation & Seam Step Detection
  - Shadow Directional Coherence
- **Classification Outputs:**
  - `REAL`: Fused Suspicion Score $< 0.38$
  - `FAKE`: Fused Suspicion Score $\ge 0.62$
  - `INCONCLUSIVE`: Suspicion Score in $[0.38, 0.62)$ or insufficient evidence count ($< 2$ branches).

## Model Honesty & Safety Guidelines
- **Untrained Model Policy:** An ImageNet pretrained backbone without verified satellite deepfake fine-tuning is marked `UNAVAILABLE` and excluded from score fusion. An untrained classification head is never presented as an authenticity detector.
- **False Positive Hazards:** Naturally uniform monoculture crop plantations, still water bodies, and orthorectification tile boundaries can trigger texture or seam warnings.
- **Scientific Disclaimer:** SatVerify provides automated heuristic screening. Results must be validated by remote-sensing analysts and physical plot inspection.
