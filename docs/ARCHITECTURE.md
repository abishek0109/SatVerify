# SatVerify System Architecture Documentation

**Product:** SatVerify  
**Tagline:** AI Forensics for Trusted Carbon Markets — Deepfake Satellite Image Detection  
**Version:** 2.0.0  

---

## 1. High-Level Pipeline Workflow

SatVerify implements a multi-branch image forensics pipeline designed to identify digital manipulation, generative synthesis, and vegetation enhancement in satellite imagery:

```
+-------------------------------------------------------------+
|               Satellite Image Upload (RGB / NIR)            |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|              Validation & Image Preprocessing               |
|      (Format Check, EXIF Extraction, Multi-Scale Resizing)  |
+-------------------------------------------------------------+
                              |
      +-----------------------+-----------------------+
      |                       |                       |
      v                       v                       v
[Branch A: ResNet50]   [Branch B: ELA]        [Branch C: 2D FFT]
(CNN Classifier)       (Compression Residuals) (Frequency Harmonics)
      |                       |                       |
      v                       v                       v
[Branch D: ViT]        [Branch E: LBP]        [Branch F: Structural]
(Patch Transformers)   (Canopy Texture)       (Seams & Boundaries)
      |                       |                       |
      v                       v                       v
[Branch G: Geometry]   [Branch H: Shadow]     [Branch I: Elevation]
(Spatial Coherence)    (Solar Orientation)    (DEM Topography)
      |                       |                       |
      +-----------------------+-----------------------+
                              |
                              v
+-------------------------------------------------------------+
|                   Weighted Score Fusion                     |
|           (Dynamic Re-normalization over Active Signals)     |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|                     Decision Engine                         |
|     (REAL < 0.38 <= INCONCLUSIVE < 0.62 <= FAKE)            |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|         Explainable Visualizations & PDF Audit Report       |
+-------------------------------------------------------------+
```

---

## 2. Forensic Branch Specifications

### Branch A — ResNet50 Deep Learning Classifier
- **Architecture:** ResNet50 backbone (ImageNet pre-trained) with custom binary classification head:
  - `GlobalAveragePooling2D`
  - `Dropout(0.4)`
  - `Dense(128, activation='relu')`
  - `Dropout(0.2)`
  - `Dense(1, activation='sigmoid')`
- **Input:** $224 \times 224 \times 3$ float32 with Caffe channel zero-centering.
- **Model Honesty Rule:** An uncalibrated ImageNet model without verified satellite fine-tuning is marked `UNAVAILABLE` and excluded from fusion.

### Branch B — Error Level Analysis (ELA)
- **Method:** Recompresses image to JPEG at quality 90 and evaluates absolute pixel difference $\Delta(x, y) = |I(x, y) - I_{\text{recomp}}(x, y)| \times 10$.
- **Heuristic:** Genuine optical imagery displays consistent compression residual variance across the frame ($\sigma_{\text{ELA}} \approx 2\text{--}12$). Spliced regions or generative patches exhibit sharp regional variance spikes ($\sigma_{\text{ELA}} > 20$).

### Branch C — 2D Fast Fourier Transform (FFT)
- **Method:** Transforms single-channel intensity into the frequency domain:
  $F(u, v) = \mathcal{F}\{I(x, y)\}$. Magnitude spectrum is centered and log-scaled: $M(u, v) = 20 \ln(1 + |F(u, v)|)$.
- **Heuristic:** Natural imagery displays smooth $1/f^\alpha$ power-law roll-off. GAN/Diffusion generators leave periodic checkerboard spikes and elevated high-frequency noise floors ($> 25\%$ total energy).

### Branch D — Vision Transformer (ViT)
- **Architecture:** `google/vit-base-patch16-224` self-attention over $16 \times 16$ patches.
- **Role:** Identifies subtle long-range semantic inconsistencies across uniform canopy and water bodies.

### Branch E — Local Binary Pattern (LBP) Texture Analysis
- **Method:** Computes uniform LBP ($P=16, R=2$) and regional Shannon entropy $H = -\sum p_i \log_2 p_i$.
- **Heuristic:** Detects unnatural smoothing ($H < 2.0$) or cloned canopy tiles across grid patches.

### Branch F — Structural Consistency & Boundaries
- **Method:** Evaluates Sobel spatial gradients and Excess Green index ($2G - R - B$) boundary sharpness.
- **Heuristic:** Flags unnatural cookie-cutter tree cutouts and disconnected linear corridors.

### Branch G — Geometry & Spatial Coherence
- **Method:** Step-gradient profiles along rows/columns and Pearson cross-channel correlation ($r_{RG}, r_{GB}, r_{RB}$).
- **Heuristic:** Identifies rectangular splicing seams and artificial green channel saturation.

### Branch H — Shadow and Solar-Geometry Validation
- **Method:** Segments low-luminance candidate shadows, measures directional gradient coherence, and compares against known solar azimuth if metadata is present.

### Branch I — Elevation and Terrain Validation
- **Method:** Cross-references canopy masks against Digital Elevation Models (DEM) to flag physical impossibilities (e.g. dense mature forest on $>65^\circ$ vertical cliffs).

---

## 3. Weighted Fusion & Calibration Logic

| Forensic Category | Base Weight | Active Branches Included |
|---|---:|---|
| Error Level Analysis (ELA) | 22% | Branch B |
| Fourier Frequency (FFT) | 22% | Branch C |
| Deep Learning (ViT / CNN) | 18% | Branch A / Branch D (When calibrated) |
| Structural Consistency | 16% | Branch F |
| Texture & Geometry | 12% | Mean of Branch E & G |
| Shadow & Elevation | 10% | Mean of Branch H & I |

### Dynamic Re-normalization Formula
Let $\mathcal{A}$ denote the set of active, available branches. The effective weight for branch $i \in \mathcal{A}$ is:
$$w_i^{\text{eff}} = \frac{w_i^{\text{base}}}{\sum_{j \in \mathcal{A}} w_j^{\text{base}}}$$
The fused suspicion score is:
$$S = \sum_{i \in \mathcal{A}} w_i^{\text{eff}} \cdot s_i \quad \in [0.0, 1.0]$$

---

## 4. Tri-State Decision Policy

1. **REAL (Authentic Candidate):** $S < 0.38$. Low forensic suspicion across all active branches.
2. **FAKE (Altered Candidate):** $S \ge 0.62$. Significant anomalies detected.
3. **INCONCLUSIVE:** $0.38 \le S < 0.62$ OR fewer than 2 active forensic branches OR high internal branch contradiction. Requires human auditor inspection.
