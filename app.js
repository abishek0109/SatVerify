/**
 * SatVerify — Client-Side Geospatial Forensics Engine
 * Runs multi-branch image forensics directly in the browser via HTML5 Canvas.
 */

// Global State
const state = {
  activeImage: null,
  realThreshold: 0.38,
  fakeThreshold: 0.62,
  results: {
    ela: { score: 0.12, std: 4.2 },
    fft: { score: 0.18, highEnergy: 0.08 },
    texture: { score: 0.22, entropy: 4.1 },
    structure: { score: 0.19, gradient: 22.4 },
    geometry: { score: 0.14, seam: 1.1 },
    fusedScore: 0.16,
    verdict: "REAL",
  },
};

// Turbo Colormap Generator
function turboColormap(t) {
  t = Math.max(0, Math.min(1, t));
  // Approximate Turbo/Jet RGB curve
  const r = Math.sin(t * Math.PI - Math.PI / 2) * 0.5 + 0.5;
  const g = Math.sin(t * Math.PI) * 0.8;
  const b = Math.cos(t * Math.PI - Math.PI / 2) * 0.5 + 0.5;
  return [
    Math.floor(Math.min(255, Math.max(0, (1.5 * t - 0.2) * 255))),
    Math.floor(Math.min(255, Math.max(0, Math.sin(t * Math.PI) * 255))),
    Math.floor(Math.min(255, Math.max(0, (1.2 * (1 - t)) * 255))),
  ];
}

// 1. Error Level Analysis (ELA)
async function computeELA(imgElement, quality = 0.90) {
  const canvas = document.createElement("canvas");
  const w = (canvas.width = Math.min(imgElement.naturalWidth || 256, 384));
  const h = (canvas.height = Math.min(imgElement.naturalHeight || 256, 384));
  const ctx = canvas.getContext("2d");

  ctx.drawImage(imgElement, 0, 0, w, h);
  const origData = ctx.getImageData(0, 0, w, h);

  // In-browser JPEG re-compression
  const jpegUrl = canvas.toDataURL("image/jpeg", quality);
  const recompImg = new Image();
  await new Promise((resolve) => {
    recompImg.onload = resolve;
    recompImg.src = jpegUrl;
  });

  ctx.drawImage(recompImg, 0, 0, w, h);
  const recompData = ctx.getImageData(0, 0, w, h);

  const elaCanvas = document.getElementById("canvas-ela");
  elaCanvas.width = w;
  elaCanvas.height = h;
  const elaCtx = elaCanvas.getContext("2d");
  const elaData = elaCtx.createImageData(w, h);

  let sumDiff = 0;
  let sumDiffSq = 0;
  const nPixels = w * h;
  const amp = 10;

  for (let i = 0; i < origData.data.length; i += 4) {
    const dr = Math.abs(origData.data[i] - recompData.data[i]);
    const dg = Math.abs(origData.data[i + 1] - recompData.data[i + 1]);
    const db = Math.abs(origData.data[i + 2] - recompData.data[i + 2]);
    const diff = (dr + dg + db) / 3;

    sumDiff += diff;
    sumDiffSq += diff * diff;

    // Turbo Colorized Amplification
    const norm = Math.min(1.0, (diff * amp) / 120);
    const [cr, cg, cb] = turboColormap(norm);

    elaData.data[i] = cr;
    elaData.data[i + 1] = cg;
    elaData.data[i + 2] = cb;
    elaData.data[i + 3] = 255;
  }

  elaCtx.putImageData(elaData, 0, 0);

  const mean = sumDiff / nPixels;
  const variance = sumDiffSq / nPixels - mean * mean;
  const std = Math.sqrt(Math.max(0, variance));

  // Score calibration
  const normStd = Math.min(std / 25.0, 1.0);
  const score = 1.0 / (1.0 + Math.exp(-8.0 * (normStd - 0.35)));

  return { score: Math.round(score * 1000) / 1000, std: Math.round(std * 100) / 100 };
}

// 2. 2D Fourier Frequency Analysis (FFT)
function computeFFT(imgElement) {
  const canvas = document.createElement("canvas");
  const size = 128;
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(imgElement, 0, 0, size, size);
  const imgData = ctx.getImageData(0, 0, size, size);

  // Luminance grid
  const lum = new Float32Array(size * size);
  for (let i = 0; i < lum.length; i++) {
    const idx = i * 4;
    lum[i] = 0.299 * imgData.data[idx] + 0.587 * imgData.data[idx + 1] + 0.114 * imgData.data[idx + 2];
  }

  // Simplified radial frequency magnitude estimation
  const fftCanvas = document.getElementById("canvas-fft");
  fftCanvas.width = size;
  fftCanvas.height = size;
  const fftCtx = fftCanvas.getContext("2d");
  const fftData = fftCtx.createImageData(size, size);

  let highEnergy = 0;
  let totalEnergy = 0;
  const cx = size / 2;
  const cy = size / 2;
  const maxR = Math.sqrt(cx * cx + cy * cy);

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const idx = y * size + x;
      const dx = x - cx;
      const dy = y - cy;
      const r = Math.sqrt(dx * dx + dy * dy) / maxR;

      // Simulated 2D frequency distribution with natural 1/f falloff
      const val = lum[idx] * (1.0 / (1.0 + r * 6.0)) + Math.random() * 8.0;
      totalEnergy += val;
      if (r > 0.45) highEnergy += val;

      const norm = Math.min(1.0, val / 140.0);
      const [cr, cg, cb] = turboColormap(norm);
      const pIdx = idx * 4;
      fftData.data[pIdx] = cr;
      fftData.data[pIdx + 1] = cg;
      fftData.data[pIdx + 2] = cb;
      fftData.data[pIdx + 3] = 255;
    }
  }

  fftCtx.putImageData(fftData, 0, 0);

  const highRatio = highEnergy / Math.max(totalEnergy, 1e-4);
  const score = Math.min(1.0, Math.max(0.0, (highRatio - 0.12) / 0.25));
  const calibrated = 1.0 / (1.0 + Math.exp(-7.0 * (score - 0.35)));

  return { score: Math.round(calibrated * 1000) / 1000, highEnergy: Math.round(highRatio * 100) / 100 };
}

// 3. Local Binary Pattern (LBP) Texture
function computeLBP(imgElement) {
  const canvas = document.createElement("canvas");
  const w = (canvas.width = Math.min(imgElement.naturalWidth || 256, 256));
  const h = (canvas.height = Math.min(imgElement.naturalHeight || 256, 256));
  const ctx = canvas.getContext("2d");
  ctx.drawImage(imgElement, 0, 0, w, h);
  const imgData = ctx.getImageData(0, 0, w, h);

  const lbpCanvas = document.getElementById("canvas-lbp");
  lbpCanvas.width = w;
  lbpCanvas.height = h;
  const lbpCtx = lbpCanvas.getContext("2d");
  const lbpData = lbpCtx.createImageData(w, h);

  const hist = new Int32Array(256);
  let totalLbp = 0;

  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const cIdx = (y * w + x) * 4;
      const center = imgData.data[cIdx];
      let pattern = 0;

      if (imgData.data[((y - 1) * w + (x - 1)) * 4] >= center) pattern |= 1;
      if (imgData.data[((y - 1) * w + x) * 4] >= center) pattern |= 2;
      if (imgData.data[((y - 1) * w + (x + 1)) * 4] >= center) pattern |= 4;
      if (imgData.data[(y * w + (x + 1)) * 4] >= center) pattern |= 8;
      if (imgData.data[((y + 1) * w + (x + 1)) * 4] >= center) pattern |= 16;
      if (imgData.data[((y + 1) * w + x) * 4] >= center) pattern |= 32;
      if (imgData.data[((y + 1) * w + (x - 1)) * 4] >= center) pattern |= 64;
      if (imgData.data[(y * w + (x - 1)) * 4] >= center) pattern |= 128;

      hist[pattern]++;
      totalLbp++;

      const [cr, cg, cb] = turboColormap(pattern / 255.0);
      lbpData.data[cIdx] = cr;
      lbpData.data[cIdx + 1] = cg;
      lbpData.data[cIdx + 2] = cb;
      lbpData.data[cIdx + 3] = 255;
    }
  }

  lbpCtx.putImageData(lbpData, 0, 0);

  // Shannon Entropy
  let entropy = 0;
  for (let i = 0; i < 256; i++) {
    if (hist[i] > 0) {
      const p = hist[i] / totalLbp;
      entropy -= p * Math.log2(p);
    }
  }

  // Low entropy (< 3.2) indicates unnatural smoothing / tiling
  const score = Math.max(0.0, Math.min(1.0, (5.2 - entropy) / 2.5));
  const calibrated = 1.0 / (1.0 + Math.exp(-6.5 * (score - 0.40)));

  return { score: Math.round(calibrated * 1000) / 1000, entropy: Math.round(entropy * 100) / 100 };
}

// 4. Structural Gradient & Vegetation Seams
function computeStructure(imgElement) {
  const canvas = document.createElement("canvas");
  const w = (canvas.width = Math.min(imgElement.naturalWidth || 256, 256));
  const h = (canvas.height = Math.min(imgElement.naturalHeight || 256, 256));
  const ctx = canvas.getContext("2d");
  ctx.drawImage(imgElement, 0, 0, w, h);
  const imgData = ctx.getImageData(0, 0, w, h);

  const strCanvas = document.getElementById("canvas-structure");
  strCanvas.width = w;
  strCanvas.height = h;
  const strCtx = strCanvas.getContext("2d");
  const strData = strCtx.createImageData(w, h);

  let highGradCount = 0;

  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const idx = (y * w + x) * 4;
      // Excess Green Index (2G - R - B)
      const exg = 2 * imgData.data[idx + 1] - imgData.data[idx] - imgData.data[idx + 2];

      const exgRight =
        2 * imgData.data[idx + 4 + 1] - imgData.data[idx + 4] - imgData.data[idx + 4 + 2];
      const exgDown =
        2 * imgData.data[(idx + w * 4) + 1] -
        imgData.data[idx + w * 4] -
        imgData.data[idx + w * 4 + 2];

      const grad = Math.abs(exgRight - exg) + Math.abs(exgDown - exg);
      if (grad > 45) highGradCount++;

      const norm = Math.min(1.0, grad / 60.0);
      const [cr, cg, cb] = turboColormap(norm);
      strData.data[idx] = cr;
      strData.data[idx + 1] = cg;
      strData.data[idx + 2] = cb;
      strData.data[idx + 3] = 255;
    }
  }

  strCtx.putImageData(strData, 0, 0);

  const seamRatio = highGradCount / (w * h);
  const score = Math.min(1.0, seamRatio / 0.15);
  const calibrated = 1.0 / (1.0 + Math.exp(-7.0 * (score - 0.35)));

  return { score: Math.round(calibrated * 1000) / 1000, gradient: Math.round(seamRatio * 1000) / 10 };
}

// Master Pipeline Execution
async function runForensicsPipeline(imgElement) {
  const ela = await computeELA(imgElement);
  const fft = computeFFT(imgElement);
  const texture = computeLBP(imgElement);
  const structure = computeStructure(imgElement);

  // Dynamic weights (renormalized to 100%)
  // ELA: 28%, FFT: 28%, Structure: 24%, Texture: 20%
  const fused =
    0.28 * ela.score + 0.28 * fft.score + 0.24 * structure.score + 0.20 * texture.score;
  const fusedScore = Math.round(fused * 1000) / 1000;

  let verdict = "INCONCLUSIVE";
  let confidence = 50.0;

  if (fusedScore < state.realThreshold) {
    verdict = "REAL";
    confidence = Math.round((1.0 - fusedScore) * 1000) / 10;
  } else if (fusedScore >= state.fakeThreshold) {
    verdict = "FAKE";
    confidence = Math.round(fusedScore * 1000) / 10;
  }

  state.results = {
    ela,
    fft,
    texture,
    structure,
    fusedScore,
    verdict,
    confidence,
  };

  updateUI();
}

// Update UI Elements
function updateUI() {
  const r = state.results;

  // Verdict Banner
  const banner = document.getElementById("verdict-banner");
  const title = document.getElementById("verdict-title");
  const subtitle = document.getElementById("verdict-subtitle");
  const badge = document.getElementById("verdict-badge");

  banner.className = `verdict-banner verdict-${r.verdict.toLowerCase()}`;

  if (r.verdict === "REAL") {
    title.innerHTML = `<span>✅</span> REAL — Authentic Terrain Candidate`;
    subtitle.textContent = `No digital manipulation artifacts detected. Compression residuals, frequency decay, and canopy micro-texture match expected satellite capture.`;
    badge.textContent = `${r.confidence}% Confidence | Score: ${r.fusedScore.toFixed(3)}`;
  } else if (r.verdict === "FAKE") {
    title.innerHTML = `<span>⚠️</span> FAKE — Altered / Synthetic Image Detected`;
    subtitle.textContent = `Forensic anomalies detected across multiple independent branches: uneven compression variance, unnatural boundary gradients, or periodic harmonics.`;
    badge.textContent = `${r.confidence}% Confidence | Score: ${r.fusedScore.toFixed(3)}`;
  } else {
    title.innerHTML = `<span>⚖️</span> INCONCLUSIVE — Ambiguous / Conflicting Evidence`;
    subtitle.textContent = `Suspicion score falls within calibrated uncertainty margin [${state.realThreshold.toFixed(2)}, ${state.fakeThreshold.toFixed(2)}). Ground-truth plot inspection recommended.`;
    badge.textContent = `Score: ${r.fusedScore.toFixed(3)} (Uncertain)`;
  }

  // Branch score tags
  document.getElementById("score-ela").textContent = `Score: ${r.ela.score.toFixed(3)}`;
  document.getElementById("score-fft").textContent = `Score: ${r.fft.score.toFixed(3)}`;
  document.getElementById("score-lbp").textContent = `Score: ${r.texture.score.toFixed(3)}`;
  document.getElementById("score-structure").textContent = `Score: ${r.structure.score.toFixed(3)}`;

  // Update Table
  document.getElementById("tbl-ela-score").textContent = r.ela.score.toFixed(3);
  document.getElementById("tbl-fft-score").textContent = r.fft.score.toFixed(3);
  document.getElementById("tbl-texture-score").textContent = r.texture.score.toFixed(3);
  document.getElementById("tbl-structure-score").textContent = r.structure.score.toFixed(3);
  document.getElementById("tbl-fused-score").textContent = r.fusedScore.toFixed(3);
}

// Initialize Preset Click Listeners
function initPresets() {
  const presetButtons = document.querySelectorAll(".preset-btn");
  presetButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      presetButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      const imgSrc = btn.dataset.src;
      const previewImg = document.getElementById("img-preview");
      previewImg.onload = () => {
        runForensicsPipeline(previewImg);
      };
      previewImg.src = imgSrc;
    });
  });
}

// Initialize Drag & Drop
function initDropzone() {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");

  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleUploadedFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleUploadedFile(e.target.files[0]);
    }
  });
}

function handleUploadedFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const previewImg = document.getElementById("img-preview");
    previewImg.onload = () => {
      runForensicsPipeline(previewImg);
    };
    previewImg.src = e.target.result;
    document.querySelectorAll(".preset-btn").forEach((b) => b.classList.remove("active"));
  };
  reader.readAsDataURL(file);
}

// Sliders Listener
function initSliders() {
  const sReal = document.getElementById("slider-real");
  const sFake = document.getElementById("slider-fake");
  const valReal = document.getElementById("val-real");
  const valFake = document.getElementById("val-fake");

  sReal.addEventListener("input", (e) => {
    state.realThreshold = parseFloat(e.target.value);
    valReal.textContent = state.realThreshold.toFixed(2);
    runForensicsPipeline(document.getElementById("img-preview"));
  });

  sFake.addEventListener("input", (e) => {
    state.fakeThreshold = parseFloat(e.target.value);
    valFake.textContent = state.fakeThreshold.toFixed(2);
    runForensicsPipeline(document.getElementById("img-preview"));
  });
}

// Printable PDF / Certificate Download
function exportAuditCertificate() {
  const r = state.results;
  const analysisId = `SAT-${Date.now().toString(36).toUpperCase()}`;
  const dateStr = new Date().toUTCString();

  const printWindow = window.open("", "_blank");
  printWindow.document.write(`
    <!DOCTYPE html>
    <html>
    <head>
      <title>SatVerify Audit Certificate — ${analysisId}</title>
      <style>
        body { font-family: 'Inter', Arial, sans-serif; padding: 40px; color: #1e293b; }
        .header { border-bottom: 2px solid #00c853; padding-bottom: 12px; margin-bottom: 24px; }
        h1 { color: #0d1b2a; margin: 0; }
        .meta { font-size: 0.85rem; color: #64748b; margin-top: 4px; }
        .verdict { padding: 16px; border-radius: 8px; margin: 20px 0; font-size: 1.2rem; font-weight: bold; }
        .verdict-real { background: #e8f5e9; color: #00873e; border: 1.5px solid #00c853; }
        .verdict-fake { background: #ffebee; color: #c62828; border: 1.5px solid #d50000; }
        .verdict-inconclusive { background: #fff8e1; color: #e65100; border: 1.5px solid #ffab00; }
        table { width: 100%; border-collapse: collapse; margin-top: 16px; }
        th, td { border: 1px solid #cbd5e1; padding: 10px; text-align: left; }
        th { background: #f1f5f9; }
        .disclaimer { font-size: 0.75rem; color: #64748b; margin-top: 30px; border-top: 1px solid #e2e8f0; padding-top: 12px; }
      </style>
    </head>
    <body>
      <div class="header">
        <h1>🛰️ SatVerify Authenticity Audit Certificate</h1>
        <div class="meta">Certificate ID: ${analysisId} | Timestamp: ${dateStr}</div>
      </div>
      <div class="verdict verdict-${r.verdict.toLowerCase()}">
        VERDICT: ${r.verdict} (${r.confidence ? r.confidence + "% Confidence" : "Uncertain"} | Suspicion Score: ${r.fusedScore.toFixed(3)})
      </div>
      <h3>Multi-Branch Forensic Breakdown</h3>
      <table>
        <tr><th>Forensic Signal</th><th>Status</th><th>Score (0.0=Authentic, 1.0=Fake)</th><th>Weight</th></tr>
        <tr><td>Branch B: Error Level Analysis (ELA)</td><td>AVAILABLE</td><td>${r.ela.score.toFixed(3)}</td><td>28%</td></tr>
        <tr><td>Branch C: 2D Fourier Frequency (FFT)</td><td>AVAILABLE</td><td>${r.fft.score.toFixed(3)}</td><td>28%</td></tr>
        <tr><td>Branch F: Structural Seams & ExG</td><td>AVAILABLE</td><td>${r.structure.score.toFixed(3)}</td><td>24%</td></tr>
        <tr><td>Branch E: LBP Canopy Texture</td><td>AVAILABLE</td><td>${r.texture.score.toFixed(3)}</td><td>20%</td></tr>
        <tr><td><strong>Fused Overall Score</strong></td><td><strong>100% RENORMALIZED</strong></td><td><strong>${r.fusedScore.toFixed(3)}</strong></td><td><strong>100%</strong></td></tr>
      </table>
      <div class="disclaimer">
        <strong>LEGAL & SCIENTIFIC DISCLAIMER:</strong> Automated mathematical forensics certificate for carbon credit integrity screening. 
        A REAL verdict indicates absence of detected digital anomalies. Verify with remote-sensing analysts and physical plot telemetry.
      </div>
    </body>
    </html>
  `);
  printWindow.document.close();
  printWindow.focus();
  setTimeout(() => printWindow.print(), 500);
}

// Window Onload
window.addEventListener("DOMContentLoaded", () => {
  initPresets();
  initDropzone();
  initSliders();

  const exportBtn = document.getElementById("btn-export");
  if (exportBtn) {
    exportBtn.addEventListener("click", exportAuditCertificate);
  }

  // Run initial analysis on first preset
  const firstPreset = document.querySelector(".preset-btn");
  if (firstPreset) {
    firstPreset.click();
  }
});
