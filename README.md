# RhythmMamba Edge AI - Web-based Heart Rate Monitor

> Client-side real-time remote Photoplethysmography (rPPG) using deep learning in the browser.

---

## Overview

This project ports the RhythmMamba model to run entirely in the browser using WebAssembly, enabling serverless heart rate detection from webcam video without any backend server.

---

## Technical Specifications

### Model Architecture

| Component | Specification |
|-----------|---------------|
| **Model Name** | RhythmMamba |
| **Type** | State Space Model (SSM) + Frequency Domain FFN |
| **Input Shape** | `[1, 160, 3, 128, 128]` (Batch, Frames, Channels, Height, Width) |
| **Output Shape** | `[1, 160]` (rPPG signal waveform) |
| **Parameters** | ~26M |
| **Depth** | 24 Mamba blocks |
| **Embed Dimension** | 96 |
| **MLP Ratio** | 2 |
| **State Dimension** | 48 |

### Pretrained Model

| Item | Description |
|------|-------------|
| **Weights File** | `UBFC_cross_RhythmMamba.pth` |
| **Training Dataset** | UBFC-rPPG Dataset |
| **Training Method** | Cross-dataset validation |
| **Original Framework** | PyTorch (MPS/CPU compatible) |
| **Exported Format** | ONNX (opset 17) |
| **Exported File** | `rhythm_mamba.onnx` (~26MB) |

---

## Technology Stack

### Frontend

| Technology | Version | Purpose |
|------------|---------|---------|
| **HTML5** | - | UI structure |
| **CSS3** | - | Responsive styling, animations |
| **JavaScript (ES6+)** | - | Application logic |
| **Canvas API** | - | Waveform visualization |

### AI/ML Runtime

| Technology | Version | Purpose |
|------------|---------|---------|
| **ONNX Runtime Web** | 1.17.0 | Neural network inference in browser |
| **WebAssembly (WASM)** | - | High-performance execution backend |

### Computer Vision

| Technology | Version | Purpose |
|------------|---------|---------|
| **MediaPipe Face Mesh** | 0.4 | Real-time face detection & tracking |
| **MediaPipe Camera Utils** | 0.3 | Webcam stream handling |

### Signal Processing (Pure JavaScript)

| Component | Description |
|-----------|-------------|
| **FFT** | Cooley-Tukey radix-2 implementation |
| **Bandpass Filter** | Cascaded biquad IIR filter (0.75-2.5 Hz) |
| **Detrending** | Moving average subtraction |
| **PSD Estimation** | Power spectral density via FFT |

---

## Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT STAGE                               │
├─────────────────────────────────────────────────────────────────┤
│  Webcam (640x480 @ 30fps)                                       │
│       ↓                                                          │
│  MediaPipe Face Mesh (468 landmarks)                            │
│       ↓                                                          │
│  Face ROI Extraction (1.5x bounding box)                        │
│       ↓                                                          │
│  Resize to 128x128 RGB                                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       BUFFER STAGE                               │
├─────────────────────────────────────────────────────────────────┤
│  FIFO Sliding Window Buffer                                     │
│  - Capacity: 160 frames (~5.3 seconds @ 30fps)                  │
│  - Inference triggered every 15 frames                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING STAGE                           │
├─────────────────────────────────────────────────────────────────┤
│  1. RGB Normalization: pixel / (R + G + B + ε)                  │
│  2. Temporal Standardization: (x - μ) / σ                       │
│  3. Tensor Permutation: (H,W,C) → (C,H,W)                       │
│  4. Shape: [1, 160, 3, 128, 128]                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     INFERENCE STAGE                              │
├─────────────────────────────────────────────────────────────────┤
│  ONNX Runtime Web (WASM Backend)                                │
│       ↓                                                          │
│  RhythmMamba Model                                              │
│  - Fusion Stem (temporal difference + spatial features)         │
│  - 3D Conv → Attention Mask                                     │
│  - 24x Mamba Blocks (SSM + Frequency FFN)                       │
│  - Upsample → Conv1D                                            │
│       ↓                                                          │
│  Output: rPPG Signal [1, 160]                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  POST-PROCESSING STAGE                           │
├─────────────────────────────────────────────────────────────────┤
│  1. Signal Normalization                                        │
│  2. Detrending (remove baseline drift)                          │
│  3. Bandpass Filter (0.75-2.5 Hz → 45-150 BPM)                  │
│  4. FFT → Power Spectral Density                                │
│  5. Peak Frequency Detection → Heart Rate (BPM)                 │
│  6. Temporal Smoothing (5-sample moving average)                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    VISUALIZATION STAGE                           │
├─────────────────────────────────────────────────────────────────┤
│  - Real-time HR display with animated heartbeat                 │
│  - rPPG waveform (green line on dark grid)                      │
│  - Synthetic ECG (cyan line, QRS template at peaks)             │
│  - Face detection bounding box overlay                          │
│  - Performance stats (FPS, inference time)                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration Parameters

```javascript
const CONFIG = {
    MODEL_PATH: 'rhythm_mamba.onnx',
    BUFFER_SIZE: 160,           // Frames in buffer
    INFERENCE_INTERVAL: 15,     // Run inference every N frames
    TARGET_FPS: 30,             // Target camera framerate
    FACE_SIZE: 128,             // Model input resolution
    EXPAND_RATIO: 1.5,          // Face bounding box expansion
    HR_MIN: 45,                 // Minimum detectable HR (BPM)
    HR_MAX: 150,                // Maximum detectable HR (BPM)
    FS: 30,                     // Sampling frequency (Hz)
    HR_HISTORY_SIZE: 5          // HR smoothing window
};
```

---

## Model Architecture Details

### Fusion Stem
- Parallel processing of raw frames and temporal differences
- 7x7 Conv2D with BatchNorm and ReLU
- MaxPooling for spatial downsampling
- Alpha-beta fusion of spatial and temporal features

### Mamba Block (x24)
- LayerNorm → Mamba SSM → Residual
- LayerNorm → Frequency Domain FFN → Residual
- Multi-temporal parallelization (3-path, 4-segment)

### Frequency Domain FFN
- Conv1D projection
- Matrix-based DFT (ONNX-compatible)
- Learnable frequency domain transformation
- Inverse DFT → Conv1D output projection

### State Space Model (Mamba)
- Input projection → Conv1D → SiLU activation
- Selective scan with learnable A, B, C, D parameters
- Output projection with gated residual

---

## Browser Compatibility

| Browser | Support | Notes |
|---------|---------|-------|
| Chrome 90+ | Full | Recommended |
| Edge 90+ | Full | Chromium-based |
| Firefox 89+ | Full | WebAssembly SIMD |
| Safari 15+ | Partial | Limited WASM SIMD |

---

## Performance Benchmarks

| Metric | Typical Value |
|--------|---------------|
| **Model Load Time** | 2-5 seconds |
| **Inference Time** | 500-1500 ms |
| **Face Detection** | ~30 FPS |
| **Memory Usage** | ~200-400 MB |

---

## File Structure

```
web/
├── index.html          # Main HTML with embedded CSS
├── script.js           # Application logic
├── rhythm_mamba.onnx   # Exported ONNX model (26MB)
└── README.md           # This documentation
```

---

## Deployment

### Local Development
```bash
cd web
python -m http.server 8080
# Open http://localhost:8080
```

### GitHub Pages
1. Push `web/` contents to repository
2. Enable GitHub Pages (Settings → Pages)
3. Access via `https://<username>.github.io/<repo>/`

### Static Hosting (Vercel, Netlify, etc.)
- Simply deploy the `web/` folder
- No build step required

---

## Limitations & Disclaimers

1. **Not Medical Device**: For educational/research purposes only
2. **Lighting Conditions**: Requires adequate, stable lighting
3. **Motion Sensitivity**: Subject should remain relatively still
4. **Skin Tone**: Performance may vary across skin tones
5. **Browser Resources**: Requires modern browser with WebAssembly support

---

## References

- **RhythmMamba Paper**: State Space Models for Remote Photoplethysmography
- **UBFC-rPPG Dataset**: Public dataset for rPPG research
- **ONNX Runtime Web**: https://onnxruntime.ai/docs/tutorials/web/
- **MediaPipe**: https://mediapipe.dev/

---

## License

Research/Educational Use Only

---

*Generated for RhythmMamba Edge AI Web Application*
