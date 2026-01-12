/**
 * RhythmMamba Edge AI Heart Rate Monitor
 * Client-side real-time rPPG inference using ONNX Runtime Web
 *
 * Architecture:
 * - Camera -> MediaPipe Face Mesh -> ROI Crop -> Frame Buffer
 * - Every 15 frames: Buffer -> Preprocessing -> ONNX Inference -> Signal Processing -> HR
 *
 * Dependencies: onnxruntime-web, @mediapipe/face_mesh
 */

// ============================================================
// Configuration
// ============================================================
const CONFIG = {
    MODEL_PATH: 'rhythm_mamba.onnx',
    BUFFER_SIZE: 160,           // Number of frames in buffer
    INFERENCE_INTERVAL: 15,     // Run inference every N frames
    TARGET_FPS: 30,
    FACE_SIZE: 128,             // Model input size
    EXPAND_RATIO: 1.5,          // Face box expansion
    HR_MIN: 45,                 // Minimum HR (BPM)
    HR_MAX: 150,                // Maximum HR (BPM)
    FS: 30,                     // Sampling rate (fps)
    HR_HISTORY_SIZE: 5          // Number of HR values to average
};

// ============================================================
// Global State
// ============================================================
let onnxSession = null;
let faceMesh = null;
let camera = null;
let isRunning = false;

const state = {
    frameBuffer: [],            // FIFO buffer of face crops
    frameCount: 0,
    lastBbox: null,
    lastHr: 0,
    lastRppg: null,
    lastEcg: null,
    hrHistory: [],
    inferenceCount: 0,
    lastInferenceTime: 0,
    fpsHistory: [],
    lastFrameTime: performance.now()
};

// ============================================================
// DOM Elements
// ============================================================
const elements = {
    video: document.getElementById('videoElement'),
    overlay: document.getElementById('overlayCanvas'),
    rppgCanvas: document.getElementById('rppgCanvas'),
    ecgCanvas: document.getElementById('ecgCanvas'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    loadingProgress: document.getElementById('loadingProgress'),
    statusBadge: document.getElementById('statusBadge'),
    statusText: document.getElementById('statusText'),
    hrValue: document.getElementById('hrValue'),
    hrStatus: document.getElementById('hrStatus'),
    bufferValue: document.getElementById('bufferValue'),
    bufferProgress: document.getElementById('bufferProgress'),
    fpsDisplay: document.getElementById('fpsDisplay'),
    inferenceDisplay: document.getElementById('inferenceDisplay'),
    statInference: document.getElementById('statInference'),
    statFps: document.getElementById('statFps'),
    statFace: document.getElementById('statFace'),
    statCount: document.getElementById('statCount'),
    startBtn: document.getElementById('startBtn'),
    resetBtn: document.getElementById('resetBtn')
};

// ============================================================
// Signal Processing (Pure JavaScript)
// ============================================================

/**
 * Simple FFT implementation (Cooley-Tukey radix-2)
 */
class FFT {
    static transform(real, imag) {
        const n = real.length;
        if (n <= 1) return;

        // Bit-reversal permutation
        for (let i = 0, j = 0; i < n; i++) {
            if (j > i) {
                [real[i], real[j]] = [real[j], real[i]];
                [imag[i], imag[j]] = [imag[j], imag[i]];
            }
            let m = n >> 1;
            while (m >= 1 && j >= m) {
                j -= m;
                m >>= 1;
            }
            j += m;
        }

        // Cooley-Tukey FFT
        for (let len = 2; len <= n; len *= 2) {
            const halfLen = len / 2;
            const angleStep = -2 * Math.PI / len;
            for (let i = 0; i < n; i += len) {
                let angle = 0;
                for (let j = 0; j < halfLen; j++) {
                    const cos = Math.cos(angle);
                    const sin = Math.sin(angle);
                    const evenIdx = i + j;
                    const oddIdx = i + j + halfLen;
                    const tr = real[oddIdx] * cos - imag[oddIdx] * sin;
                    const ti = real[oddIdx] * sin + imag[oddIdx] * cos;
                    real[oddIdx] = real[evenIdx] - tr;
                    imag[oddIdx] = imag[evenIdx] - ti;
                    real[evenIdx] += tr;
                    imag[evenIdx] += ti;
                    angle += angleStep;
                }
            }
        }
    }

    static computePSD(signal, fs) {
        // Pad to next power of 2
        const n = Math.pow(2, Math.ceil(Math.log2(signal.length)));
        const real = new Float32Array(n);
        const imag = new Float32Array(n);

        // Apply Hanning window and copy signal
        for (let i = 0; i < signal.length; i++) {
            const window = 0.5 * (1 - Math.cos(2 * Math.PI * i / (signal.length - 1)));
            real[i] = signal[i] * window;
        }

        this.transform(real, imag);

        // Compute power spectral density
        const psd = new Float32Array(n / 2);
        const freqs = new Float32Array(n / 2);
        for (let i = 0; i < n / 2; i++) {
            psd[i] = (real[i] * real[i] + imag[i] * imag[i]) / n;
            freqs[i] = i * fs / n;
        }

        return { freqs, psd };
    }
}

/**
 * Simple bandpass filter using cascaded biquad sections
 */
class BandpassFilter {
    constructor(lowFreq, highFreq, fs, order = 2) {
        this.filters = [];
        const nyq = fs / 2;
        const low = lowFreq / nyq;
        const high = highFreq / nyq;

        // Create simple IIR bandpass
        for (let i = 0; i < order; i++) {
            this.filters.push(this.createBiquad(low, high));
        }
    }

    createBiquad(low, high) {
        const bw = high - low;
        const f0 = (low + high) / 2;
        const Q = f0 / bw;
        const w0 = 2 * Math.PI * f0;
        const alpha = Math.sin(w0) / (2 * Q);

        const b0 = alpha;
        const b1 = 0;
        const b2 = -alpha;
        const a0 = 1 + alpha;
        const a1 = -2 * Math.cos(w0);
        const a2 = 1 - alpha;

        return {
            b: [b0 / a0, b1 / a0, b2 / a0],
            a: [1, a1 / a0, a2 / a0],
            x: [0, 0],
            y: [0, 0]
        };
    }

    filter(signal) {
        const output = new Float32Array(signal.length);
        for (let i = 0; i < signal.length; i++) {
            let x = signal[i];
            for (const bq of this.filters) {
                const y = bq.b[0] * x + bq.b[1] * bq.x[0] + bq.b[2] * bq.x[1]
                    - bq.a[1] * bq.y[0] - bq.a[2] * bq.y[1];
                bq.x[1] = bq.x[0];
                bq.x[0] = x;
                bq.y[1] = bq.y[0];
                bq.y[0] = y;
                x = y;
            }
            output[i] = x;
        }
        return output;
    }

    static apply(signal, lowFreq, highFreq, fs) {
        const filter = new BandpassFilter(lowFreq, highFreq, fs);
        // Forward pass
        const forward = filter.filter(signal);
        // Backward pass for zero-phase filtering
        const filter2 = new BandpassFilter(lowFreq, highFreq, fs);
        const reversed = new Float32Array(signal.length);
        for (let i = 0; i < signal.length; i++) {
            reversed[i] = forward[signal.length - 1 - i];
        }
        const backward = filter2.filter(reversed);
        const result = new Float32Array(signal.length);
        for (let i = 0; i < signal.length; i++) {
            result[i] = backward[signal.length - 1 - i];
        }
        return result;
    }
}

/**
 * Detrend signal using simple high-pass filter
 */
function detrend(signal) {
    const n = signal.length;
    if (n < 2) return signal;

    // Simple moving average subtraction
    const windowSize = Math.min(30, Math.floor(n / 4));
    const result = new Float32Array(n);

    for (let i = 0; i < n; i++) {
        let sum = 0;
        let count = 0;
        for (let j = Math.max(0, i - windowSize); j < Math.min(n, i + windowSize + 1); j++) {
            sum += signal[j];
            count++;
        }
        result[i] = signal[i] - sum / count;
    }

    return result;
}

/**
 * Calculate heart rate from rPPG signal
 */
function calculateHeartRate(signal, fs = CONFIG.FS) {
    if (signal.length < 64) return 0;

    // Detrend
    let processed = detrend(signal);

    // Bandpass filter (0.75-2.5 Hz = 45-150 BPM)
    processed = BandpassFilter.apply(processed, 0.75, 2.5, fs);

    // Normalize
    const mean = processed.reduce((a, b) => a + b, 0) / processed.length;
    const std = Math.sqrt(processed.reduce((a, b) => a + (b - mean) ** 2, 0) / processed.length);
    if (std > 1e-6) {
        for (let i = 0; i < processed.length; i++) {
            processed[i] = (processed[i] - mean) / std;
        }
    }

    // Compute PSD
    const { freqs, psd } = FFT.computePSD(processed, fs);

    // Find peak in valid frequency range
    const minFreq = CONFIG.HR_MIN / 60;
    const maxFreq = CONFIG.HR_MAX / 60;
    let maxPsd = 0;
    let peakFreq = 0;

    for (let i = 0; i < freqs.length; i++) {
        if (freqs[i] >= minFreq && freqs[i] <= maxFreq) {
            if (psd[i] > maxPsd) {
                maxPsd = psd[i];
                peakFreq = freqs[i];
            }
        }
    }

    return peakFreq * 60;
}

/**
 * Generate synthetic ECG waveform from rPPG
 */
function generateSyntheticECG(rppgSignal, fs, hr) {
    const n = rppgSignal.length;
    const ecg = new Float32Array(n);

    if (hr <= 0 || n < 30) return ecg;

    // Filter the rPPG signal
    let filtered = BandpassFilter.apply(rppgSignal, 0.75, 2.5, fs);

    // Normalize
    const mean = filtered.reduce((a, b) => a + b, 0) / filtered.length;
    const std = Math.sqrt(filtered.reduce((a, b) => a + (b - mean) ** 2, 0) / filtered.length);
    if (std > 1e-6) {
        for (let i = 0; i < n; i++) {
            filtered[i] = (filtered[i] - mean) / std;
        }
    }

    // Find peaks
    const minDist = Math.floor(fs * 60 / hr * 0.6);
    const peaks = findPeaks(filtered, minDist, 0.3);

    // Generate QRS template
    const templateWidth = Math.floor(fs * 0.6);
    const template = generateQRSTemplate(templateWidth);
    const halfWidth = Math.floor(templateWidth / 2);

    // Place templates at peaks
    for (const peak of peaks) {
        for (let i = 0; i < templateWidth; i++) {
            const idx = peak - halfWidth + i;
            if (idx >= 0 && idx < n) {
                ecg[idx] = Math.max(ecg[idx], template[i]);
            }
        }
    }

    // Add subtle baseline
    for (let i = 0; i < n; i++) {
        ecg[i] += 0.02 * Math.sin(2 * Math.PI * 0.1 * i / fs);
    }

    return ecg;
}

function generateQRSTemplate(width) {
    const template = new Float32Array(width);
    const t = new Float32Array(width);

    for (let i = 0; i < width; i++) {
        t[i] = (2 * i / (width - 1)) - 1;
    }

    for (let i = 0; i < width; i++) {
        const x = t[i];
        // P wave
        const pWave = 0.12 * Math.exp(-((x + 0.55) ** 2) / 0.015);
        // Q wave
        const qWave = -0.08 * Math.exp(-((x + 0.12) ** 2) / 0.008);
        // R wave
        const rWave = 1.0 * Math.exp(-((x) ** 2) / 0.008);
        // S wave
        const sWave = -0.15 * Math.exp(-((x - 0.12) ** 2) / 0.008);
        // T wave
        const tWave = 0.25 * Math.exp(-((x - 0.45) ** 2) / 0.025);

        template[i] = pWave + qWave + rWave + sWave + tWave;
    }

    return template;
}

function findPeaks(signal, minDist, minProminence) {
    const peaks = [];
    for (let i = 1; i < signal.length - 1; i++) {
        if (signal[i] > signal[i - 1] && signal[i] > signal[i + 1] && signal[i] > minProminence) {
            if (peaks.length === 0 || i - peaks[peaks.length - 1] >= minDist) {
                peaks.push(i);
            }
        }
    }
    return peaks;
}

// ============================================================
// ONNX Model Loading
// ============================================================

async function loadONNXModel() {
    updateLoadingProgress('Loading ONNX model...');

    try {
        // Configure ONNX Runtime
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/';

        // Create session
        onnxSession = await ort.InferenceSession.create(CONFIG.MODEL_PATH, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });

        console.log('ONNX model loaded successfully');
        console.log('Input names:', onnxSession.inputNames);
        console.log('Output names:', onnxSession.outputNames);

        return true;
    } catch (error) {
        console.error('Failed to load ONNX model:', error);
        updateLoadingProgress(`Model load failed: ${error.message}`);
        return false;
    }
}

// ============================================================
// MediaPipe Face Mesh
// ============================================================

async function initFaceMesh() {
    updateLoadingProgress('Initializing Face Mesh...');

    return new Promise((resolve, reject) => {
        faceMesh = new FaceMesh({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4/${file}`;
            }
        });

        faceMesh.setOptions({
            maxNumFaces: 1,
            refineLandmarks: true,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });

        faceMesh.onResults(onFaceMeshResults);

        faceMesh.initialize()
            .then(() => {
                console.log('Face Mesh initialized');
                resolve(true);
            })
            .catch((error) => {
                console.error('Face Mesh init failed:', error);
                reject(error);
            });
    });
}

function onFaceMeshResults(results) {
    if (!isRunning) return;

    const canvas = elements.overlay;
    const ctx = canvas.getContext('2d');

    // Set canvas size
    canvas.width = elements.video.videoWidth;
    canvas.height = elements.video.videoHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
        const landmarks = results.multiFaceLandmarks[0];
        const bbox = computeFaceBbox(landmarks, canvas.width, canvas.height);

        if (bbox) {
            state.lastBbox = bbox;
            elements.statFace.textContent = 'Detected';
            elements.statFace.style.color = '#00ff88';

            // Draw face box (mirrored)
            ctx.strokeStyle = '#00ff88';
            ctx.lineWidth = 2;
            const mirroredX1 = canvas.width - bbox.x2;
            const mirroredX2 = canvas.width - bbox.x1;
            ctx.strokeRect(mirroredX1, bbox.y1, mirroredX2 - mirroredX1, bbox.y2 - bbox.y1);

            // Crop and process face
            processFaceFrame(bbox);
        }
    } else {
        elements.statFace.textContent = 'Not found';
        elements.statFace.style.color = '#ff4444';
    }
}

function computeFaceBbox(landmarks, width, height) {
    let xMin = Infinity, xMax = -Infinity;
    let yMin = Infinity, yMax = -Infinity;

    for (const lm of landmarks) {
        const x = lm.x * width;
        const y = lm.y * height;
        xMin = Math.min(xMin, x);
        xMax = Math.max(xMax, x);
        yMin = Math.min(yMin, y);
        yMax = Math.max(yMax, y);
    }

    const boxW = xMax - xMin;
    const boxH = yMax - yMin;
    const cx = (xMin + xMax) / 2;
    const cy = (yMin + yMax) / 2;

    const newW = boxW * CONFIG.EXPAND_RATIO;
    const newH = boxH * CONFIG.EXPAND_RATIO;

    return {
        x1: Math.max(0, Math.floor(cx - newW / 2)),
        y1: Math.max(0, Math.floor(cy - newH / 2)),
        x2: Math.min(width, Math.floor(cx + newW / 2)),
        y2: Math.min(height, Math.floor(cy + newH / 2))
    };
}

// ============================================================
// Frame Processing
// ============================================================

function processFaceFrame(bbox) {
    // Create offscreen canvas for cropping
    const cropCanvas = document.createElement('canvas');
    cropCanvas.width = CONFIG.FACE_SIZE;
    cropCanvas.height = CONFIG.FACE_SIZE;
    const ctx = cropCanvas.getContext('2d');

    // Crop and resize
    const cropW = bbox.x2 - bbox.x1;
    const cropH = bbox.y2 - bbox.y1;

    ctx.drawImage(
        elements.video,
        bbox.x1, bbox.y1, cropW, cropH,
        0, 0, CONFIG.FACE_SIZE, CONFIG.FACE_SIZE
    );

    // Get pixel data (RGBA)
    const imageData = ctx.getImageData(0, 0, CONFIG.FACE_SIZE, CONFIG.FACE_SIZE);
    const pixels = imageData.data;

    // Extract RGB values (ignore alpha)
    const rgb = new Float32Array(CONFIG.FACE_SIZE * CONFIG.FACE_SIZE * 3);
    for (let i = 0, j = 0; i < pixels.length; i += 4, j += 3) {
        rgb[j] = pixels[i];       // R
        rgb[j + 1] = pixels[i + 1]; // G
        rgb[j + 2] = pixels[i + 2]; // B
    }

    // Add to buffer
    state.frameBuffer.push(rgb);
    if (state.frameBuffer.length > CONFIG.BUFFER_SIZE) {
        state.frameBuffer.shift();
    }

    // Update buffer display
    updateBufferDisplay();

    // Increment frame count and check for inference
    state.frameCount++;

    // Calculate FPS
    const now = performance.now();
    const dt = now - state.lastFrameTime;
    state.lastFrameTime = now;
    state.fpsHistory.push(1000 / dt);
    if (state.fpsHistory.length > 30) state.fpsHistory.shift();

    const avgFps = state.fpsHistory.reduce((a, b) => a + b, 0) / state.fpsHistory.length;
    elements.fpsDisplay.textContent = `FPS: ${avgFps.toFixed(1)}`;
    elements.statFps.textContent = `${avgFps.toFixed(1)}`;

    // Run inference every N frames if buffer is ready
    if (state.frameCount % CONFIG.INFERENCE_INTERVAL === 0 &&
        state.frameBuffer.length >= CONFIG.BUFFER_SIZE) {
        runInference();
    }
}

function updateBufferDisplay() {
    const count = state.frameBuffer.length;
    const progress = (count / CONFIG.BUFFER_SIZE) * 100;

    elements.bufferValue.textContent = `${count} / ${CONFIG.BUFFER_SIZE}`;
    elements.bufferProgress.style.width = `${progress}%`;

    if (count >= CONFIG.BUFFER_SIZE) {
        elements.hrStatus.textContent = 'Analyzing...';
    } else {
        elements.hrStatus.textContent = `Calibrating... ${Math.round(progress)}%`;
    }
}

// ============================================================
// ONNX Inference
// ============================================================

async function runInference() {
    if (!onnxSession || state.frameBuffer.length < CONFIG.BUFFER_SIZE) return;

    const startTime = performance.now();

    try {
        // Prepare input tensor [1, D, C, H, W]
        const inputTensor = prepareInputTensor();

        // Run inference
        const feeds = { [onnxSession.inputNames[0]]: inputTensor };
        const results = await onnxSession.run(feeds);

        // Get output
        const outputName = onnxSession.outputNames[0];
        const outputData = results[outputName].data;

        // Normalize output
        const rppg = normalizeSignal(Array.from(outputData));
        state.lastRppg = rppg;

        // Calculate heart rate
        const hr = calculateHeartRate(rppg, CONFIG.FS);

        if (hr > CONFIG.HR_MIN && hr < CONFIG.HR_MAX) {
            state.hrHistory.push(hr);
            if (state.hrHistory.length > CONFIG.HR_HISTORY_SIZE) {
                state.hrHistory.shift();
            }
            state.lastHr = state.hrHistory.reduce((a, b) => a + b, 0) / state.hrHistory.length;
        }

        // Generate synthetic ECG
        state.lastEcg = generateSyntheticECG(rppg, CONFIG.FS, state.lastHr);

        // Update displays
        const inferenceTime = performance.now() - startTime;
        state.lastInferenceTime = inferenceTime;
        state.inferenceCount++;

        updateHRDisplay();
        updateWaveformDisplays();
        updateStatsDisplay(inferenceTime);

        console.log(`[Inference #${state.inferenceCount}] HR: ${state.lastHr.toFixed(1)} BPM | Time: ${inferenceTime.toFixed(1)}ms`);

    } catch (error) {
        console.error('Inference error:', error);
    }
}

function prepareInputTensor() {
    const D = CONFIG.BUFFER_SIZE;
    const C = 3;
    const H = CONFIG.FACE_SIZE;
    const W = CONFIG.FACE_SIZE;

    // Collect frames [D, H*W*C]
    const frames = state.frameBuffer.slice(-D);

    // Step 1: Normalize RGB by sum (as in Python)
    const normalized = frames.map(frame => {
        const result = new Float32Array(frame.length);
        for (let i = 0; i < frame.length; i += 3) {
            const sum = frame[i] + frame[i + 1] + frame[i + 2] + 1e-6;
            result[i] = frame[i] / sum;
            result[i + 1] = frame[i + 1] / sum;
            result[i + 2] = frame[i + 2] / sum;
        }
        return result;
    });

    // Step 2: Compute temporal mean and std
    let sum = 0, sumSq = 0, count = 0;
    for (const frame of normalized) {
        for (const val of frame) {
            sum += val;
            sumSq += val * val;
            count++;
        }
    }
    const mean = sum / count;
    const std = Math.sqrt(sumSq / count - mean * mean) + 1e-6;

    // Step 3: Standardize
    const standardized = normalized.map(frame => {
        const result = new Float32Array(frame.length);
        for (let i = 0; i < frame.length; i++) {
            result[i] = (frame[i] - mean) / std;
        }
        return result;
    });

    // Step 4: Reshape to [1, D, C, H, W] from [D, H*W*C]
    // Canvas data is H, W, C format, we need to transpose to C, H, W
    const tensorData = new Float32Array(1 * D * C * H * W);

    for (let d = 0; d < D; d++) {
        const frame = standardized[d];
        for (let c = 0; c < C; c++) {
            for (let h = 0; h < H; h++) {
                for (let w = 0; w < W; w++) {
                    // Source: frame[(h * W + w) * 3 + c]
                    // Dest: tensor[d * (C*H*W) + c * (H*W) + h * W + w]
                    const srcIdx = (h * W + w) * C + c;
                    const dstIdx = d * (C * H * W) + c * (H * W) + h * W + w;
                    tensorData[dstIdx] = frame[srcIdx];
                }
            }
        }
    }

    return new ort.Tensor('float32', tensorData, [1, D, C, H, W]);
}

function normalizeSignal(signal) {
    const mean = signal.reduce((a, b) => a + b, 0) / signal.length;
    const std = Math.sqrt(signal.reduce((a, b) => a + (b - mean) ** 2, 0) / signal.length);
    return signal.map(v => (v - mean) / (std + 1e-6));
}

// ============================================================
// Visualization
// ============================================================

function updateHRDisplay() {
    if (state.lastHr > 0) {
        elements.hrValue.textContent = Math.round(state.lastHr);
        elements.hrStatus.textContent = getHRStatus(state.lastHr);

        // Animate heartbeat based on HR
        const beatInterval = 60000 / state.lastHr;
        document.querySelector('.hr-icon').style.animationDuration = `${beatInterval}ms`;
    }
}

function getHRStatus(hr) {
    if (hr < 60) return 'Resting';
    if (hr < 100) return 'Normal';
    if (hr < 120) return 'Elevated';
    return 'High';
}

function updateWaveformDisplays() {
    drawWaveform(elements.rppgCanvas, state.lastRppg, '#00ff88');
    drawECGWaveform(elements.ecgCanvas, state.lastEcg);
}

function drawWaveform(canvas, signal, color) {
    if (!signal || signal.length < 2) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width = canvas.offsetWidth * 2;
    const height = canvas.height = canvas.offsetHeight * 2;
    ctx.scale(2, 2);

    const w = width / 2;
    const h = height / 2;

    // Background (改為純白或淺灰)
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(0, 0, w, h);

    // Draw grid (改為淺色格線)
    ctx.strokeStyle = '#e9ecef';
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 30) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
    }
    for (let y = 0; y < h; y += 20) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
    }

    // Use last N samples
    const samples = signal.slice(-w);

    // Normalize for display
    const min = Math.min(...samples);
    const max = Math.max(...samples);
    const range = max - min || 1;

    // Draw waveform (使用深一點的綠色或原本顏色)
    // 如果傳入的 color 太亮，這裡可以硬編碼成深一點的顏色，例如 '#00b894'
    ctx.strokeStyle = '#00b894'; 
    ctx.lineWidth = 2;
    ctx.beginPath();

    for (let i = 0; i < samples.length; i++) {
        const x = (i / samples.length) * w;
        const normalized = (samples[i] - min) / range;
        const y = h - normalized * h * 0.8 - h * 0.1;

        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    ctx.stroke();
}

function drawECGWaveform(canvas, signal) {
    if (!signal || signal.length < 2) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width = canvas.offsetWidth * 2;
    const height = canvas.height = canvas.offsetHeight * 2;
    ctx.scale(2, 2);

    const w = width / 2;
    const h = height / 2;

    // Medical Graph Paper Background (醫療心電圖紙風格)
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, w, h);

    // Grid (細微的粉紅色或淺紅色格線)
    ctx.strokeStyle = '#ffe0e0';
    ctx.lineWidth = 0.5;
    for (let x = 0; x < w; x += 25) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
    }
    for (let y = 0; y < h; y += 20) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
    }

    // Use last N samples
    const samples = signal.slice(-w);

    // Normalize
    const min = Math.min(...samples);
    const max = Math.max(...samples);
    const range = max - min || 1;

    // Draw ECG line (改為深藍色或黑色，模仿真實儀器)
    ctx.strokeStyle = '#0984e3'; // 深藍色
    ctx.lineWidth = 1.5;
    ctx.beginPath();

    for (let i = 0; i < samples.length; i++) {
        const x = (i / samples.length) * w;
        const normalized = (samples[i] - min) / range;
        const y = h - normalized * h * 0.8 - h * 0.1;

        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    ctx.stroke();

    // 移除 Glow effect (在淺色背景下發光效果會讓線條變模糊)
    // ctx.shadowColor = ... (移除)
}

function updateStatsDisplay(inferenceTime) {
    elements.inferenceDisplay.textContent = `Inference: ${inferenceTime.toFixed(0)}ms`;
    elements.statInference.textContent = `${inferenceTime.toFixed(0)} ms`;
    elements.statCount.textContent = state.inferenceCount;
}

// ============================================================
// Camera Control
// ============================================================

async function startCamera() {
    updateLoadingProgress('Starting camera...');

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                frameRate: { ideal: CONFIG.TARGET_FPS }
            }
        });

        elements.video.srcObject = stream;
        await elements.video.play();

        console.log('Camera started');
        return true;
    } catch (error) {
        console.error('Camera error:', error);
        updateLoadingProgress(`Camera error: ${error.message}`);
        return false;
    }
}

async function startProcessing() {
    isRunning = true;
    processFrame();
}

async function processFrame() {
    if (!isRunning) return;

    try {
        await faceMesh.send({ image: elements.video });
    } catch (error) {
        console.error('Frame processing error:', error);
    }

    requestAnimationFrame(processFrame);
}

function stopProcessing() {
    isRunning = false;
}

function resetState() {
    state.frameBuffer = [];
    state.frameCount = 0;
    state.lastHr = 0;
    state.lastRppg = null;
    state.lastEcg = null;
    state.hrHistory = [];
    state.inferenceCount = 0;

    elements.hrValue.textContent = '--';
    elements.hrStatus.textContent = 'Waiting for data...';
    elements.bufferValue.textContent = '0 / 160';
    elements.bufferProgress.style.width = '0%';
    elements.statCount.textContent = '0';

    // Clear canvases
    [elements.rppgCanvas, elements.ecgCanvas].forEach(canvas => {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    });
}

// ============================================================
// UI Helpers
// ============================================================

function updateLoadingProgress(text) {
    elements.loadingProgress.textContent = text;
}

function setStatus(status, text) {
    elements.statusBadge.className = `status-badge ${status}`;
    elements.statusText.textContent = text;
}

function hideLoading() {
    elements.loadingOverlay.classList.add('hidden');
}

// ============================================================
// Initialization
// ============================================================

async function init() {
    console.log('Initializing RhythmMamba Edge AI...');

    // Load ONNX model
    const modelLoaded = await loadONNXModel();
    if (!modelLoaded) {
        setStatus('error', 'Model load failed');
        return;
    }

    // Initialize Face Mesh
    try {
        await initFaceMesh();
    } catch (error) {
        setStatus('error', 'Face Mesh failed');
        return;
    }

    // Start camera
    const cameraStarted = await startCamera();
    if (!cameraStarted) {
        setStatus('error', 'Camera failed');
        return;
    }

    // Setup complete
    hideLoading();
    setStatus('ready', 'Ready');
    elements.startBtn.disabled = false;

    console.log('Initialization complete!');
}

// Event Listeners
elements.startBtn.addEventListener('click', () => {
    if (!isRunning) {
        startProcessing();
        elements.startBtn.innerHTML = '<span>&#x23F8;</span> Pause';
        setStatus('ready', 'Running');
    } else {
        stopProcessing();
        elements.startBtn.innerHTML = '<span>&#x25B6;</span> Start';
        setStatus('ready', 'Paused');
    }
});

elements.resetBtn.addEventListener('click', () => {
    resetState();
});

// Start initialization when page loads
window.addEventListener('load', init);
