// -------------------------------------------------------------
// ACCENT COLOR STUDIO - CLIENT LOGIC & NEURAL VISUALIZER
// -------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
    // State
    let currentBaseHex = '#141414';
    let currentPrediction = null;
    let currentActivations = null;
    let modelMetrics = null;

    // DOM Elements
    const mainNav = document.getElementById('mainNav');
    const tabPanes = document.querySelectorAll('.tab-pane');
    
    // Inputs & Swatches
    const colorPickerInput = document.getElementById('colorPickerInput');
    const hexTextInput = document.getElementById('hexTextInput');
    const baseColorPreviewBox = document.getElementById('baseColorPreviewBox');
    const variationsGrid = document.getElementById('variationsGrid');
    const presetBtns = document.querySelectorAll('.preset-btn');
    
    // Background Glow
    const bgGlowPrimary = document.getElementById('bgGlowPrimary');
    const bgGlowAccent = document.getElementById('bgGlowAccent');

    // UI Playground Elements
    const previewProgress = document.getElementById('previewProgress');
    
    // Image Analyzer Elements
    const imageDropzone = document.getElementById('imageDropzone');
    const imageFileInput = document.getElementById('imageFileInput');
    const imageResultsContainer = document.getElementById('imageResultsContainer');
    const uploadedImagePreview = document.getElementById('uploadedImagePreview');
    const palettesList = document.getElementById('palettesList');

    // Metrics & Retrain
    const valLossMSE = document.getElementById('valLossMSE');
    const valMAE = document.getElementById('valMAE');
    const valDuration = document.getElementById('valDuration');
    const retrainBtn = document.getElementById('retrainBtn');

    // Export Modal & Toast
    const openExportBtn = document.getElementById('openExportBtn');
    const closeExportBtn = document.getElementById('closeExportBtn');
    const exportModal = document.getElementById('exportModal');
    const exportCodeBlock = document.getElementById('exportCodeBlock');
    const copyExportCodeBtn = document.getElementById('copyExportCodeBtn');
    const toastNotification = document.getElementById('toastNotification');
    const toastMessage = document.getElementById('toastMessage');

    // Canvas Contexts
    const netCanvas = document.getElementById('networkCanvas');
    const netCtx = netCanvas ? netCanvas.getContext('2d') : null;
    const lossCanvas = document.getElementById('lossCanvas');
    const lossCtx = lossCanvas ? lossCanvas.getContext('2d') : null;

    // ---------------------------------------------------------
    // 1. Navigation Setup
    // ---------------------------------------------------------
    mainNav.addEventListener('click', (e) => {
        const btn = e.target.closest('.nav-btn');
        if (!btn) return;

        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        const targetTab = btn.dataset.tab;
        tabPanes.forEach(pane => {
            if (pane.id === targetTab) {
                pane.classList.add('active');
                if (targetTab === 'studioTab') resizeCanvas(netCanvas);
                if (targetTab === 'metricsTab') resizeCanvas(lossCanvas);
            } else {
                pane.classList.remove('active');
            }
        });
    });

    // ---------------------------------------------------------
    // 2. Color Input Handlers & API Calls
    // ---------------------------------------------------------
    function updateColorFromPicker(hex) {
        if (!hex.startsWith('#')) hex = '#' + hex;
        currentBaseHex = hex;
        colorPickerInput.value = hex;
        hexTextInput.value = hex.replace('#', '').toUpperCase();
        baseColorPreviewBox.style.backgroundColor = hex;
        
        fetchPrediction(hex);
    }

    colorPickerInput.addEventListener('input', (e) => {
        updateColorFromPicker(e.target.value);
    });

    hexTextInput.addEventListener('input', (e) => {
        let val = e.target.value.trim().replace('#', '');
        if (val.length === 6) {
            updateColorFromPicker('#' + val);
        }
    });

    presetBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            updateColorFromPicker(btn.dataset.color);
        });
    });

    async function fetchPrediction(hex) {
        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ hex: hex })
            });
            const data = await res.json();
            currentPrediction = data;
            currentActivations = data.activations;

            renderVariations(data.variations);
            updateThemeVariables(data);
            drawNetworkGraph();
        } catch (err) {
            console.error("Prediction error:", err);
        }
    }

    function renderVariations(variations) {
        variationsGrid.innerHTML = variations.map(v => `
            <div class="swatch-card" data-hex="${v.hex}">
                <div class="swatch-info">
                    <div class="swatch-color-pill" style="background-color: ${v.hex};"></div>
                    <div class="swatch-details">
                        <span class="swatch-label">${v.label}</span>
                        <span class="swatch-hex">${v.hex.toUpperCase()}</span>
                    </div>
                </div>
                <i class="fa-solid fa-copy copy-icon"></i>
            </div>
        `).join('');

        variationsGrid.querySelectorAll('.swatch-card').forEach(card => {
            card.addEventListener('click', () => {
                copyToClipboard(card.dataset.hex);
            });
        });
    }

    function updateThemeVariables(data) {
        const primaryHex = data.variations[0].hex;
        const tintHex = data.variations[1].hex;
        const shadeHex = data.variations[2].hex;

        document.documentElement.style.setProperty('--base-color', currentBaseHex);
        document.documentElement.style.setProperty('--accent-primary', primaryHex);
        document.documentElement.style.setProperty('--accent-tint', tintHex);
        document.documentElement.style.setProperty('--accent-shade', shadeHex);

        bgGlowPrimary.style.background = currentBaseHex;
        bgGlowAccent.style.background = primaryHex;
    }

    // ---------------------------------------------------------
    // 3. Neural Network Canvas Visualizer
    // ---------------------------------------------------------
    let nodePositions = [];

    function resizeCanvas(canvas) {
        if (!canvas) return;
        const rect = canvas.parentElement.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;
        if (canvas === netCanvas) drawNetworkGraph();
        if (canvas === lossCanvas && modelMetrics) drawLossCurve();
    }

    window.addEventListener('resize', () => {
        resizeCanvas(netCanvas);
        resizeCanvas(lossCanvas);
    });

    function drawNetworkGraph() {
        if (!netCtx || !netCanvas) return;
        const w = netCanvas.width;
        const h = netCanvas.height;
        netCtx.clearRect(0, 0, w, h);

        const layers = [3, 16, 16, 3];
        const layerX = [w * 0.12, w * 0.38, w * 0.64, w * 0.88];
        nodePositions = [];

        // Compute Positions
        layers.forEach((count, lIdx) => {
            const x = layerX[lIdx];
            const nodes = [];
            const spacing = (h - 60) / (count + 1);

            for (let i = 0; i < count; i++) {
                const y = 30 + spacing * (i + 1);
                nodes.push({ x, y, layer: lIdx, index: i });
            }
            nodePositions.push(nodes);
        });

        // Draw Connections
        netCtx.lineWidth = 0.8;
        for (let l = 0; l < nodePositions.length - 1; l++) {
            const currentLayer = nodePositions[l];
            const nextLayer = nodePositions[l + 1];

            currentLayer.forEach(n1 => {
                nextLayer.forEach(n2 => {
                    netCtx.beginPath();
                    netCtx.moveTo(n1.x, n1.y);
                    netCtx.lineTo(n2.x, n2.y);
                    netCtx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
                    netCtx.stroke();
                });
            });
        }

        // Extract Node Activations
        let actValues = [[], [], [], []];
        if (currentActivations) {
            actValues[0] = currentActivations.input[0] || [0.5, 0.5, 0.5];
            actValues[1] = currentActivations.layer1[0] || new Array(16).fill(0.2);
            actValues[2] = currentActivations.layer2[0] || new Array(16).fill(0.2);
            actValues[3] = currentActivations.output[0] || [0.5, 0.5, 0.5];
        }

        // Draw Nodes
        nodePositions.forEach((layerNodes, lIdx) => {
            layerNodes.forEach((node, nIdx) => {
                const val = actValues[lIdx][nIdx] || 0.1;
                const radius = lIdx === 0 || lIdx === 3 ? 10 : 6;
                
                let fillColor = '#6366f1';
                if (lIdx === 0) fillColor = currentBaseHex;
                if (lIdx === 1) fillColor = `rgba(99, 102, 241, ${Math.max(0.2, val)})`;
                if (lIdx === 2) fillColor = `rgba(168, 85, 247, ${Math.max(0.2, val)})`;
                if (lIdx === 3 && currentPrediction) fillColor = currentPrediction.variations[0].hex;

                // Node Outer Glow
                netCtx.beginPath();
                netCtx.arc(node.x, node.y, radius + 4, 0, Math.PI * 2);
                netCtx.fillStyle = fillColor;
                netCtx.globalAlpha = 0.25;
                netCtx.fill();
                netCtx.globalAlpha = 1.0;

                // Node Body
                netCtx.beginPath();
                netCtx.arc(node.x, node.y, radius, 0, Math.PI * 2);
                netCtx.fillStyle = fillColor;
                netCtx.strokeStyle = '#ffffff';
                netCtx.lineWidth = 1.5;
                netCtx.fill();
                netCtx.stroke();
            });
        });
    }

    // ---------------------------------------------------------
    // 4. Image Drag & Drop Analyzer
    // ---------------------------------------------------------
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        imageDropzone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        imageDropzone.addEventListener(eventName, () => imageDropzone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        imageDropzone.addEventListener(eventName, () => imageDropzone.classList.remove('dragover'), false);
    });

    imageDropzone.addEventListener('click', () => imageFileInput.click());

    imageDropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) handleImageUpload(files[0]);
    });

    imageFileInput.addEventListener('change', (e) => {
        if (e.target.files.length) handleImageUpload(e.target.files[0]);
    });

    async function handleImageUpload(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            uploadedImagePreview.src = e.target.result;
        };
        reader.readAsDataURL(file);

        const formData = new FormData();
        formData.append('image', file);

        try {
            imageResultsContainer.classList.remove('hidden');
            palettesList.innerHTML = `<div class="loading-state"><i class="fa-solid fa-spinner fa-spin"></i> Running NumPy K-Means...</div>`;

            const res = await fetch('/api/extract', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();

            if (data.palettes) {
                renderExtractedPalettes(data.palettes);
            }
        } catch (err) {
            console.error("Image upload error:", err);
        }
    }

    function renderExtractedPalettes(palettes) {
        palettesList.innerHTML = palettes.map((p, idx) => `
            <div class="extracted-palette-card">
                <div class="extracted-palette-header">
                    <div class="extracted-color-chip" style="background-color: ${p.base_hex};"></div>
                    <span>Cluster #${idx + 1}: ${p.base_hex.toUpperCase()}</span>
                </div>
                <div class="extracted-swatches">
                    ${p.variations.map(v => `
                        <div class="swatch-card" style="flex: 1;" data-hex="${v.hex}">
                            <div class="swatch-info">
                                <div class="swatch-color-pill" style="background-color: ${v.hex};"></div>
                                <div class="swatch-details">
                                    <span class="swatch-label">${v.label}</span>
                                    <span class="swatch-hex">${v.hex.toUpperCase()}</span>
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `).join('');

        palettesList.querySelectorAll('.swatch-card').forEach(card => {
            card.addEventListener('click', () => copyToClipboard(card.dataset.hex));
        });
    }

    // ---------------------------------------------------------
    // 5. Metrics & Loss Curve Visualizer
    // ---------------------------------------------------------
    async function loadMetrics() {
        try {
            const res = await fetch('/api/metrics');
            modelMetrics = await res.json();
            displayMetrics(modelMetrics);
            drawLossCurve();
        } catch (err) {
            console.error("Metrics error:", err);
        }
    }

    function displayMetrics(metrics) {
        if (!metrics) return;
        valLossMSE.textContent = metrics.final_loss_mse.toFixed(6);
        valMAE.textContent = metrics.mean_absolute_error.toFixed(6);
        valDuration.textContent = metrics.training_duration_seconds + 's';
    }

    function drawLossCurve() {
        if (!lossCtx || !lossCanvas || !modelMetrics || !modelMetrics.loss_history) return;
        const w = lossCanvas.width;
        const h = lossCanvas.height;
        lossCtx.clearRect(0, 0, w, h);

        const history = modelMetrics.loss_history;
        const maxLoss = Math.max(...history);
        const minLoss = Math.min(...history);

        const padding = 40;
        const graphW = w - padding * 2;
        const graphH = h - padding * 2;

        // Grid lines
        lossCtx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
        lossCtx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {
            const y = padding + (graphH / 4) * i;
            lossCtx.beginPath();
            lossCtx.moveTo(padding, y);
            lossCtx.lineTo(w - padding, y);
            lossCtx.stroke();
        }

        // Draw Line
        lossCtx.beginPath();
        history.forEach((loss, idx) => {
            const x = padding + (graphW / (history.length - 1)) * idx;
            const normY = (loss - minLoss) / (maxLoss - minLoss || 1);
            const y = (padding + graphH) - (normY * graphH);

            if (idx === 0) lossCtx.moveTo(x, y);
            else lossCtx.lineTo(x, y);
        });

        lossCtx.strokeStyle = '#38bdf8';
        lossCtx.lineWidth = 2.5;
        lossCtx.stroke();

        // Area Fill
        lossCtx.lineTo(w - padding, padding + graphH);
        lossCtx.lineTo(padding, padding + graphH);
        lossCtx.closePath();

        const gradient = lossCtx.createLinearGradient(0, padding, 0, padding + graphH);
        gradient.addColorStop(0, 'rgba(56, 189, 248, 0.25)');
        gradient.addColorStop(1, 'rgba(56, 189, 248, 0.0)');
        lossCtx.fillStyle = gradient;
        lossCtx.fill();
    }

    retrainBtn.addEventListener('click', async () => {
        retrainBtn.disabled = true;
        retrainBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Calibrating...`;

        try {
            const res = await fetch('/api/retrain', { method: 'POST' });
            modelMetrics = await res.json();
            displayMetrics(modelMetrics);
            drawLossCurve();
            showToast("NumPy Model Calibrated Successfully!");
        } catch (err) {
            console.error("Retrain error:", err);
        } finally {
            retrainBtn.disabled = false;
            retrainBtn.innerHTML = `<i class="fa-solid fa-rotate-right"></i> Re-Calibrate Model`;
        }
    });

    // ---------------------------------------------------------
    // 6. Export Tokens & Modal
    // ---------------------------------------------------------
    openExportBtn.addEventListener('click', () => {
        exportModal.classList.remove('hidden');
        renderExportCode('css');
    });

    closeExportBtn.addEventListener('click', () => {
        exportModal.classList.add('hidden');
    });

    document.querySelectorAll('.modal-tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.modal-tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            renderExportCode(btn.dataset.format);
        });
    });

    function renderExportCode(format) {
        if (!currentPrediction) return;

        const base = currentBaseHex;
        const p = currentPrediction.variations[0].hex;
        const t = currentPrediction.variations[1].hex;
        const s = currentPrediction.variations[2].hex;

        let code = '';
        if (format === 'css') {
            code = `:root {\n  --color-base: ${base};\n  --color-accent-primary: ${p};\n  --color-accent-tint: ${t};\n  --color-accent-shade: ${s};\n}`;
        } else if (format === 'tailwind') {
            code = `module.exports = {\n  theme: {\n    extend: {\n      colors: {\n        brand: {\n          base: '${base}',\n          accent: '${p}',\n          tint: '${t}',\n          shade: '${s}'\n        }\n      }\n    }\n  }\n}`;
        } else if (format === 'json') {
            code = JSON.stringify({ base, primary_match: p, tint: t, shade: s }, null, 2);
        }

        exportCodeBlock.textContent = code;
    }

    copyExportCodeBtn.addEventListener('click', () => {
        copyToClipboard(exportCodeBlock.textContent);
    });

    function copyToClipboard(text) {
        navigator.clipboard.writeText(text);
        showToast(`Copied ${text} to clipboard!`);
    }

    function showToast(msg) {
        toastMessage.textContent = msg;
        toastNotification.classList.remove('hidden');
        setTimeout(() => toastNotification.classList.add('hidden'), 2500);
    }

    // Initialize App
    setTimeout(() => {
        resizeCanvas(netCanvas);
        resizeCanvas(lossCanvas);
        updateColorFromPicker('#141414');
        loadMetrics();
    }, 100);
});
