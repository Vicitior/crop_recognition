# 🌾 Crop Growth Stage Recognition System

[English README](README_EN.md) | [中文说明文档](README.md)

An intelligent crop growth stage recognition system powered by **CLIP + LoRA + Three Core Innovation Modules**, classifying **15 growth stages** across **Corn, Wheat, and Cotton**. Includes full support for both a **Web Application (Gradio)** and a **Fully Offline On-Device Android App (Edge AI)**.

---

## 🚀 Key Innovations & Benchmarks

1. **Confidence-aware Routing** (`models/confidence_router.py`): Replaces rigid hard-routing with dynamic soft-routing multi-branch fusion when predictions are ambiguous, preventing misclassification between morphologically similar stages (e.g., Jointing stage in Corn vs. Wheat).
2. **Phenology-aware Relation Modeling** (`models/phenology_graph.py`): Utilizes Gaussian adjacency matrices and Graph Convolutional Networks (GCN) to model temporal continuity across growth stages.
3. **Adaptive LoRA Rank Allocation** (`models/adaptive_lora.py`): Dynamically allocates LoRA ranks based on visual complexity (Corn: `rank=4`, Wheat: `rank=8`, Cotton: `rank=16`).

### Model Performance Comparison

| Model Architecture | Validation Accuracy | Description |
| :--- | :--- | :--- |
| **Innovation Model (All 3 Combined)** | **93.53%** | Trained from scratch, **Best Performance** |
| Innovation Model (Baseline Init) | 92.23% | Fine-tuned from baseline (84.73%) |
| Baseline CLIP ViT-L/14@336 + LoRA | 84.73% | Standard fine-tuned model |

---

## 🌾 Supported Crops & 15 Growth Stages

* **🌽 Corn (5 Stages)**: Seedling → Jointing → Tasseling → Grain-Filling → Maturity
* **🌾 Wheat (5 Stages)**: Seedling → Tillering → Jointing → Heading → Maturity
* **🌸 Cotton (5 Stages)**: Seedling → Squaring → Flowering → Boll Setting → Boll Opening

---

## 📱 On-Device Offline Android App (Edge AI)

The project includes a production-ready **Android Studio project** (`android_app/`) capable of **100% offline edge AI inference** without any internet connection.

### Features
* **LoRA Weight Fusion & Vectorized Graph**: Fuses LoRA weights into base CLIP ViT linear layers and vectorizes routing logic via tensor masks.
* **INT8 Dynamic Quantization**: Quantizes FP32 ONNX model (1.16GB) down to **293MB (-74.9% compression)**.
* **Zero Java Heap OOM Memory Overhead**: Streams model copy in 64KB chunks and initializes ONNX Runtime via `mmap` C++ file path.
* **Instant Chinese / English Bilingual Switch**: Seamless `🌐 English / 🌐 中文` toggle for UI headers, stage descriptions, and agronomic management advice (Water, Fertilizer, Pest Control).
* **Material Design 3 Glassmorphism UI**: High-aesthetic emerald theme with Top-3 candidate matching progress bars and runtime camera permission check.

### Exporting ONNX & Quantization

```bash
# 1. Fuse LoRA weights & export end-to-end ONNX model
python scripts/export_onnx.py

# 2. Quantize FP32 ONNX to INT8 (1.16GB -> 293MB)
python scripts/quantize_onnx.py

# 3. Verify MSE residual & Top-1 category alignment
python scripts/verify_onnx.py
```

### Running the Android App in Android Studio

1. Launch **Android Studio**, click `Open`, and select the `android_app` folder (`crop_recognition/android_app`).
2. Wait for Gradle Sync to complete.
3. Connect your Android device, then click `Run 'app'` (or `Build` -> `Build APK(s)` to generate the standalone `.apk`).

---

## 💻 Web Interface Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Launch Gradio Web UI (Default port 7860)
python app.py

# Specify port and create public share link
python app.py --port 7860 --share
```

---

## 📂 Project Structure

```text
crop_recognition/
├── app.py                          # Gradio Web Application
├── README.md                       # Chinese Documentation
├── README_EN.md                    # English Documentation
├── PROJECT_MEMORY.md               # Architecture & Development Memory Archive
├── models/                         # Core Model Definitions
│   ├── clip_classifier.py          # Zero-shot CLIP/SigLIP Classifier
│   ├── confidence_router.py        # Innovation #1: Confidence Router
│   ├── phenology_graph.py          # Innovation #2: Phenology Graph GCN
│   ├── adaptive_lora.py            # Innovation #3: Adaptive LoRA Rank
│   └── growth_stages.py            # Crop Stages & Agronomic Knowledge
├── scripts/                        # Training, Export & Verification Scripts
│   ├── export_onnx.py              # ONNX Fusion & Exporter
│   ├── quantize_onnx.py            # INT8 Dynamic Quantizer
│   ├── verify_onnx.py              # PyTorch vs ONNX Verifier
│   ├── train_innovations.py        # Training Script for Innovations
│   └── ...
├── saved_models/                   # Trained Weights & ONNX Models
└── android_app/                    # Android Studio Edge AI Project
    ├── build.gradle.kts
    └── app/src/main/
        ├── assets/crop_model_int8.onnx  # Quantized Model (293MB)
        └── java/com/example/croprecognition/
            ├── ImageUtils.kt             # Image Normalization & FloatBuffer
            ├── CropRecognitionEngine.kt # ONNX Runtime Offline Engine
            ├── AgronomicKnowledge.kt    # Bilingual Advice Database
            └── MainActivity.kt          # Material3 UI & Bilingual Switch
```

---

## 📄 License

This project is released for academic, learning, and research purposes.
