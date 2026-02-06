# 🎭 Deepfake Detection System

**Optimized for:** Intel i7-13650HX + NVIDIA RTX 4060 + 24GB RAM

A state-of-the-art deepfake detection system using EfficientNet with GPU acceleration.

---

## 🚀 Quick Start

### 1. Install Dependencies

```powershell
# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\activate

# Install PyTorch with CUDA support for RTX 4060
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

### 2. Verify GPU Setup

```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### 3. Run Demo

```powershell
python detect.py --demo
```

---

## 📁 Project Structure

```
deepfake_detection/
├── config.py          # Hardware-optimized configuration
├── model.py           # EfficientNet-based detection model
├── dataset.py         # Data loading and augmentation
├── train.py           # Training script with mixed precision
├── detect.py          # Inference and visualization
├── requirements.txt   # Dependencies
├── data/              # Dataset directory
│   ├── train/
│   └── test/
├── models/            # Saved model checkpoints
└── output/            # Output visualizations
```

---

## 🎯 Usage

### Training

```powershell
# Train with demo data
python train.py

# With custom data, place images in:
# data/train/real/  - Real face images
# data/train/fake/  - Fake face images
```

### Inference

```powershell
# Detect on video
python detect.py --input path/to/video.mp4

# Detect on image
python detect.py --input path/to/image.jpg

# Run demo
python detect.py --demo
```

---

## ⚡ Performance Optimizations

| Feature | Your Setup | Optimization |
|---------|-----------|--------------|
| **GPU** | RTX 4060 (8GB) | Mixed Precision (FP16), Batch Size 32 |
| **CPU** | i7-13650HX | 8 DataLoader workers |
| **RAM** | 24GB | Pin Memory, Persistent Workers |

### Expected Performance

- **Training Speed:** ~200-300 images/second
- **Inference Speed:** ~50-100 FPS (single image)
- **Video Analysis:** ~30-50 FPS

---

## 🧠 Model Architecture

```
EfficientNet-B0 (Pretrained on ImageNet)
    ↓
Global Average Pooling
    ↓
Dense (512) + ReLU + Dropout(0.3)
    ↓
Dense (128) + ReLU + Dropout(0.2)
    ↓
Dense (2) → Softmax → [REAL, FAKE]
```

---

## 📊 Training Features

- ✅ **Mixed Precision Training** (FP16) - 2x faster, 50% less VRAM
- ✅ **Gradient Accumulation** - Effective batch size of 64
- ✅ **Cosine Annealing with Warm Restarts** - Better convergence
- ✅ **Label Smoothing** - Reduces overfitting
- ✅ **Advanced Augmentations** - Blur, compression, color jitter

---

## 📈 Data Augmentation Pipeline

```python
# Training augmentations for robustness
- Horizontal Flip (50%)
- Gaussian Noise/Blur/Motion Blur
- Brightness/Contrast adjustment
- Hue/Saturation/Value shifts
- JPEG Compression simulation
- Downscaling simulation
- Cutout/CoarseDropout
```

---

## 🔧 Configuration

Edit `config.py` to adjust:

```python
# Batch size (adjust if OOM)
BATCH_SIZE = 32

# Mixed precision (disable if issues)
MIXED_PRECISION = True

# Number of frames per video
FRAMES_PER_VIDEO = 32

# Learning rate
LEARNING_RATE = 1e-4
```

---

## 📋 Requirements

- Python 3.9+
- CUDA 12.1+
- cuDNN 8.x
- 8GB+ VRAM (RTX 4060 ✓)

---

## 🐛 Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size in config.py
BATCH_SIZE = 16

# Enable gradient checkpointing
GRADIENT_CHECKPOINTING = True
```

### Slow Training
```python
# Increase workers
NUM_WORKERS = 12

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 📚 References

- [Deepfake Detection Challenge (Kaggle)](https://www.kaggle.com/c/deepfake-detection-challenge)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [FaceForensics++ Dataset](https://github.com/ondyari/FaceForensics)

---

## 📄 License

MIT License - Free for educational and research purposes.

---

**Built with ❤️ for your RTX 4060 + i7-13650HX setup**
