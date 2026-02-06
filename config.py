"""
Configuration file for Deepfake Detection
Optimized for: i7-13650HX + RTX 4060 + 24GB RAM
"""

import torch
import os

# ============================================
# HARDWARE CONFIGURATION
# ============================================
# Auto-detect CUDA device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 8  # i7-13650HX has 24 threads, use 8 for data loading
PIN_MEMORY = True  # Enable for faster GPU transfer

# GPU Memory Optimization for RTX 4060 (8GB VRAM)
MIXED_PRECISION = True  # Use FP16 for faster training
GRADIENT_CHECKPOINTING = False  # Enable if running out of VRAM

# ============================================
# PATHS
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Create directories if they don't exist
for dir_path in [DATA_DIR, TRAIN_DIR, TEST_DIR, MODEL_DIR, OUTPUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ============================================
# MODEL CONFIGURATION
# ============================================
MODEL_NAME = "efficientnet_b0"  # Good balance of speed and accuracy
NUM_CLASSES = 2  # REAL (0), FAKE (1)
PRETRAINED = True

# ============================================
# TRAINING CONFIGURATION (Optimized for 24GB RAM + RTX 4060)
# ============================================
BATCH_SIZE = 32  # Can handle this with RTX 4060
ACCUMULATION_STEPS = 2  # Effective batch size = 64
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# ============================================
# IMAGE/VIDEO PROCESSING
# ============================================
IMAGE_SIZE = 224  # Standard size for EfficientNet
FRAMES_PER_VIDEO = 32  # Number of frames to sample per video
FACE_DETECTION_CONFIDENCE = 0.9

# ============================================
# DETECTION THRESHOLDS (Tuned for sensitivity)
# ============================================
# Lower threshold = more sensitive to fakes
FAKE_THRESHOLD = 0.42  # Above this = FAKE
SUSPICIOUS_THRESHOLD = 0.38  # 0.38+ = SUSPICIOUS, above FAKE_THRESHOLD = FAKE
# Below SUSPICIOUS_THRESHOLD (0.38) = REAL

# ============================================
# DATA AUGMENTATION
# ============================================
AUGMENTATION_PROBABILITY = 0.5

# ============================================
# LOGGING
# ============================================
LOG_INTERVAL = 10  # Log every N batches
SAVE_INTERVAL = 5  # Save model every N epochs

print(f"🖥️  Device: {DEVICE}")
print(f"📊 CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
