
"""
Training Script for FaceForensics++ Dataset
Optimized for: i7-13650HX + RTX 4060 + 24GB RAM

This script trains the deepfake detection model using the FaceForensics++ dataset.
"""

import os
import sys
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import random
from datetime import datetime
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DEVICE, BATCH_SIZE, EPOCHS, NUM_WORKERS, PIN_MEMORY,
    TRAIN_DIR, TEST_DIR, DATA_DIR, IMAGE_SIZE
)
from model import get_model
from dataset import create_dataloaders, get_transforms
from train import Trainer
from face_detector import FaceDetector


# FaceForensics++ Dataset path
FF_DATASET_PATH = r"D:\download\archive\FaceForensics++_C23"
FF_METADATA_CSV = os.path.join(FF_DATASET_PATH, "csv", "FF++_Metadata.csv")


def setup_directories():
    """Create required directory structure"""
    dirs = [
        os.path.join(TRAIN_DIR, "real"),
        os.path.join(TRAIN_DIR, "fake"),
        os.path.join(TEST_DIR, "real"),
        os.path.join(TEST_DIR, "fake"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✅ Created directory structure")


def extract_frames_from_video(video_path, output_dir, num_frames=10, prefix="", 
                               face_detector=None):
    """Extract frames from a video file with optional face detection"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return 0
    
    # Sample frames uniformly
    indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames), dtype=int)
    saved = 0
    
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            if face_detector is not None:
                # Extract face from frame
                try:
                    face_crop, face_found = face_detector.extract_face_or_frame(frame)
                    frame_to_save = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
                except:
                    frame_to_save = frame
            else:
                frame_to_save = frame
            
            # Resize for efficiency
            frame_to_save = cv2.resize(frame_to_save, (IMAGE_SIZE, IMAGE_SIZE))
            
            filename = f"{prefix}frame_{i:03d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame_to_save)
            saved += 1
    
    cap.release()
    return saved


def load_faceforensics_metadata():
    """Load and parse the FaceForensics++ metadata CSV"""
    print(f"📁 Loading metadata from: {FF_METADATA_CSV}")
    
    if not os.path.exists(FF_METADATA_CSV):
        print(f"❌ Metadata file not found: {FF_METADATA_CSV}")
        return None, None
    
    df = pd.read_csv(FF_METADATA_CSV)
    print(f"📊 Found {len(df)} videos in metadata")
    
    # Get unique labels
    print(f"📑 Labels: {df['Label'].value_counts().to_dict()}")
    
    return df


def prepare_dataset_from_metadata(df, frames_per_video=15, test_split=0.2, 
                                   use_faces=True, max_videos=None):
    """
    Prepare training dataset from FaceForensics++ metadata
    
    Args:
        df: DataFrame with video metadata
        frames_per_video: Number of frames to extract per video
        test_split: Fraction of data for testing
        use_faces: Whether to extract face crops
        max_videos: Maximum number of videos to process (None for all)
    """
    setup_directories()
    
    # Initialize face detector if needed
    face_detector = None
    if use_faces:
        print("🔍 Initializing face detector...")
        try:
            face_detector = FaceDetector()
            print("✅ Face detector ready")
        except Exception as e:
            print(f"⚠️ Face detector failed to initialize: {e}")
            print("   Proceeding without face detection")
            face_detector = None
    
    # Prepare video list with full paths
    videos = []
    for _, row in df.iterrows():
        video_path = os.path.join(FF_DATASET_PATH, row['File Path'])
        # Handle both formats - some paths have folder prefix, some don't
        if not os.path.exists(video_path):
            # Try alternative path formats
            alt_path = os.path.join(FF_DATASET_PATH, row['File Path'].replace('/', '\\'))
            if os.path.exists(alt_path):
                video_path = alt_path
            else:
                continue
        
        videos.append({
            'path': video_path,
            'label': row['Label'].upper()
        })
    
    print(f"📹 Found {len(videos)} valid video paths")
    
    # Limit videos if specified
    if max_videos and max_videos < len(videos):
        random.shuffle(videos)
        videos = videos[:max_videos]
        print(f"📉 Limited to {max_videos} videos")
    
    # Shuffle and split
    random.shuffle(videos)
    split_idx = int(len(videos) * (1 - test_split))
    train_videos = videos[:split_idx]
    test_videos = videos[split_idx:]
    
    print(f"\n📊 Dataset split:")
    print(f"   Training: {len(train_videos)} videos")
    print(f"   Testing: {len(test_videos)} videos")
    
    # Process training videos
    print(f"\n🔄 Processing training videos...")
    train_real = 0
    train_fake = 0
    
    for i, video in enumerate(tqdm(train_videos, desc="Training")):
        if video['label'] == 'FAKE':
            output_dir = os.path.join(TRAIN_DIR, "fake")
            prefix = f"fake_{train_fake:05d}_"
            train_fake += 1
        else:
            output_dir = os.path.join(TRAIN_DIR, "real")
            prefix = f"real_{train_real:05d}_"
            train_real += 1
        
        extract_frames_from_video(
            video['path'], output_dir, frames_per_video, 
            prefix, face_detector
        )
    
    # Process test videos
    print(f"\n🔄 Processing test videos...")
    test_real = 0
    test_fake = 0
    
    for video in tqdm(test_videos, desc="Testing"):
        if video['label'] == 'FAKE':
            output_dir = os.path.join(TEST_DIR, "fake")
            prefix = f"fake_{test_fake:05d}_"
            test_fake += 1
        else:
            output_dir = os.path.join(TEST_DIR, "real")
            prefix = f"real_{test_real:05d}_"
            test_real += 1
        
        extract_frames_from_video(
            video['path'], output_dir, frames_per_video,
            prefix, face_detector
        )
    
    print_dataset_stats()


def print_dataset_stats():
    """Print dataset statistics"""
    print("\n" + "="*50)
    print("📊 DATASET STATISTICS")
    print("="*50)
    
    for split in ["train", "test"]:
        split_dir = os.path.join(DATA_DIR, split)
        if os.path.exists(split_dir):
            real_dir = os.path.join(split_dir, "real")
            fake_dir = os.path.join(split_dir, "fake")
            
            real_count = len(os.listdir(real_dir)) if os.path.exists(real_dir) else 0
            fake_count = len(os.listdir(fake_dir)) if os.path.exists(fake_dir) else 0
            
            print(f"\n{split.upper()}:")
            print(f"   Real images: {real_count}")
            print(f"   Fake images: {fake_count}")
            print(f"   Total: {real_count + fake_count}")
    
    print("\n" + "="*50)


def get_image_paths_and_labels(data_dir):
    """Get all image paths and labels from the prepared dataset"""
    paths = []
    labels = []
    
    real_dir = os.path.join(data_dir, "real")
    fake_dir = os.path.join(data_dir, "fake")
    
    # Load real images
    if os.path.exists(real_dir):
        for img_file in os.listdir(real_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                paths.append(os.path.join(real_dir, img_file))
                labels.append(0)  # 0 = REAL
    
    # Load fake images
    if os.path.exists(fake_dir):
        for img_file in os.listdir(fake_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                paths.append(os.path.join(fake_dir, img_file))
                labels.append(1)  # 1 = FAKE
    
    return paths, labels


def main():
    parser = argparse.ArgumentParser(description='Train on FaceForensics++ dataset')
    parser.add_argument('--prepare-only', action='store_true',
                        help='Only prepare dataset, do not train')
    parser.add_argument('--train-only', action='store_true',
                        help='Only train (assumes data is prepared)')
    parser.add_argument('--frames', type=int, default=15,
                        help='Frames per video (default: 15)')
    parser.add_argument('--max-videos', type=int, default=None,
                        help='Max videos to process (default: all)')
    parser.add_argument('--no-faces', action='store_true',
                        help='Disable face detection (use full frames)')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Training epochs (default: {EPOCHS})')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎯 FaceForensics++ Training Pipeline")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📍 Dataset: {FF_DATASET_PATH}")
    print(f"🖥️  Device: {DEVICE}")
    print("=" * 60)
    
    # Step 1: Prepare dataset (if not train-only)
    if not args.train_only:
        print("\n📦 STEP 1: Preparing Dataset")
        print("-" * 40)
        
        df = load_faceforensics_metadata()
        if df is None:
            print("❌ Failed to load metadata. Exiting.")
            return
        
        prepare_dataset_from_metadata(
            df,
            frames_per_video=args.frames,
            use_faces=not args.no_faces,
            max_videos=args.max_videos
        )
        
        if args.prepare_only:
            print("\n✅ Dataset preparation complete!")
            return
    
    # Step 2: Train model
    print("\n🧠 STEP 2: Training Model")
    print("-" * 40)
    
    # Load prepared data
    train_paths, train_labels = get_image_paths_and_labels(TRAIN_DIR)
    val_paths, val_labels = get_image_paths_and_labels(TEST_DIR)
    
    if len(train_paths) == 0:
        print("❌ No training data found. Run with --prepare-only first or check paths.")
        return
    
    print(f"📊 Training samples: {len(train_paths)}")
    print(f"📊 Validation samples: {len(val_paths)}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_paths, train_labels,
        val_paths, val_labels,
        dataset_type='image',
        batch_size=BATCH_SIZE
    )
    
    # Create model
    print("\n🔧 Creating model...")
    model = get_model(video_level=False)
    
    # Create trainer
    trainer = Trainer(
        model, train_loader, val_loader,
        epochs=args.epochs
    )
    
    # Train
    history = trainer.train()
    
    print(f"\n📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
