"""
Data Preparation Script for Deepfake Detection
Helps prepare training data from various formats:
1. Videos + JSON labels (Kaggle DFDC format)
2. Single folder of videos with manual labeling
3. Download sample dataset
"""

import os
import cv2
import json
import shutil
import numpy as np
from tqdm import tqdm
import random

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")


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
    print("✅ Created directory structure:")
    print(f"   {TRAIN_DIR}/real/")
    print(f"   {TRAIN_DIR}/fake/")
    print(f"   {TEST_DIR}/real/")
    print(f"   {TEST_DIR}/fake/")


def extract_frames_from_video(video_path, output_dir, num_frames=10, prefix=""):
    """Extract frames from a video file"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"⚠️ Could not open: {video_path}")
        return 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return 0
    
    # Sample frames uniformly
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    saved = 0
    
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            filename = f"{prefix}frame_{i:03d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved += 1
    
    cap.release()
    return saved


def prepare_from_json(videos_dir, metadata_json, frames_per_video=10, test_split=0.2):
    """
    Prepare dataset from Kaggle DFDC format (videos + metadata.json)
    
    metadata.json format:
    {
        "video1.mp4": {"label": "FAKE"},
        "video2.mp4": {"label": "REAL"},
        ...
    }
    """
    setup_directories()
    
    # Load metadata
    with open(metadata_json, 'r') as f:
        metadata = json.load(f)
    
    print(f"📊 Found {len(metadata)} videos in metadata")
    
    # Process each video
    real_count = 0
    fake_count = 0
    
    videos = list(metadata.items())
    random.shuffle(videos)
    
    split_idx = int(len(videos) * (1 - test_split))
    train_videos = videos[:split_idx]
    test_videos = videos[split_idx:]
    
    print(f"\n🔄 Processing {len(train_videos)} training videos...")
    for video_name, info in tqdm(train_videos):
        video_path = os.path.join(videos_dir, video_name)
        
        if not os.path.exists(video_path):
            continue
        
        label = info.get('label', '').upper()
        if label == 'FAKE':
            output_dir = os.path.join(TRAIN_DIR, "fake")
            prefix = f"fake_{fake_count:04d}_"
            fake_count += 1
        else:
            output_dir = os.path.join(TRAIN_DIR, "real")
            prefix = f"real_{real_count:04d}_"
            real_count += 1
        
        extract_frames_from_video(video_path, output_dir, frames_per_video, prefix)
    
    print(f"\n🔄 Processing {len(test_videos)} test videos...")
    for video_name, info in tqdm(test_videos):
        video_path = os.path.join(videos_dir, video_name)
        
        if not os.path.exists(video_path):
            continue
        
        label = info.get('label', '').upper()
        if label == 'FAKE':
            output_dir = os.path.join(TEST_DIR, "fake")
        else:
            output_dir = os.path.join(TEST_DIR, "real")
        
        prefix = f"{os.path.splitext(video_name)[0]}_"
        extract_frames_from_video(video_path, output_dir, frames_per_video, prefix)
    
    print_dataset_stats()


def prepare_with_interactive_labeling(videos_dir, frames_per_video=10):
    """
    Interactive labeling: Shows a frame from each video and asks for label
    """
    setup_directories()
    
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    videos = [f for f in os.listdir(videos_dir) if f.lower().endswith(video_extensions)]
    
    print(f"\n📹 Found {len(videos)} videos to label")
    print("For each video, enter: R (Real), F (Fake), or S (Skip)\n")
    
    labeled = {'real': [], 'fake': []}
    
    for i, video_name in enumerate(videos):
        video_path = os.path.join(videos_dir, video_name)
        
        # Show first frame
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Resize for display
            display = cv2.resize(frame, (640, 480))
            cv2.imshow(f"Video {i+1}/{len(videos)}: {video_name}", display)
            cv2.waitKey(500)  # Show for 500ms
            cv2.destroyAllWindows()
        
        while True:
            label = input(f"[{i+1}/{len(videos)}] {video_name} - Label (R/F/S): ").upper()
            if label in ['R', 'F', 'S']:
                break
            print("Invalid input. Enter R, F, or S")
        
        if label == 'R':
            labeled['real'].append(video_path)
        elif label == 'F':
            labeled['fake'].append(video_path)
    
    # Process labeled videos
    print(f"\n✅ Labeled {len(labeled['real'])} REAL, {len(labeled['fake'])} FAKE")
    print("🔄 Extracting frames...")
    
    for i, video_path in enumerate(tqdm(labeled['real'])):
        output_dir = TRAIN_DIR if i < len(labeled['real']) * 0.8 else TEST_DIR
        output_dir = os.path.join(output_dir, "real")
        prefix = f"real_{i:04d}_"
        extract_frames_from_video(video_path, output_dir, frames_per_video, prefix)
    
    for i, video_path in enumerate(tqdm(labeled['fake'])):
        output_dir = TRAIN_DIR if i < len(labeled['fake']) * 0.8 else TEST_DIR
        output_dir = os.path.join(output_dir, "fake")
        prefix = f"fake_{i:04d}_"
        extract_frames_from_video(video_path, output_dir, frames_per_video, prefix)
    
    print_dataset_stats()


def prepare_from_two_folders(real_folder, fake_folder, frames_per_video=10, test_split=0.2):
    """
    Prepare from two separate folders containing real and fake videos
    """
    setup_directories()
    
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    
    real_videos = [os.path.join(real_folder, f) for f in os.listdir(real_folder) 
                   if f.lower().endswith(video_extensions)]
    fake_videos = [os.path.join(fake_folder, f) for f in os.listdir(fake_folder) 
                   if f.lower().endswith(video_extensions)]
    
    print(f"📹 Found {len(real_videos)} real videos, {len(fake_videos)} fake videos")
    
    # Shuffle and split
    random.shuffle(real_videos)
    random.shuffle(fake_videos)
    
    real_split = int(len(real_videos) * (1 - test_split))
    fake_split = int(len(fake_videos) * (1 - test_split))
    
    # Process real videos
    print("\n🔄 Processing REAL videos...")
    for i, video_path in enumerate(tqdm(real_videos)):
        if i < real_split:
            output_dir = os.path.join(TRAIN_DIR, "real")
        else:
            output_dir = os.path.join(TEST_DIR, "real")
        
        prefix = f"real_{i:04d}_"
        extract_frames_from_video(video_path, output_dir, frames_per_video, prefix)
    
    # Process fake videos
    print("\n🔄 Processing FAKE videos...")
    for i, video_path in enumerate(tqdm(fake_videos)):
        if i < fake_split:
            output_dir = os.path.join(TRAIN_DIR, "fake")
        else:
            output_dir = os.path.join(TEST_DIR, "fake")
        
        prefix = f"fake_{i:04d}_"
        extract_frames_from_video(video_path, output_dir, frames_per_video, prefix)
    
    print_dataset_stats()


def create_labels_file(videos_dir, output_file="labels.json"):
    """
    Create a template labels.json file for manual labeling
    """
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    videos = [f for f in os.listdir(videos_dir) if f.lower().endswith(video_extensions)]
    
    labels = {}
    for video in videos:
        labels[video] = {"label": "UNKNOWN"}  # User fills in REAL or FAKE
    
    output_path = os.path.join(videos_dir, output_file)
    with open(output_path, 'w') as f:
        json.dump(labels, f, indent=2)
    
    print(f"✅ Created {output_path}")
    print(f"📝 Edit this file and change 'UNKNOWN' to 'REAL' or 'FAKE' for each video")
    print(f"   Then run: python prepare_data.py --from-json {videos_dir} {output_path}")


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


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare data for deepfake detection')
    parser.add_argument('--from-json', nargs=2, metavar=('VIDEOS_DIR', 'METADATA_JSON'),
                        help='Prepare from videos folder + metadata.json')
    parser.add_argument('--interactive', type=str, metavar='VIDEOS_DIR',
                        help='Interactive labeling of videos')
    parser.add_argument('--two-folders', nargs=2, metavar=('REAL_FOLDER', 'FAKE_FOLDER'),
                        help='Prepare from separate real and fake folders')
    parser.add_argument('--create-labels', type=str, metavar='VIDEOS_DIR',
                        help='Create a template labels.json for manual labeling')
    parser.add_argument('--stats', action='store_true',
                        help='Show current dataset statistics')
    parser.add_argument('--frames', type=int, default=10,
                        help='Number of frames to extract per video (default: 10)')
    
    args = parser.parse_args()
    
    if args.from_json:
        prepare_from_json(args.from_json[0], args.from_json[1], args.frames)
    elif args.interactive:
        prepare_with_interactive_labeling(args.interactive, args.frames)
    elif args.two_folders:
        prepare_from_two_folders(args.two_folders[0], args.two_folders[1], args.frames)
    elif args.create_labels:
        create_labels_file(args.create_labels)
    elif args.stats:
        print_dataset_stats()
    else:
        print("""
🎯 Data Preparation Tool for Deepfake Detection

USAGE OPTIONS:

1️⃣  If you have videos + a metadata.json file (Kaggle format):
    python prepare_data.py --from-json <videos_folder> <metadata.json>

2️⃣  If you have a folder of unlabeled videos (interactive labeling):
    python prepare_data.py --interactive <videos_folder>

3️⃣  If you have separate folders for real and fake videos:
    python prepare_data.py --two-folders <real_folder> <fake_folder>

4️⃣  Create a labels.json template to fill manually:
    python prepare_data.py --create-labels <videos_folder>

5️⃣  Check current dataset statistics:
    python prepare_data.py --stats

EXAMPLE:
    python prepare_data.py --create-labels D:\\download
    # Edit the labels.json file, then:
    python prepare_data.py --from-json D:\\download D:\\download\\labels.json
""")


if __name__ == "__main__":
    main()
