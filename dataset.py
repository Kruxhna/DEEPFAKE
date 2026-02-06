"""
Dataset classes for Deepfake Detection
Handles video loading, face extraction, and data augmentation
Optimized for 24GB RAM
Using torchvision transforms for better Windows compatibility
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import random

from config import (
    IMAGE_SIZE, FRAMES_PER_VIDEO, BATCH_SIZE, NUM_WORKERS, 
    PIN_MEMORY, AUGMENTATION_PROBABILITY, DEVICE
)


def get_transforms(is_training=True):
    """Get augmentation transforms optimized for deepfake detection"""
    
    if is_training:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
            ], p=0.3),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            ], p=0.4),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.85, 1.15)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


class DeepfakeImageDataset(Dataset):
    """Dataset for frame-level deepfake detection"""
    
    def __init__(self, image_paths, labels, transform=None, is_training=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform if transform else get_transforms(is_training)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            # Return a black image if loading fails
            image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms (torchvision transforms work directly on arrays)
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)


class DeepfakeVideoDataset(Dataset):
    """Dataset for video-level deepfake detection"""
    
    def __init__(self, video_paths, labels, frames_per_video=FRAMES_PER_VIDEO, 
                 transform=None, is_training=True, face_detector=None):
        self.video_paths = video_paths
        self.labels = labels
        self.frames_per_video = frames_per_video
        self.transform = transform if transform else get_transforms(is_training)
        self.is_training = is_training
        self.face_detector = face_detector
    
    def __len__(self):
        return len(self.video_paths)
    
    def extract_frames(self, video_path):
        """Extract frames from video with uniform sampling"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"⚠️ Could not open video: {video_path}")
            return [np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)] * self.frames_per_video
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            cap.release()
            return [np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)] * self.frames_per_video
        
        # Calculate frame indices to sample
        if self.is_training:
            # Random sampling during training
            indices = sorted(random.sample(range(total_frames), min(self.frames_per_video, total_frames)))
        else:
            # Uniform sampling during evaluation
            indices = np.linspace(0, total_frames - 1, self.frames_per_video, dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                frames.append(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))
        
        cap.release()
        
        # Pad if necessary
        while len(frames) < self.frames_per_video:
            frames.append(frames[-1] if frames else np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))
        
        return frames[:self.frames_per_video]
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Extract frames
        frames = self.extract_frames(video_path)
        
        # Apply transforms to each frame (torchvision transforms)
        transformed_frames = []
        for frame in frames:
            if self.transform:
                transformed_frames.append(self.transform(frame))
            else:
                transformed_frames.append(torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0)
        
        # Stack frames: (num_frames, C, H, W)
        video_tensor = torch.stack(transformed_frames)
        
        return video_tensor, torch.tensor(label, dtype=torch.long)


def create_dataloaders(train_paths, train_labels, val_paths, val_labels, 
                       dataset_type='image', batch_size=BATCH_SIZE):
    """Create train and validation dataloaders"""
    
    DatasetClass = DeepfakeImageDataset if dataset_type == 'image' else DeepfakeVideoDataset
    
    train_dataset = DatasetClass(
        train_paths, train_labels,
        transform=get_transforms(is_training=True),
        is_training=True
    )
    
    val_dataset = DatasetClass(
        val_paths, val_labels,
        transform=get_transforms(is_training=False),
        is_training=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    print(f"📊 Train samples: {len(train_dataset)}, Batches: {len(train_loader)}")
    print(f"📊 Val samples: {len(val_dataset)}, Batches: {len(val_loader)}")
    
    return train_loader, val_loader


# Demo/Test function
def create_demo_dataset(num_samples=100, output_dir="data/demo"):
    """Create a demo dataset with synthetic images for testing"""
    os.makedirs(f"{output_dir}/real", exist_ok=True)
    os.makedirs(f"{output_dir}/fake", exist_ok=True)
    
    paths = []
    labels = []
    
    print("Creating demo dataset...")
    for i in tqdm(range(num_samples)):
        # Create random images (in real usage, these would be face images)
        is_fake = i % 2 == 0
        label_dir = "fake" if is_fake else "real"
        
        # Generate random image
        img = np.random.randint(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        
        # Add some visual difference between real and fake
        if is_fake:
            img = cv2.GaussianBlur(img, (5, 5), 0)  # Fake images have blur
        
        path = f"{output_dir}/{label_dir}/sample_{i:04d}.jpg"
        cv2.imwrite(path, img)
        
        paths.append(path)
        labels.append(1 if is_fake else 0)
    
    print(f"✅ Created {num_samples} demo samples")
    return paths, labels


if __name__ == "__main__":
    # Test dataset creation
    paths, labels = create_demo_dataset(num_samples=20)
    
    # Test dataloader
    train_loader, val_loader = create_dataloaders(
        paths[:16], labels[:16],
        paths[16:], labels[16:],
        dataset_type='image',
        batch_size=4
    )
    
    # Test batch
    for images, targets in train_loader:
        print(f"✅ Batch shape: {images.shape}, Labels: {targets}")
        break
