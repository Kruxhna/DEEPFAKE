"""
Deepfake Detection Inference Script
Detect deepfakes in images and videos
Optimized for RTX 4060
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from datetime import datetime

from config import DEVICE, IMAGE_SIZE, FRAMES_PER_VIDEO, MODEL_DIR, OUTPUT_DIR
from model import DeepfakeDetector
from dataset import get_transforms


class DeepfakeInference:
    """Inference class for deepfake detection"""
    
    def __init__(self, model_path=None):
        self.device = DEVICE
        self.transform = get_transforms(is_training=False)
        
        # Load model
        self.model = DeepfakeDetector()
        
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded model from: {model_path}")
        else:
            print("⚠️ No pretrained model found. Using random weights.")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Class labels
        self.labels = ['REAL', 'FAKE']
        self.colors = {'REAL': (0, 255, 0), 'FAKE': (0, 0, 255)}  # BGR
    
    def preprocess_image(self, image):
        """Preprocess image for inference"""
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # torchvision transforms work directly on numpy arrays/PIL images
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        return tensor
    
    @torch.no_grad()
    def predict_image(self, image):
        """Predict on a single image"""
        tensor = self.preprocess_image(image)
        
        # Use autocast only if CUDA is available
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                output = self.model(tensor)
        else:
            output = self.model(tensor)
        
        probs = torch.softmax(output, dim=1)[0]
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item()
        
        return {
            'class': self.labels[pred_class],
            'confidence': confidence,
            'probabilities': {
                'REAL': probs[0].item(),
                'FAKE': probs[1].item()
            }
        }
    
    @torch.no_grad()
    def predict_video(self, video_path, show_progress=True):
        """Predict on a video file"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📹 Video Info: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")
        
        # Sample frames uniformly
        sample_indices = np.linspace(0, total_frames - 1, FRAMES_PER_VIDEO, dtype=int)
        
        frame_predictions = []
        
        iterator = tqdm(sample_indices, desc="Analyzing frames") if show_progress else sample_indices
        
        for idx in iterator:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                result = self.predict_image(frame)
                frame_predictions.append({
                    'frame': idx,
                    **result
                })
        
        cap.release()
        
        # Aggregate predictions
        fake_probs = [p['probabilities']['FAKE'] for p in frame_predictions]
        avg_fake_prob = np.mean(fake_probs)
        
        video_result = {
            'video_path': video_path,
            'total_frames': total_frames,
            'sampled_frames': len(frame_predictions),
            'average_fake_probability': avg_fake_prob,
            'verdict': 'FAKE' if avg_fake_prob > 0.5 else 'REAL',
            'confidence': max(avg_fake_prob, 1 - avg_fake_prob),
            'frame_predictions': frame_predictions
        }
        
        return video_result
    
    def analyze_and_visualize(self, video_path, output_path=None):
        """Analyze video and create visualization"""
        result = self.predict_video(video_path)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Frame probability timeline
        frame_nums = [p['frame'] for p in result['frame_predictions']]
        fake_probs = [p['probabilities']['FAKE'] for p in result['frame_predictions']]
        
        axes[0, 0].plot(frame_nums, fake_probs, 'o-', color='#e74c3c', linewidth=2, markersize=6)
        axes[0, 0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
        axes[0, 0].fill_between(frame_nums, fake_probs, 0.5, 
                                 where=[p > 0.5 for p in fake_probs],
                                 color='#e74c3c', alpha=0.3)
        axes[0, 0].fill_between(frame_nums, fake_probs, 0.5,
                                 where=[p <= 0.5 for p in fake_probs],
                                 color='#2ecc71', alpha=0.3)
        axes[0, 0].set_xlabel('Frame Number', fontsize=11)
        axes[0, 0].set_ylabel('Fake Probability', fontsize=11)
        axes[0, 0].set_title('Frame-by-Frame Analysis', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Probability distribution
        axes[0, 1].hist(fake_probs, bins=20, color='#3498db', edgecolor='white', alpha=0.8)
        axes[0, 1].axvline(x=0.5, color='gray', linestyle='--', linewidth=2)
        axes[0, 1].axvline(x=result['average_fake_probability'], color='#e74c3c', 
                           linestyle='-', linewidth=2, label=f"Avg: {result['average_fake_probability']:.2f}")
        axes[0, 1].set_xlabel('Fake Probability', fontsize=11)
        axes[0, 1].set_ylabel('Frequency', fontsize=11)
        axes[0, 1].set_title('Probability Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Sample frames
        cap = cv2.VideoCapture(video_path)
        sample_frames = []
        sample_indices = np.linspace(0, result['total_frames'] - 1, 4, dtype=int)
        
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                sample_frames.append(frame)
        cap.release()
        
        if len(sample_frames) >= 4:
            sample_grid = np.concatenate([
                np.concatenate(sample_frames[:2], axis=1),
                np.concatenate(sample_frames[2:4], axis=1)
            ], axis=0)
            axes[1, 0].imshow(sample_grid)
            axes[1, 0].set_title('Sample Frames', fontsize=12, fontweight='bold')
            axes[1, 0].axis('off')
        
        # Final verdict
        axes[1, 1].axis('off')
        verdict_color = '#e74c3c' if result['verdict'] == 'FAKE' else '#2ecc71'
        axes[1, 1].text(0.5, 0.6, result['verdict'], fontsize=48, fontweight='bold',
                        ha='center', va='center', color=verdict_color,
                        transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.5, 0.35, f"Confidence: {result['confidence']*100:.1f}%",
                        fontsize=20, ha='center', va='center',
                        transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.5, 0.15, f"Avg Fake Prob: {result['average_fake_probability']:.3f}",
                        fontsize=14, ha='center', va='center', color='gray',
                        transform=axes[1, 1].transAxes)
        
        video_name = os.path.basename(video_path)
        plt.suptitle(f'Deepfake Analysis: {video_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path is None:
            output_path = os.path.join(OUTPUT_DIR, f"analysis_{video_name}.png")
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"📊 Analysis saved to: {output_path}")
        
        return result


def demo():
    """Demo function - creates a sample video and runs detection"""
    print("\n" + "="*60)
    print("🎯 Deepfake Detection Demo")
    print("="*60)
    
    # Create a sample video for testing
    demo_dir = os.path.join(OUTPUT_DIR, "demo")
    os.makedirs(demo_dir, exist_ok=True)
    
    demo_video = os.path.join(demo_dir, "sample_video.mp4")
    
    # Create sample video
    print("\n📹 Creating sample video for demo...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(demo_video, fourcc, 30, (640, 480))
    
    for i in range(150):  # 5 seconds at 30 fps
        # Create gradient background
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = np.linspace(0, 255, 640).reshape(1, -1)  # Blue gradient
        frame[:, :, 1] = i % 128 + 64  # Green animation
        frame[:, :, 2] = 128  # Red constant
        
        # Add some moving elements
        cv2.circle(frame, (320 + int(100*np.sin(i/10)), 240), 50, (255, 255, 255), -1)
        cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"✅ Created sample video: {demo_video}")
    
    # Run detection
    print("\n🔍 Running deepfake detection...")
    detector = DeepfakeInference(os.path.join(MODEL_DIR, 'best_model.pth'))
    
    result = detector.analyze_and_visualize(demo_video)
    
    print("\n" + "="*60)
    print("📊 DETECTION RESULTS")
    print("="*60)
    print(f"   Video: {os.path.basename(demo_video)}")
    print(f"   Verdict: {result['verdict']}")
    print(f"   Confidence: {result['confidence']*100:.1f}%")
    print(f"   Fake Probability: {result['average_fake_probability']:.3f}")
    print(f"   Frames Analyzed: {result['sampled_frames']}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Deepfake Detection Inference')
    parser.add_argument('--input', type=str, help='Path to image or video file')
    parser.add_argument('--model', type=str, default=os.path.join(MODEL_DIR, 'best_model.pth'),
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, help='Output path for visualization')
    parser.add_argument('--demo', action='store_true', help='Run demo with sample video')
    
    args = parser.parse_args()
    
    if args.demo or args.input is None:
        demo()
        return
    
    detector = DeepfakeInference(model_path=args.model)
    
    if args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        result = detector.analyze_and_visualize(args.input, args.output)
    else:
        result = detector.predict_image(args.input)
        print(f"\n📊 Result: {result['class']} (Confidence: {result['confidence']*100:.1f}%)")


if __name__ == "__main__":
    main()
