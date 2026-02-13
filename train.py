"""
Training script for Deepfake Detection
Optimized for: i7-13650HX + RTX 4060 + 24GB RAM
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from config import (
    DEVICE, BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    MIXED_PRECISION, ACCUMULATION_STEPS, LOG_INTERVAL, SAVE_INTERVAL,
    MODEL_DIR, OUTPUT_DIR, TRAIN_DIR, TEST_DIR, RESUME_CHECKPOINT
)
from model import get_model
from dataset import create_dataloaders, create_demo_dataset


class Trainer:
    """Trainer class with optimizations for RTX 4060"""
    
    def __init__(self, model, train_loader, val_loader, 
                 learning_rate=LEARNING_RATE, epochs=EPOCHS):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = DEVICE
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=WEIGHT_DECAY,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=5,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Mixed precision training for RTX 4060
        self.scaler = GradScaler() if MIXED_PRECISION else None
        
        # Training history
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'val_f1': [], 'val_auc': []
        }
        
        self.best_val_acc = 0.0
        self.best_epoch = 0
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Mixed precision forward pass
            if MIXED_PRECISION:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss = loss / ACCUMULATION_STEPS
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss = loss / ACCUMULATION_STEPS
                loss.backward()
                
                if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
            
            total_loss += loss.item() * ACCUMULATION_STEPS
            
            # Predictions
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            if batch_idx % LOG_INTERVAL == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item() * ACCUMULATION_STEPS:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
        
        # Update scheduler
        self.scheduler.step()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Val]")
        
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            if MIXED_PRECISION:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0
        
        return avg_loss, accuracy, f1, auc, all_preds, all_labels
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_acc': self.best_val_acc
        }
        
        # Save latest
        torch.save(checkpoint, os.path.join(MODEL_DIR, 'latest_checkpoint.pth'))
        
        # Save best
        if is_best:
            torch.save(checkpoint, os.path.join(MODEL_DIR, 'best_model.pth'))
            print(f"💾 Saved best model with accuracy: {self.best_val_acc:.4f}")
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train', color='#2ecc71')
        axes[0, 0].plot(self.history['val_loss'], label='Validation', color='#e74c3c')
        axes[0, 0].set_title('Loss', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(self.history['train_acc'], label='Train', color='#2ecc71')
        axes[0, 1].plot(self.history['val_acc'], label='Validation', color='#e74c3c')
        axes[0, 1].set_title('Accuracy', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score
        axes[1, 0].plot(self.history['val_f1'], label='F1 Score', color='#3498db')
        axes[1, 0].set_title('Validation F1 Score', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # AUC
        axes[1, 1].plot(self.history['val_auc'], label='AUC', color='#9b59b6')
        axes[1, 1].set_title('Validation AUC', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Deepfake Detection Training History', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved training history plot")
    
    def train(self):
        """Full training loop"""
        print(f"\n{'='*60}")
        print(f"🚀 Starting Training")
        print(f"📱 Device: {self.device}")
        print(f"🔢 Epochs: {self.epochs}")
        print(f"📦 Batch Size: {BATCH_SIZE} (Effective: {BATCH_SIZE * ACCUMULATION_STEPS})")
        print(f"⚡ Mixed Precision: {MIXED_PRECISION}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, val_f1, val_auc, preds, labels = self.validate(epoch)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            self.history['val_auc'].append(val_auc)
            
            # Print epoch summary
            print(f"\n📈 Epoch {epoch+1}/{self.epochs} Summary:")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"   F1: {val_f1:.4f} | AUC: {val_auc:.4f}")
            
            # Save best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
            
            # Save checkpoint
            if (epoch + 1) % SAVE_INTERVAL == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"✅ Training Complete!")
        print(f"🏆 Best Accuracy: {self.best_val_acc:.4f} (Epoch {self.best_epoch})")
        print(f"{'='*60}")
        
        # Plot history
        self.plot_training_history()
        
        return self.history


def load_dataset_from_folders():
    """Load image paths and labels from data/train and data/test folders"""
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    
    # Load training data
    for label_name, label_val in [('real', 0), ('fake', 1)]:
        folder = os.path.join(TRAIN_DIR, label_name)
        if not os.path.exists(folder):
            print(f"⚠️  Warning: {folder} not found")
            continue
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                train_paths.append(os.path.join(folder, fname))
                train_labels.append(label_val)
    
    # Load test/validation data
    for label_name, label_val in [('real', 0), ('fake', 1)]:
        folder = os.path.join(TEST_DIR, label_name)
        if not os.path.exists(folder):
            print(f"⚠️  Warning: {folder} not found")
            continue
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                val_paths.append(os.path.join(folder, fname))
                val_labels.append(label_val)
    
    return train_paths, train_labels, val_paths, val_labels


def main():
    """Main training function with incremental fine-tuning support"""
    print("🎯 Deepfake Detection Training")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load real dataset from prepared folders
    print("\n📁 Loading dataset from data/ folders...")
    train_paths, train_labels, val_paths, val_labels = load_dataset_from_folders()
    
    if len(train_paths) == 0:
        print("❌ No training data found in data/train/real and data/train/fake!")
        print("   Run prepare_data.py first to extract frames.")
        print("   Falling back to demo dataset...")
        paths, labels = create_demo_dataset(num_samples=200)
        split_idx = int(len(paths) * 0.8)
        train_paths, train_labels = paths[:split_idx], labels[:split_idx]
        val_paths, val_labels = paths[split_idx:], labels[split_idx:]
    
    print(f"   Train: {len(train_paths)} images (Real: {train_labels.count(0) if isinstance(train_labels, list) else sum(1 for l in train_labels if l==0)}, Fake: {train_labels.count(1) if isinstance(train_labels, list) else sum(1 for l in train_labels if l==1)})")
    print(f"   Val:   {len(val_paths)} images (Real: {val_labels.count(0) if isinstance(val_labels, list) else sum(1 for l in val_labels if l==0)}, Fake: {val_labels.count(1) if isinstance(val_labels, list) else sum(1 for l in val_labels if l==1)})")
    
    # Create dataloaders
    print("\n📦 Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_paths, train_labels,
        val_paths, val_labels,
        dataset_type='image',
        batch_size=BATCH_SIZE
    )
    
    # Create model
    print("\n🧠 Creating model...")
    model = get_model(video_level=False)
    
    # Load existing checkpoint for fine-tuning
    if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):
        print(f"\n🔄 Loading existing checkpoint for fine-tuning: {RESUME_CHECKPOINT}")
        checkpoint = torch.load(RESUME_CHECKPOINT, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"   ✅ Loaded model weights (previous best acc: {checkpoint.get('best_val_acc', 'N/A')})")
        else:
            model.load_state_dict(checkpoint)
            print(f"   ✅ Loaded model weights")
    else:
        print("\n🆕 Training from scratch (no existing checkpoint found)")
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader)
    
    # Train
    history = trainer.train()
    
    print(f"\n📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
