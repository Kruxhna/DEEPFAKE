"""
Deepfake Detection Model
Using EfficientNet with custom classification head
Optimized for RTX 4060
"""

import torch
import torch.nn as nn
import timm
from config import DEVICE, MODEL_NAME, NUM_CLASSES, PRETRAINED


class DeepfakeDetector(nn.Module):
    """
    EfficientNet-based Deepfake Detector
    Optimized for RTX 4060 with ~8GB VRAM
    """
    
    def __init__(self, model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=PRETRAINED):
        super(DeepfakeDetector, self).__init__()
        
        # Load pretrained backbone
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0,  # Remove classifier
            global_pool=''  # Remove pooling
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            self.feature_dim = features.shape[1]
        
        # Custom pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.3)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier()
    
    def _init_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Global pooling
        pooled = self.global_pool(features)
        pooled = pooled.view(pooled.size(0), -1)
        
        # Dropout
        pooled = self.dropout(pooled)
        
        # Classification
        output = self.classifier(pooled)
        
        return output
    
    def extract_features(self, x):
        """Extract feature embeddings for analysis"""
        features = self.backbone(x)
        pooled = self.global_pool(features)
        return pooled.view(pooled.size(0), -1)


class VideoDeepfakeDetector(nn.Module):
    """
    Video-level deepfake detector using temporal aggregation
    Processes multiple frames and combines predictions
    """
    
    def __init__(self, frame_model=None, aggregation='attention'):
        super(VideoDeepfakeDetector, self).__init__()
        
        self.frame_model = frame_model if frame_model else DeepfakeDetector()
        self.aggregation = aggregation
        
        # Freeze backbone for faster training
        for param in self.frame_model.backbone.parameters():
            param.requires_grad = False
        
        # Attention-based temporal aggregation
        if aggregation == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(self.frame_model.feature_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
                nn.Softmax(dim=1)
            )
        
        # Video-level classifier
        self.video_classifier = nn.Sequential(
            nn.Linear(self.frame_model.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES)
        )
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, num_frames, C, H, W)
        Returns:
            Video-level predictions
        """
        batch_size, num_frames, C, H, W = x.shape
        
        # Reshape for batch processing
        x = x.view(batch_size * num_frames, C, H, W)
        
        # Extract frame features
        features = self.frame_model.extract_features(x)
        features = features.view(batch_size, num_frames, -1)
        
        if self.aggregation == 'attention':
            # Attention-weighted aggregation
            attention_weights = self.attention(features)
            video_features = torch.sum(attention_weights * features, dim=1)
        elif self.aggregation == 'mean':
            # Simple mean aggregation
            video_features = torch.mean(features, dim=1)
        else:
            # Max aggregation
            video_features, _ = torch.max(features, dim=1)
        
        # Video classification
        output = self.video_classifier(video_features)
        
        return output


def get_model(video_level=False):
    """Factory function to get the appropriate model"""
    if video_level:
        model = VideoDeepfakeDetector()
    else:
        model = DeepfakeDetector()
    
    model = model.to(DEVICE)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total Parameters: {total_params:,}")
    print(f"🎯 Trainable Parameters: {trainable_params:,}")
    
    return model


if __name__ == "__main__":
    # Test model
    print("Testing DeepfakeDetector...")
    model = get_model(video_level=False)
    
    # Test with dummy input
    dummy_input = torch.randn(4, 3, 224, 224).to(DEVICE)
    output = model(dummy_input)
    print(f"✅ Output shape: {output.shape}")
    
    print("\nTesting VideoDeepfakeDetector...")
    video_model = get_model(video_level=True)
    
    # Test with video input (batch, frames, C, H, W)
    video_input = torch.randn(2, 8, 3, 224, 224).to(DEVICE)
    video_output = video_model(video_input)
    print(f"✅ Video output shape: {video_output.shape}")
