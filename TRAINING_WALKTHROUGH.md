# Deepfake Detection Model Training - Walkthrough

## Summary
Successfully trained an EfficientNet B0-based deepfake detection model on the FaceForensics++ dataset.

## Training Configuration
| Parameter | Value |
|-----------|-------|
| Dataset | FaceForensics++_C23 |
| Training Videos | 400 |
| Test Videos | 100 |
| Frames per Video | 10 |
| Total Training Images | ~4,000 |
| Epochs | 20 |
| Batch Size | 32 (Effective: 64) |
| Device | CPU |

## Results

### Model Checkpoints
- [best_model.pth](file:///c:/Users/Krushna/df/deepfake%202/DEEPFAKE/models/best_model.pth) (57.2 MB)
- [latest_checkpoint.pth](file:///c:/Users/Krushna/df/deepfake%202/DEEPFAKE/models/latest_checkpoint.pth) (57.2 MB)

### Training History
![Training History](C:/Users/Krushna/.gemini/antigravity/brain/c831a703-e809-4688-abb4-ce3eac16627c/training_history.png)

### Metrics Analysis
| Metric | Final Value | Notes |
|--------|-------------|-------|
| Training Accuracy | ~99% | Model successfully learned training data |
| Validation Accuracy | ~78% | Gap indicates overfitting |
| Validation F1 Score | ~0.76 | Moderate performance |
| Validation AUC | ~0.57 | Room for improvement |

> [!WARNING]
> The model shows signs of **overfitting** - training loss continues to decrease while validation loss increases. This is common with limited training data.

## Recommendations for Improvement
1. **Use more training data** - Process more of the 7,000+ available videos
2. **Enable CUDA** - Install PyTorch with CUDA support for faster training
3. **Add regularization** - Increase dropout or weight decay
4. **Use face detection** - Enable `--use-faces` flag for better accuracy
5. **Data augmentation** - Add more augmentation to reduce overfitting

## Using the Trained Model
To use the trained model for detection:
```powershell
cd "c:\Users\Krushna\df\deepfake 2\DEEPFAKE"
python detect.py --video <path_to_video>
```

## Files Created
- [train_faceforensics.py](file:///c:/Users/Krushna/df/deepfake%202/DEEPFAKE/train_faceforensics.py) - Training script for FaceForensics++ dataset
