"""Test the trained deepfake detection model"""
import sys
sys.path.append('.')
from detect import DeepfakeInference
import os

# Initialize model
print("Loading model...")
inference = DeepfakeInference('models/best_model.pth')
print("[OK] Model loaded successfully!")
print()

# Define test videos (from metadata: aagundkpoh.mp4 is FAKE, abjqjcvhwe.mp4 is REAL)
videos_dir = r"c:\Users\Krushna\gp\df\deepfake_detection\dfdc_data\dfdc_train_part_48"
fake_video = os.path.join(videos_dir, "aagundkpoh.mp4")
real_video = os.path.join(videos_dir, "abjqjcvhwe.mp4")

# Test FAKE video
print("=" * 50)
print("[TEST] Testing FAKE video (aagundkpoh.mp4)")
print("=" * 50)
result = inference.predict_video(fake_video, show_progress=False)
print("   Verdict: {}".format(result['verdict']))
print("   Confidence: {:.1f}%".format(result['confidence']*100))
print("   Fake Probability: {:.1f}%".format(result['average_fake_probability']*100))
print("   (Expected: FAKE)")

# Test REAL video
print()
print("=" * 50)
print("[TEST] Testing REAL video (abjqjcvhwe.mp4)")
print("=" * 50)
result2 = inference.predict_video(real_video, show_progress=False)
print("   Verdict: {}".format(result2['verdict']))
print("   Confidence: {:.1f}%".format(result2['confidence']*100))
print("   Fake Probability: {:.1f}%".format(result2['average_fake_probability']*100))
print("   (Expected: REAL)")

print()
print("[OK] Model testing complete!")
