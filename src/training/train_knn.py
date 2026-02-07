"""Train KNN Model"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
import numpy as np
import cv2
import pickle
from config.config import *

print("=== KNN Training ===")
data_file = os.path.join(TRAINING_DATA_DIR, 'gesture_data.csv')

if not os.path.exists(data_file):
    print("❌ No training data found!")
    print(f"Run: python src/data_collection/collect_static.py")
    exit(1)

# Load the dataset
dataset = np.loadtxt(data_file, delimiter=',', dtype=np.float32)

# --- START FIX ---
# Check if the dataset is empty
if dataset.size == 0:
    print("❌ Dataset is empty.")
    exit(1)

# Check if the dataset is 1D (happens if there is only 1 sample) and reshape it
if dataset.ndim == 1:
    dataset = dataset.reshape(1, -1)
# --- END FIX ---

# Now safe to slice
X, y = dataset[:, :-1], dataset[:, -1]

print(f"✓ Loaded {len(dataset)} samples")

# Ensure y (labels) is the correct shape and type (integer) for OpenCV
y = y.astype(np.float32) # KNN typically expects float responses in OpenCV, but int is safer for classification logic. 
# Note: cv2.ml.KNearest works with float32 for regression or classification, 
# but usually standardizes on float32 for inputs.

knn = cv2.ml.KNearest_create()
knn.train(X, cv2.ml.ROW_SAMPLE, y)

os.makedirs(MODELS_DIR, exist_ok=True)
knn.save(KNN_CONFIG['model_file'])

metadata = {'gesture_names': STATIC_GESTURES, 'n_samples': len(dataset)}
with open(KNN_CONFIG['metadata_file'], 'wb') as f:
    pickle.dump(metadata, f)

print(f"✓ Model saved to {KNN_CONFIG['model_file']}")
print("Run: python src/controllers/advanced_controller.py --model knn")