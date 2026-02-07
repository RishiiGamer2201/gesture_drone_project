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
    print(f"Run: python3 src/data_collection/collect_static.py")
    exit(1)

dataset = np.loadtxt(data_file, delimiter=',', dtype=np.float32)
X, y = dataset[:, :-1], dataset[:, -1]

print(f"✓ Loaded {len(dataset)} samples")

knn = cv2.ml.KNearest_create()
knn.train(X, cv2.ml.ROW_SAMPLE, y)

os.makedirs(MODELS_DIR, exist_ok=True)
knn.save(KNN_CONFIG['model_file'])

metadata = {'gesture_names': STATIC_GESTURES, 'n_samples': len(dataset)}
with open(KNN_CONFIG['metadata_file'], 'wb') as f:
    pickle.dump(metadata, f)

print(f"✓ Model saved to {KNN_CONFIG['model_file']}")
print("Run: python3 src/controllers/advanced_controller.py --model knn")
