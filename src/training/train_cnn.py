"""Train CNN Model"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
import numpy as np
import cv2
import pickle
from tensorflow import keras
from sklearn.model_selection import train_test_split
from config.config import *

print("=== CNN Training ===")

images, labels = [], []

# --- FIX START: Iterate dynamically instead of hardcoding range(10) ---
for gid, gname in STATIC_GESTURES.items():
    gdir = os.path.join(IMAGE_DATA_DIR, f"{gid}_{gname}")
    
    if not os.path.exists(gdir):
        print(f"⚠️ Warning: Directory not found for {gname} (ID: {gid})")
        continue
    
    count = 0
    for img_file in os.listdir(gdir):
        if not img_file.endswith('.png'): continue
        
        # Load in grayscale
        img = cv2.imread(os.path.join(gdir, img_file), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # Resize to 64x64 and normalize to 0-1
            img = cv2.resize(img, (64, 64)) / 255.0
            images.append(img)
            labels.append(gid)
            count += 1
            
    print(f"  {gname}: {count} images")
# --- FIX END ---

if len(images) == 0:
    print("❌ No image data! Run: python src/data_collection/collect_images.py")
    exit(1)

# Reshape for CNN: (Batch Size, Height, Width, Channels)
X = np.array(images).reshape(-1, 64, 64, 1)
y = np.array(labels)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Use the config length, not just the unique data found, to ensure model architecture matches config
num_classes = len(STATIC_GESTURES) 

y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

# Define the CNN Architecture
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(f"\nTraining on {len(X_train)} samples, Validating on {len(X_val)} samples...")
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=16, verbose=1)

os.makedirs(MODELS_DIR, exist_ok=True)
model.save(CNN_CONFIG['model_file'])

metadata = {'gesture_names': STATIC_GESTURES, 'n_samples': len(X), 'image_size': 64}
with open(CNN_CONFIG['metadata_file'], 'wb') as f:
    pickle.dump(metadata, f)

print(f"✓ CNN model saved to {CNN_CONFIG['model_file']}")
print("Run: python src/controllers/advanced_controller.py --model cnn")