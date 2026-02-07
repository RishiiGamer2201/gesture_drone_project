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
for gid in range(9):
    gname = STATIC_GESTURES[gid]
    gdir = os.path.join(IMAGE_DATA_DIR, f"{gid}_{gname}")
    if not os.path.exists(gdir): continue
    
    for img_file in os.listdir(gdir):
        if not img_file.endswith('.png'): continue
        img = cv2.imread(os.path.join(gdir, img_file), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (64, 64)) / 255.0
            images.append(img)
            labels.append(gid)
    print(f"  {gname}: {len([l for l in labels if l == gid])} images")

if len(images) == 0:
    print("❌ No image data! Run: python3 src/data_collection/collect_images.py")
    exit(1)

X = np.array(images).reshape(-1, 64, 64, 1)
y = np.array(labels)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

num_classes = len(np.unique(y))
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

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
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=16, verbose=1)

os.makedirs(MODELS_DIR, exist_ok=True)
model.save(CNN_CONFIG['model_file'])

metadata = {'gesture_names': STATIC_GESTURES, 'n_samples': len(X), 'image_size': 64}
with open(CNN_CONFIG['metadata_file'], 'wb') as f:
    pickle.dump(metadata, f)

print("✓ CNN model saved!")
