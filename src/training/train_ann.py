"""Train ANN Model"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
import numpy as np
import pickle
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config.config import *

print("=== ANN Training ===")
data_file = os.path.join(TRAINING_DATA_DIR, 'gesture_data.csv')

if not os.path.exists(data_file):
    print("❌ No training data!")
    exit(1)

dataset = np.loadtxt(data_file, delimiter=',', dtype=np.float32)
X, y = dataset[:, :-1], dataset[:, -1]

print(f"✓ Loaded {len(dataset)} samples")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

num_classes = len(np.unique(y))
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=8, verbose=1)

os.makedirs(MODELS_DIR, exist_ok=True)
model.save(ANN_CONFIG['model_file'])

with open(ANN_CONFIG['scaler_file'], 'wb') as f:
    pickle.dump(scaler, f)

metadata = {'gesture_names': STATIC_GESTURES, 'n_samples': len(dataset)}
with open(ANN_CONFIG['metadata_file'], 'wb') as f:
    pickle.dump(metadata, f)

print(f"✓ Model saved!")
