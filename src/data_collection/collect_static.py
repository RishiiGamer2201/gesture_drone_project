"""Static Gesture Data Collection - Simplified"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import cv2
import numpy as np
import mediapipe as mp
from config.config import *

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

data = []
labels = []
counter = {i: 0 for i in range(10)}  # Changed to 10 gestures

print("=== STATIC GESTURE COLLECTOR ===")
print("Press 0-9 to capture gesture, 's' to save, 'q' to quit")
for i, name in STATIC_GESTURES.items():
    print(f"  {i}: {name}")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        
        wrist = hand.landmark[0]
        features = []
        for lm in hand.landmark:
            features.append(lm.x - wrist.x)
            features.append(lm.y - wrist.y)
        
        key = cv2.waitKey(1) & 0xFF
        if ord('0') <= key <= ord('9'):  # Changed to include '9'
            label = key - ord('0')
            data.append(features)
            labels.append(label)
            counter[label] += 1
            print(f"Captured {STATIC_GESTURES[label]}: {counter[label]}")
        elif key == ord('s'):
            if len(data) > 0:
                dataset = np.hstack((np.array(data), np.array(labels).reshape(-1, 1)))
                os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
                np.savetxt(os.path.join(TRAINING_DATA_DIR, 'gesture_data.csv'), dataset, delimiter=',')
                print(f"\n✓ Saved {len(data)} samples!")
                break
        elif key == ord('q'):
            break
    
    cv2.putText(frame, f"Total: {len(data)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Collector", frame)

cap.release()
cv2.destroyAllWindows()
