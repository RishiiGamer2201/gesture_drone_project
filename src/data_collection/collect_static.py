"""Static Gesture Data Collection - Final Stabilized Version"""
import sys
import os
import cv2
import numpy as np
import mediapipe as mp

# Ensure project root is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from config.config import *

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

data = []
labels = []
counter = {i: 0 for i in range(9)}

print("=== STATIC GESTURE COLLECTOR ===")
print("Instructions:")
print("1. Position hand in front of webcam until landmarks (dots) appear.")
print("2. Press 0-8 to capture that specific gesture.")
print("3. Press 's' to save all captured data and exit.")
print("4. Press 'q' to quit without saving.")
for i, name in STATIC_GESTURES.items():
    print(f"  {i}: {name}")

# Initialize Webcam
cap = cv2.VideoCapture(0)
# Optional: Force lower resolution if lagging
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    # Reset features for this frame
    current_features = None
    
    # Process landmarks if a hand is found
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Relative coordinate extraction
        wrist = hand_landmarks.landmark[0]
        current_features = []
        for lm in hand_landmarks.landmark:
            current_features.append(lm.x - wrist.x)
            current_features.append(lm.y - wrist.y)
    
    # UI Overlays
    cv2.putText(frame, f"Total Samples: {len(data)}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if not current_features:
        cv2.putText(frame, "HAND NOT DETECTED", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display the window
    cv2.imshow("Static Gesture Collector", frame)
    
    # Handle Keyboard Input
    key = cv2.waitKey(1) & 0xFF
    
    # Capture Logic (0-8)
    if ord('0') <= key <= ord('8'):
        if current_features:
            label = key - ord('0')
            data.append(current_features)
            labels.append(label)
            counter[label] += 1
            print(f"Captured {STATIC_GESTURES[label]} (Total: {counter[label]})")
        else:
            print("Warning: Cannot capture. No hand landmarks detected.")
            
    # Save Logic
    elif key == ord('s'):
        if len(data) > 0:
            dataset = np.hstack((np.array(data), np.array(labels).reshape(-1, 1)))
            os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
            save_path = os.path.join(TRAINING_DATA_DIR, 'gesture_data.csv')
            np.savetxt(save_path, dataset, delimiter=',')
            print(f"\n✓ SUCCESS: Saved {len(data)} samples to {save_path}!")
            break
        else:
            print("Error: No data to save. Capture some gestures first!")
            
    # Quit Logic
    elif key == ord('q'):
        print("Exiting without saving...")
        break

cap.release()
cv2.destroyAllWindows()
