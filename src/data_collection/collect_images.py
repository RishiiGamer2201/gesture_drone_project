"""
CNN Image Data Collection Tool
===============================
Collects hand region IMAGES for training CNN models.
Extracts and saves cropped hand images for each gesture.

DIFFERENCE FROM PREVIOUS COLLECTORS:
- Saves actual IMAGE files (not just coordinates)
- Crops hand region automatically
- Preprocesses for CNN input
- Creates separate folders per gesture
"""

import cv2
import numpy as np
import mediapipe as mp
import os
from datetime import datetime

# Configuration
OUTPUT_DIR = 'data/hand_images'
IMAGE_SIZE = 64  # 64x64 pixels for CNN input

# Gesture definitions
GESTURES = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
    4: "HOVER",
    5: "LAND",
    6: "FORWARD",
    7: "BACKWARD",
    8: "FLIP",
    9: "ROCK"  # NEW: Rock gesture for mode switching
}

# Colors for visual feedback
COLORS = {
    'recording': (0, 255, 0),
    'idle': (255, 255, 0),
    'bbox': (0, 255, 255)
}

class HandImageCollector:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Create directories
        self.setup_directories()
        
        # Counters
        self.gesture_counts = {i: 0 for i in range(10)}  # Now 10 gestures (0-9)
        self.total_images = 0
        
    def setup_directories(self):
        """Create folder structure for images"""
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        
        for gesture_id, gesture_name in GESTURES.items():
            gesture_dir = os.path.join(OUTPUT_DIR, f"{gesture_id}_{gesture_name}")
            if not os.path.exists(gesture_dir):
                os.makedirs(gesture_dir)
                print(f"✓ Created directory: {gesture_dir}")
    
    def get_hand_bbox(self, hand_landmarks, frame_shape):
        """
        Calculate bounding box around hand with padding
        Returns: (x_min, y_min, x_max, y_max)
        """
        h, w = frame_shape[:2]
        
        # Get all landmark coordinates
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
        
        # Find bounding box
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Add padding (20% on each side)
        width = x_max - x_min
        height = y_max - y_min
        padding_x = int(width * 0.2)
        padding_y = int(height * 0.2)
        
        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        x_max = min(w, x_max + padding_x)
        y_max = min(h, y_max + padding_y)
        
        return (x_min, y_min, x_max, y_max)
    
    def extract_hand_image(self, frame, hand_landmarks):
        """
        Extract hand region from frame and preprocess for CNN
        Returns: 64x64 grayscale image
        """
        # Get bounding box
        x_min, y_min, x_max, y_max = self.get_hand_bbox(hand_landmarks, frame.shape)
        
        # Crop hand region
        hand_roi = frame[y_min:y_max, x_min:x_max]
        
        if hand_roi.size == 0:
            return None, None
        
        # Convert to grayscale
        if len(hand_roi.shape) == 3:
            hand_gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
        else:
            hand_gray = hand_roi
        
        # Resize to fixed size
        hand_resized = cv2.resize(hand_gray, (IMAGE_SIZE, IMAGE_SIZE))
        
        # Normalize to 0-255 range (for saving)
        hand_normalized = cv2.normalize(hand_resized, None, 0, 255, cv2.NORM_MINMAX)
        
        return hand_normalized, (x_min, y_min, x_max, y_max)
    
    def save_hand_image(self, image, gesture_id):
        """Save hand image to appropriate folder"""
        gesture_name = GESTURES[gesture_id]
        gesture_dir = os.path.join(OUTPUT_DIR, f"{gesture_id}_{gesture_name}")
        
        # Create unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{gesture_name}_{timestamp}.png"
        filepath = os.path.join(gesture_dir, filename)
        
        # Save image
        cv2.imwrite(filepath, image)
        
        self.gesture_counts[gesture_id] += 1
        self.total_images += 1
        
        return filepath
    
    def draw_ui(self, frame, bbox=None, recording=False, current_gesture=None):
        """Draw user interface on frame"""
        h, w = frame.shape[:2]
        
        # Semi-transparent info panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 280), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Title
        cv2.putText(frame, "CNN IMAGE COLLECTOR", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Instructions
        y_offset = 75
        cv2.putText(frame, "Press 0-9 to capture gesture", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
        cv2.putText(frame, "Hold key to capture multiple", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
        cv2.putText(frame, "Press 'q' to quit", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Gesture counts
        y_offset = 145
        cv2.putText(frame, "Samples Collected:", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 25
        
        for i in range(10):  # Changed to 10 gestures (0-9)
            count = self.gesture_counts[i]
            color = (0, 255, 0) if count >= 50 else (0, 165, 255) if count >= 30 else (100, 100, 100)
            cv2.putText(frame, f"{i}:{GESTURES[i][:3]} {count:3d}", 
                        (20 + (i % 3) * 160, y_offset + (i // 3) * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Total count
        cv2.putText(frame, f"TOTAL: {self.total_images}", (20, 265),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Draw bounding box if hand detected
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            color = COLORS['recording'] if recording else COLORS['bbox']
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            
            if current_gesture is not None:
                cv2.putText(frame, f"Recording: {current_gesture}", 
                            (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Recording indicator
        if recording:
            cv2.circle(frame, (w - 30, 30), 15, (0, 0, 255), -1)
        
        return frame
    
    def run(self):
        """Main collection loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\n" + "=" * 60)
        print("CNN IMAGE DATA COLLECTOR")
        print("=" * 60)
        print("\n📸 Collecting hand images for CNN training")
        print("\nGestures:")
        for key, name in GESTURES.items():
            print(f"  Press '{key}' for {name}")
        print("\nRecommendation: Collect 50-100 images per gesture")
        print("Press 'q' to quit and proceed to training\n")
        print("=" * 60 + "\n")
        
        recording = False
        current_gesture = None
        bbox = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            recording = False
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Draw landmarks
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                # Extract hand image
                hand_image, bbox = self.extract_hand_image(frame, hand_landmarks)
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                
                if key >= ord('0') and key <= ord('9'):  # Changed to include '9'
                    gesture_id = key - ord('0')
                    current_gesture = GESTURES[gesture_id]
                    
                    if hand_image is not None:
                        # Save image
                        filepath = self.save_hand_image(hand_image, gesture_id)
                        recording = True
                        print(f"✓ Saved {current_gesture}: {self.gesture_counts[gesture_id]} samples")
                
                elif key == ord('q'):
                    break
            
            else:
                bbox = None
                current_gesture = None
            
            # Draw UI
            frame = self.draw_ui(frame, bbox, recording, current_gesture)
            
            cv2.imshow("CNN Image Collector", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        self.print_summary()
    
    def print_summary(self):
        """Print collection summary"""
        print("\n" + "=" * 60)
        print("COLLECTION SUMMARY")
        print("=" * 60)
        print(f"\nTotal images collected: {self.total_images}")
        print("\nBreakdown by gesture:")
        for gesture_id, gesture_name in GESTURES.items():
            count = self.gesture_counts[gesture_id]
            status = "✓ Good" if count >= 50 else "⚠ Need more" if count >= 30 else "❌ Insufficient"
            print(f"  {gesture_name:10s}: {count:3d} images - {status}")
        
        print(f"\n📁 Images saved in: {OUTPUT_DIR}/")
        print("\n" + "=" * 60)
        print("Next step: Run 'python train_model_cnn.py' to train CNN")
        print("=" * 60)

if __name__ == "__main__":
    try:
        collector = HandImageCollector()
        collector.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Collection interrupted by user")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
