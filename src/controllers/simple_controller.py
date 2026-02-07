"""
Simple Advanced Drone Controller - Robust Version
==================================================
Fixed version with proper error handling and two-hand detection
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import cv2
import numpy as np
import mediapipe as mp
import pickle
import time

try:
    from config.config import *
except:
    # Fallback config
    class Config:
        MODELS_DIR = 'models'
        STATIC_GESTURES = {0:"UP",1:"DOWN",2:"LEFT",3:"RIGHT",4:"HOVER",5:"LAND",6:"FORWARD",7:"BACKWARD",8:"FLIP"}
    
    STATIC_GESTURES = Config.STATIC_GESTURES
    MODELS_DIR = Config.MODELS_DIR

# ===================================================================
#                         MOCK DRONE
# ===================================================================

class SimpleMockDrone:
    def __init__(self):
        self.is_flying = False
        self.position = {"x": 0, "y": 0, "z": 0}
        self.follow_mode = False
        print("🚁 [DRONE] Initialized")
    
    def connect(self):
        print("🔌 [DRONE] Connected")
        return True
    
    def takeoff(self):
        if not self.is_flying:
            print("🚁 [DRONE] ✈️  TAKEOFF!")
            self.is_flying = True
            self.position["y"] = 100
    
    def land(self):
        if self.is_flying:
            print("🚁 [DRONE] 🛬 LANDING")
            self.is_flying = False
            self.position = {"x": 0, "y": 0, "z": 0}
            self.follow_mode = False
    
    def move_up(self, d=20):
        if self.is_flying:
            self.position["y"] += d
            print(f"🚁 ⬆️  UP (Height: {self.position['y']}cm)")
    
    def move_down(self, d=20):
        if self.is_flying:
            self.position["y"] = max(20, self.position["y"] - d)
            print(f"🚁 ⬇️  DOWN (Height: {self.position['y']}cm)")
    
    def move_left(self, d=20):
        if self.is_flying:
            self.position["x"] -= d
            print(f"🚁 ⬅️  LEFT (X: {self.position['x']}cm)")
    
    def move_right(self, d=20):
        if self.is_flying:
            self.position["x"] += d
            print(f"🚁 ➡️  RIGHT (X: {self.position['x']}cm)")
    
    def move_forward(self, d=20):
        if self.is_flying:
            self.position["z"] += d
            print(f"🚁 ⬆️  FORWARD (Z: {self.position['z']}cm)")
    
    def move_back(self, d=20):
        if self.is_flying:
            self.position["z"] -= d
            print(f"🚁 ⬇️  BACKWARD (Z: {self.position['z']}cm)")
    
    def emergency(self):
        print("🚁 🛑 EMERGENCY STOP!")
        self.is_flying = False
        self.follow_mode = False
        self.position = {"x": 0, "y": 0, "z": 0}
    
    def enable_follow_mode(self):
        if self.is_flying:
            self.follow_mode = True
            print("🚁 👤 FOLLOW MODE ENABLED")
    
    def disable_follow_mode(self):
        self.follow_mode = False
        print("🚁 👤 FOLLOW MODE DISABLED")

# ===================================================================
#                    SIMPLE CONTROLLER
# ===================================================================

class SimpleAdvancedController:
    def __init__(self, model_type='knn'):
        self.model_type = model_type
        self._load_model()
        
        # MediaPipe - CRITICAL FIX: Ensure max_num_hands=2
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # MUST BE 2 for two-hand detection
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        
        self.drone = SimpleMockDrone()
        self.drone.connect()
        
        self.is_flying = False
        self.last_gesture = "NONE"
        self.last_flip_time = 0
        
        print("✓ Controller initialized")
    
    def _load_model(self):
        """Load model"""
        model_file = os.path.join(MODELS_DIR, 'gesture_model_knn.yml')
        if os.path.exists(model_file):
            self.model = cv2.ml.KNearest_load(model_file)
            print("✓ KNN model loaded")
        else:
            print("⚠️  No model found - will use basic hand detection")
            self.model = None
    
    def count_fingers(self, hand_landmarks):
        """Count fingers"""
        tips = [8, 12, 16, 20]
        thumb = 4
        
        count = 0
        
        # Thumb
        if hand_landmarks.landmark[thumb].x < hand_landmarks.landmark[thumb - 1].x:
            count += 1
        
        # Other fingers
        for tip in tips:
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                count += 1
        
        return count
    
    def extract_features(self, hand_landmarks):
        """Extract features"""
        wrist = hand_landmarks.landmark[0]
        features = []
        for lm in hand_landmarks.landmark:
            features.append(lm.x - wrist.x)
            features.append(lm.y - wrist.y)
        return np.array([features], dtype=np.float32)
    
    def predict_gesture(self, hand_landmarks):
        """Predict gesture"""
        if self.model is None:
            return "HOVER", 0.5
        
        try:
            features = self.extract_features(hand_landmarks)
            ret, results, neighbours, dist = self.model.findNearest(features, k=3)
            predicted_id = int(results[0][0])
            confidence = 1.0 - (np.mean(dist) / 1000)
            gesture_name = STATIC_GESTURES.get(predicted_id, "UNKNOWN")
            return gesture_name, confidence
        except Exception as e:
            print(f"Prediction error: {e}")
            return "HOVER", 0.0
    
    def handle_two_hands(self, hands_data):
        """Handle two-hand gestures"""
        if len(hands_data) != 2:
            return None
        
        try:
            f1 = self.count_fingers(hands_data[0])
            f2 = self.count_fingers(hands_data[1])
            
            print(f"DEBUG: Two hands detected - Fingers: {f1}, {f2}")  # Debug
            
            # TAKEOFF - Two open palms
            if f1 == 5 and f2 == 5:
                if not self.is_flying:
                    self.drone.takeoff()
                    self.is_flying = True
                    return "TAKEOFF"
            
            # EMERGENCY - Two fists
            elif f1 == 0 and f2 == 0:
                self.drone.emergency()
                self.is_flying = False
                return "EMERGENCY"
            
            # FOLLOW MODE - One fist + one open
            elif (f1 == 0 and f2 == 5) or (f1 == 5 and f2 == 0):
                if self.is_flying:
                    self.drone.enable_follow_mode()
                    return "FOLLOW_MODE"
        
        except Exception as e:
            print(f"Two-hand error: {e}")
        
        return None
    
    def handle_gesture(self, gesture):
        """Execute gesture command"""
        if not self.is_flying:
            return
        
        if gesture == "UP":
            self.drone.move_up()
        elif gesture == "DOWN":
            self.drone.move_down()
        elif gesture == "LEFT":
            self.drone.move_left()
        elif gesture == "RIGHT":
            self.drone.move_right()
        elif gesture == "FORWARD":
            self.drone.move_forward()
        elif gesture == "BACKWARD":
            self.drone.move_back()
        elif gesture == "LAND":
            self.drone.land()
            self.is_flying = False
        elif gesture == "HOVER":
            pass
    
    def run(self):
        """Main loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\n" + "=" * 60)
        print("SIMPLE ADVANCED DRONE CONTROLLER")
        print("=" * 60)
        print("\n🎮 Controls:")
        print("  ✋✋ Two open palms = TAKEOFF")
        print("  ✊✊ Two fists = EMERGENCY")
        print("  ✊✋ Fist + Open = FOLLOW MODE")
        print("  Single hand gestures = Control")
        print("  'q' = Quit")
        print("\n" + "=" * 60 + "\n")
        
        gesture_text = "NONE"
        safety_text = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera error")
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Process frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            
            safety_text = None
            
            if results.multi_hand_landmarks:
                hands_data = results.multi_hand_landmarks
                
                print(f"DEBUG: Detected {len(hands_data)} hand(s)")  # Debug
                
                # Draw all hands
                for hand in hands_data:
                    self.mp_draw.draw_landmarks(
                        frame, hand, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
                        self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                
                # Two-hand gestures
                if len(hands_data) >= 2:
                    safety_text = self.handle_two_hands(hands_data)
                    if safety_text:
                        print(f"Safety gesture: {safety_text}")
                
                # Single-hand gestures
                elif len(hands_data) == 1 and self.is_flying:
                    gesture_text, confidence = self.predict_gesture(hands_data[0])
                    
                    if confidence > 0.5 and gesture_text != self.last_gesture:
                        self.handle_gesture(gesture_text)
                        self.last_gesture = gesture_text
            
            # Draw UI
            status = "FLYING" if self.is_flying else "GROUNDED"
            status_color = (0, 255, 0) if self.is_flying else (0, 165, 255)
            
            cv2.rectangle(frame, (10, 10), (400, 120), (0, 0, 0), -1)
            cv2.putText(frame, f"Status: {status}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(frame, f"Gesture: {gesture_text}", (20, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            if safety_text:
                cv2.putText(frame, f"Action: {safety_text}", (20, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if self.drone.follow_mode:
                cv2.putText(frame, "FOLLOW MODE", (w//2 - 100, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            
            cv2.imshow("Drone Controller", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if self.is_flying:
            self.drone.land()
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Shutdown complete")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='knn', choices=['knn'])
    args = parser.parse_args()
    
    try:
        controller = SimpleAdvancedController(model_type=args.model)
        controller.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
