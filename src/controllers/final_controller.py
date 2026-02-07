import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque

# Config
STATIC_GESTURES = {0:"UP", 1:"DOWN", 2:"LEFT", 3:"RIGHT", 4:"HOVER", 5:"LAND", 6:"FORWARD", 7:"BACKWARD", 8:"FLIP", 9:"ROCK"}
COOLDOWN_TWO_HAND = 2.0
COOLDOWN_GESTURE = 0.5
COOLDOWN_DYNAMIC = 2.0
COOLDOWN_MODE_SWITCH = 1.5  # Cooldown for mode switching

# Drone
class Drone:
    def __init__(self):
        self.flying = False
        self.pos = {"x":0, "y":100, "z":0}
        self.follow = False
        print("🚁 Drone ready")
    
    def takeoff(self):
        if not self.flying:
            self.flying = True
            self.pos["y"] = 100
            print("\n🚁 ✈️  TAKEOFF!\n")
    
    def land(self):
        if self.flying:
            self.flying = False
            self.pos = {"x":0, "y":0, "z":0}
            print("\n🚁 🛬 LANDING\n")
    
    def emergency(self):
        self.flying = False
        self.follow = False
        print("\n🚁 🛑 EMERGENCY!\n")
    
    def move(self, cmd):
        if not self.flying: return
        if cmd == "UP": self.pos["y"] += 20; print(f"⬆️  UP (Y:{self.pos['y']})")
        elif cmd == "DOWN": self.pos["y"] = max(20, self.pos["y"]-20); print(f"⬇️  DOWN (Y:{self.pos['y']})")
        elif cmd == "LEFT": self.pos["x"] -= 20; print(f"⬅️  LEFT (X:{self.pos['x']})")
        elif cmd == "RIGHT": self.pos["x"] += 20; print(f"➡️  RIGHT (X:{self.pos['x']})")
        elif cmd == "FORWARD": self.pos["z"] += 20; print(f"⬆️  FORWARD (Z:{self.pos['z']})")
        elif cmd == "BACKWARD": self.pos["z"] -= 20; print(f"⬇️  BACKWARD (Z:{self.pos['z']})")

# Dynamic Detector
class DynamicDetector:
    def __init__(self):
        self.poses = deque(maxlen=15)
        self.states = deque(maxlen=15)
        self.last_photo_state = None  # Track last state to prevent double trigger
    
    def add(self, x, y, is_open):
        self.poses.append((x, y))
        self.states.append(is_open)
    
    def circle(self):
        if len(self.poses) < 15: return False
        p = np.array(list(self.poses))
        c = np.mean(p, axis=0)
        r = np.linalg.norm(p - c, axis=1)
        mean_r = np.mean(r)
        if mean_r < 0.05: return False
        return np.std(r) / mean_r < 0.3
    
    def swipe(self):
        if len(self.poses) < 8: return False, None
        d = np.array(self.poses[-1]) - np.array(self.poses[0])
        dist = np.linalg.norm(d)
        if dist < 0.15: return False, None
        angle = np.degrees(np.arctan2(d[1], d[0]))
        if -45 <= angle < 45: return True, "RIGHT"
        elif 45 <= angle < 135: return True, "DOWN"
        elif -135 <= angle < -45: return True, "UP"
        else: return True, "LEFT"
    
    def photo(self):
        """Legacy method - kept for compatibility"""
        is_transition, transition_type = self.detect_photo_transition()
        return is_transition and transition_type == "OPEN"
    
    def detect_photo_transition(self):
        """
        Detect hand opening/closing transition
        Returns: (is_transition, transition_type)
        transition_type can be "OPEN" (fist→open) or "CLOSE" (open→fist)
        """
        if len(self.states) < 10: return False, None
        
        s = list(self.states)
        first_half = s[:len(s)//2]
        second_half = s[len(s)//2:]
        
        first_open = sum(first_half) / len(first_half)
        second_open = sum(second_half) / len(second_half)
        
        # Fist → Open (take photo)
        if first_open < 0.3 and second_open > 0.7:
            if self.last_photo_state != "OPEN":
                self.last_photo_state = "OPEN"
                return True, "OPEN"
        
        # Open → Fist (closing, don't take photo)
        elif first_open > 0.7 and second_open < 0.3:
            if self.last_photo_state != "CLOSE":
                self.last_photo_state = "CLOSE"
                return True, "CLOSE"
        
        return False, None
    
    def clear(self):
        self.poses.clear()
        self.states.clear()
        self.last_photo_state = None  # Reset photo state

# Controller
class Controller:
    def __init__(self):
        # Load model
        try:
            self.model = cv2.ml.KNearest_load('models/gesture_model_knn.yml')
            print("✓ Model loaded")
        except:
            self.model = None
            print("⚠️  No model - using finger count")
        
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        
        # Drone
        self.drone = Drone()
        
        # State
        self.last_two_hand = 0
        self.last_gesture = 0
        self.last_dynamic = 0
        self.last_mode_switch = 0
        self.current_gesture = "NONE"
        
        # MODE SYSTEM - NEW
        self.mode = "STATIC"  # "STATIC" or "DYNAMIC"
        self.mode_locked = False  # Prevents rapid switching
        
        # Dynamic detector
        self.dynamic_detector = DynamicDetector()
        
        print("✓ Ready - STATIC MODE (Default)\n")
    
    def count_fingers(self, hand):
        tips = [8, 12, 16, 20]
        count = 0
        if hand.landmark[4].x < hand.landmark[3].x: count += 1
        for tip in tips:
            if hand.landmark[tip].y < hand.landmark[tip-2].y: count += 1
        return count
    
    def detect_rock_gesture(self, hand):
        """Detect rock gesture (🤘) - index and pinky up, others down"""
        # Index finger up
        index_up = hand.landmark[8].y < hand.landmark[6].y
        # Pinky up
        pinky_up = hand.landmark[20].y < hand.landmark[18].y
        # Middle finger down
        middle_down = hand.landmark[12].y > hand.landmark[10].y
        # Ring finger down
        ring_down = hand.landmark[16].y > hand.landmark[14].y
        # Thumb (flexible - can be up or down)
        
        return index_up and pinky_up and middle_down and ring_down
    
    def switch_mode(self):
        """Switch between STATIC and DYNAMIC modes"""
        t = time.time()
        if t - self.last_mode_switch < COOLDOWN_MODE_SWITCH:
            return
        
        if self.mode == "STATIC":
            self.mode = "DYNAMIC"
            self.dynamic_detector.clear()
            print("\n" + "="*60)
            print("🎬 DYNAMIC MODE ACTIVATED")
            print("="*60)
            print("Drone will HOVER - only dynamic gestures work")
            print("Static gestures DISABLED")
            print("To exit: Show ROCK gesture (🤘) or press 'd'")
            print("="*60 + "\n")
        else:
            self.mode = "STATIC"
            self.dynamic_detector.clear()
            print("\n" + "="*60)
            print("🎯 STATIC MODE ACTIVATED")
            print("="*60)
            print("Static gestures enabled")
            print("Dynamic gestures DISABLED")
            print("You can now control drone normally")
            print("="*60 + "\n")
        
        self.last_mode_switch = t
    
    def count_fingers(self, hand):
        tips = [8, 12, 16, 20]
        count = 0
        if hand.landmark[4].x < hand.landmark[3].x: count += 1
        for tip in tips:
            if hand.landmark[tip].y < hand.landmark[tip-2].y: count += 1
        return count
    
    def predict(self, hand):
        if self.model is None:
            f = self.count_fingers(hand)
            if f == 1: return "UP", 0.8
            elif f == 5: return "LAND", 0.8
            elif f == 0: return "HOVER", 0.8
            return "HOVER", 0.5
        
        try:
            wrist = hand.landmark[0]
            features = []
            for lm in hand.landmark:
                features.append(lm.x - wrist.x)
                features.append(lm.y - wrist.y)
            features = np.array([features], dtype=np.float32)
            ret, results, _, dist = self.model.findNearest(features, k=3)
            gid = int(results[0][0])
            conf = 1.0 - (np.mean(dist) / 1000)
            if conf > 0.6:
                return STATIC_GESTURES.get(gid, "HOVER"), conf
        except:
            pass
        return "HOVER", 0.5
    
    def handle_two_hands(self, hands):
        t = time.time()
        if t - self.last_two_hand < COOLDOWN_TWO_HAND or len(hands) != 2:
            return None
        
        f1, f2 = self.count_fingers(hands[0]), self.count_fingers(hands[1])
        
        if f1 == 5 and f2 == 5 and not self.drone.flying:
            self.drone.takeoff()
            self.last_two_hand = t
            return "TAKEOFF"
        elif f1 == 0 and f2 == 0:
            self.drone.emergency()
            self.last_two_hand = t
            return "EMERGENCY"
        elif (f1 == 0 and f2 == 5) or (f1 == 5 and f2 == 0):
            self.drone.follow = not self.drone.follow
            print(f"👤 Follow mode: {'ON' if self.drone.follow else 'OFF'}")
            self.last_two_hand = t
            return "FOLLOW_TOGGLE"
        return None
    
    def handle_gesture(self, gesture):
        """Handle static gestures - only in STATIC mode"""
        if not self.drone.flying: return
        if self.mode != "STATIC": return  # Block static gestures in dynamic mode
        
        t = time.time()
        if t - self.last_gesture < COOLDOWN_GESTURE: return
        
        if gesture == "LAND":
            self.drone.land()
        elif gesture == "ROCK":
            # Rock gesture switches to dynamic mode
            self.switch_mode()
        else:
            self.drone.move(gesture)
        self.last_gesture = t
        self.current_gesture = gesture
    
    def handle_dynamic(self):
        """Handle dynamic gestures - only in DYNAMIC mode"""
        if self.mode != "DYNAMIC" or not self.drone.flying: return None
        
        t = time.time()
        if t - self.last_dynamic < COOLDOWN_DYNAMIC: return None
        
        # Circle detection
        if self.dynamic_detector.circle():
            print("🔄 CIRCLE detected - Orbit mode")
            self.dynamic_detector.clear()
            self.last_dynamic = t
            return "CIRCLE"
        
        # Swipe detection
        is_swipe, direction = self.dynamic_detector.swipe()
        if is_swipe:
            print(f"👋 SWIPE {direction} - Fast movement")
            # Execute fast movement in swipe direction
            if self.drone.flying:
                if direction == "UP": self.drone.pos["y"] += 40
                elif direction == "DOWN": self.drone.pos["y"] = max(20, self.drone.pos["y"] - 40)
                elif direction == "LEFT": self.drone.pos["x"] -= 40
                elif direction == "RIGHT": self.drone.pos["x"] += 40
                print(f"   Position: X:{self.drone.pos['x']} Y:{self.drone.pos['y']} Z:{self.drone.pos['z']}")
            self.dynamic_detector.clear()
            self.last_dynamic = t
            return f"SWIPE_{direction}"
        
        # Photo capture - ONLY on open transition (fist → open)
        # Don't trigger on close (open → fist)
        is_photo, transition = self.dynamic_detector.detect_photo_transition()
        if is_photo and transition == "OPEN":
            print("📸 PHOTO CAPTURED!")
            self.dynamic_detector.clear()
            self.last_dynamic = t
            return "PHOTO"
        
        return None
    
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("="*60)
        print("FINAL PRODUCTION CONTROLLER - MODE SWITCHING SYSTEM")
        print("="*60)
        print("\n🎮 TWO-HAND GESTURES (Always Active):")
        print("  ✋✋ = TAKEOFF")
        print("  ✊✊ = EMERGENCY")
        print("  ✊✋ = Toggle FOLLOW")
        print("\n🎯 STATIC MODE (Default):")
        print("  Single hand gestures = UP/DOWN/LEFT/RIGHT/etc.")
        print("  🤘 ROCK gesture = Switch to DYNAMIC mode")
        print("  'd' key = Switch to DYNAMIC mode")
        print("\n🎬 DYNAMIC MODE:")
        print("  Drone HOVERS automatically")
        print("  Static gestures DISABLED")
        print("  Circle = Orbit")
        print("  Swipe = Fast movement")
        print("  Fist→Open = Photo capture")
        print("  🤘 ROCK gesture = Switch to STATIC mode")
        print("  'd' key = Switch to STATIC mode")
        print("\n⚠️  TO LAND:")
        print("  1. Switch to STATIC mode (show ROCK 🤘 or press 'd')")
        print("  2. Then show PALM to land")
        print("\n⌨️  KEYS:")
        print("  'q' = Quit")
        print("  'd' = Switch modes")
        print("="*60 + "\n")
        
        safety_text = None
        dynamic_text = None
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            
            num_hands = 0
            safety_text = None
            dynamic_text = None
            
            if results.multi_hand_landmarks:
                hands = results.multi_hand_landmarks
                num_hands = len(hands)
                
                # Draw all hands - BRIGHT COLORS
                for i, hand in enumerate(hands):
                    color = (0, 255, 0) if i == 0 else (255, 0, 255)  # Green for first, magenta for second
                    self.mp_draw.draw_landmarks(
                        frame, hand, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=color, thickness=4, circle_radius=6),
                        self.mp_draw.DrawingSpec(color=(255, 255, 0), thickness=4)
                    )
                
                # Two-hand gestures
                if num_hands >= 2:
                    safety_text = self.handle_two_hands(hands)
                
                # Single-hand gestures
                elif num_hands == 1:
                    hand = hands[0]
                    wrist = hand.landmark[0]
                    
                    # Check for ROCK gesture (mode switch)
                    if self.detect_rock_gesture(hand):
                        t = time.time()
                        if t - self.last_mode_switch >= COOLDOWN_MODE_SWITCH:
                            self.switch_mode()
                    
                    # Update dynamic detector (always, for both modes)
                    is_open = self.count_fingers(hand) >= 4
                    self.dynamic_detector.add(wrist.x, wrist.y, is_open)
                    
                    # MODE: STATIC - only static gestures work
                    if self.mode == "STATIC":
                        gesture, conf = self.predict(hand)
                        self.handle_gesture(gesture)
                        dynamic_text = None
                    
                    # MODE: DYNAMIC - only dynamic gestures work, drone hovers
                    else:
                        # Force hover in dynamic mode
                        self.current_gesture = "HOVER (Dynamic Mode)"
                        # Check dynamic gestures
                        dynamic_text = self.handle_dynamic()
            
            # UI
            cv2.rectangle(frame, (10, 10), (500, 220), (0, 0, 0), -1)
            
            status = "FLYING ✈️" if self.drone.flying else "GROUNDED 🛬"
            color = (0, 255, 0) if self.drone.flying else (100, 100, 100)
            cv2.putText(frame, f"Status: {status}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # MODE display - BIG AND PROMINENT
            mode_color = (255, 128, 0) if self.mode == "DYNAMIC" else (0, 255, 0)
            cv2.putText(frame, f"Mode: {self.mode}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, mode_color, 2)
            
            # Hand count - BIG AND BRIGHT
            hand_color = (0, 255, 255) if num_hands == 2 else (255, 255, 255)
            cv2.putText(frame, f"Hands: {num_hands}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, hand_color, 2)
            
            # Gesture/Dynamic action
            if dynamic_text:
                cv2.putText(frame, f"Dynamic: {dynamic_text}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2)
            else:
                cv2.putText(frame, f"Gesture: {self.current_gesture}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            if safety_text:
                cv2.putText(frame, f"Action: {safety_text}", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Mode indicator banner
            if self.mode == "DYNAMIC":
                cv2.rectangle(frame, (w//2 - 200, 10), (w//2 + 200, 70), (255, 128, 0), -1)
                cv2.putText(frame, "DYNAMIC MODE", (w//2 - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
                cv2.putText(frame, "Show ROCK (🤘) or press 'd' to exit", (w//2 - 190, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Follow mode
            if self.drone.follow:
                cv2.putText(frame, "FOLLOW MODE", (w//2 - 100, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            
            cv2.putText(frame, "q=Quit | d=Toggle Mode | Rock 🤘=Switch", (20, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Gesture Drone Controller", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                self.switch_mode()
        
        if self.drone.flying:
            self.drone.land()
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Done\n")

if __name__ == "__main__":
    try:
        Controller().run()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
