import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import cv2
import numpy as np
import mediapipe as mp
import pickle
import time
from tensorflow import keras

from config.config import *
from src.utils.gesture_detection import DynamicGestureDetector, HandPoseEstimator
from src.utils.ar_overlay import AROverlay
from src.utils.online_learning import OnlineLearningManager, CorrectionInterface

# ===================================================================
#                         MOCK DRONE CLASS
# ===================================================================

class AdvancedMockDrone:
    """Enhanced mock drone with follow mode and advanced features"""
    
    def __init__(self):
        self.is_flying = False
        self.battery = 100
        self.position = {"x": 0, "y": 0, "z": 0}
        self.follow_mode = False
        self.target_position = None
        print("🚁 [ADVANCED MOCK DRONE] Initialized")
    
    def connect(self):
        print("🔌 [DRONE] Connected")
        return True
    
    def takeoff(self):
        if not self.is_flying:
            print("🚁 [DRONE] ✈️  TAKEOFF!")
            self.is_flying = True
            self.position["y"] = 100
            time.sleep(0.5)
    
    def land(self):
        if self.is_flying:
            print("🚁 [DRONE] 🛬 LANDING...")
            self.is_flying = False
            self.follow_mode = False
            self.position = {"x": 0, "y": 0, "z": 0}
            time.sleep(0.5)
    
    def move_up(self, distance):
        if self.is_flying:
            self.position["y"] += distance
            print(f"🚁 [DRONE] ⬆️  UP {distance}cm (H: {self.position['y']}cm)")
    
    def move_down(self, distance):
        if self.is_flying:
            self.position["y"] = max(20, self.position["y"] - distance)
            print(f"🚁 [DRONE] ⬇️  DOWN {distance}cm (H: {self.position['y']}cm)")
    
    def move_left(self, distance):
        if self.is_flying:
            self.position["x"] -= distance
            print(f"🚁 [DRONE] ⬅️  LEFT {distance}cm")
    
    def move_right(self, distance):
        if self.is_flying:
            self.position["x"] += distance
            print(f"🚁 [DRONE] ➡️  RIGHT {distance}cm)")
    
    def move_forward(self, distance):
        if self.is_flying:
            self.position["z"] += distance
            print(f"🚁 [DRONE] ⬆️  FORWARD {distance}cm")
    
    def move_back(self, distance):
        if self.is_flying:
            self.position["z"] -= distance
            print(f"🚁 [DRONE] ⬇️  BACKWARD {distance}cm")
    
    def rotate_clockwise(self, angle):
        if self.is_flying:
            print(f"🚁 [DRONE] 🔄 ROTATE CW {angle}°")
    
    def rotate_counterclockwise(self, angle):
        if self.is_flying:
            print(f"🚁 [DRONE] 🔄 ROTATE CCW {angle}°")
    
    def flip_back(self):
        if self.is_flying:
            print("🚁 [DRONE] 🔄 BACKFLIP!")
            time.sleep(0.5)
    
    def emergency(self):
        print("🚁 [DRONE] 🛑 EMERGENCY STOP!")
        self.is_flying = False
        self.follow_mode = False
        self.position = {"x": 0, "y": 0, "z": 0}
    
    def enable_follow_mode(self, target_pos):
        """Enable follow mode with target position"""
        if self.is_flying:
            self.follow_mode = True
            self.target_position = target_pos
            print("🚁 [DRONE] 👤 FOLLOW MODE ENABLED")
    
    def disable_follow_mode(self):
        """Disable follow mode"""
        self.follow_mode = False
        self.target_position = None
        print("🚁 [DRONE] 👤 FOLLOW MODE DISABLED")
    
    def follow_target(self, current_hand_pos, frame_width, frame_height):
        """Move drone to maintain distance from target"""
        if not self.follow_mode or not self.is_flying:
            return
        
        # Calculate desired position based on hand position
        # Hand in center = maintain position
        # Hand moves = drone follows
        
        center_x = frame_width / 2
        center_y = frame_height / 2
        
        hand_x = current_hand_pos[0] * frame_width
        hand_y = current_hand_pos[1] * frame_height
        
        # Calculate offset from center
        offset_x = hand_x - center_x
        offset_y = hand_y - center_y
        
        # Threshold to avoid jittery movements
        threshold = 50
        
        if abs(offset_x) > threshold:
            move_dist = int(abs(offset_x) / 10)
            if offset_x > 0:
                self.move_right(move_dist)
            else:
                self.move_left(move_dist)
        
        if abs(offset_y) > threshold:
            move_dist = int(abs(offset_y) / 10)
            if offset_y > 0:
                self.move_down(move_dist)
            else:
                self.move_up(move_dist)
    
    def get_battery(self):
        return self.battery

# ===================================================================
#                    ADVANCED DRONE CONTROLLER
# ===================================================================

class AdvancedDroneController:
    """Advanced controller with all features"""
    
    def __init__(self, model_type='cnn'):
        self.model_type = model_type
        
        # Load model
        self._load_model()
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=HAND_TRACKING['static_image_mode'],
            max_num_hands=HAND_TRACKING['max_num_hands'],
            min_detection_confidence=HAND_TRACKING['min_detection_confidence'],
            min_tracking_confidence=HAND_TRACKING['min_tracking_confidence']
        )
        
        # Initialize drone
        self.drone = AdvancedMockDrone()
        self.drone.connect()
        
        # Initialize detectors
        self.dynamic_detector = DynamicGestureDetector(
            sequence_length=RECOGNITION['dynamic_gesture_frames']
        )
        self.pose_estimator = HandPoseEstimator()
        
        # Initialize AR overlay
        self.ar_overlay = AROverlay(AR_OVERLAY)
        
        # Initialize online learning
        if ONLINE_LEARNING['enabled']:
            self.learning_manager = OnlineLearningManager(
                ONLINE_LEARNING, self.model, self.model_type
            )
            self.correction_interface = CorrectionInterface(
                list(STATIC_GESTURES.values())
            )
        else:
            self.learning_manager = None
            self.correction_interface = None
        
        # Control state
        self.is_flying = False
        self.last_gesture = "NONE"
        self.last_flip_time = 0
        self.last_capture_time = 0
        
        # Gesture smoothing
        self.gesture_buffer = []
        self.buffer_size = RECOGNITION['buffer_size']
        
        # FPS calculation
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = time.time()
        
        print("✓ Advanced controller initialized")
        print(f"   Model: {model_type.upper()}")
        print(f"   Features: Dynamic gestures, Hand pose, Follow mode, AR, Online learning")
    
    def _load_model(self):
        """Load the selected model"""
        if self.model_type == 'cnn':
            if not os.path.exists(CNN_CONFIG['model_file']):
                print("❌ CNN model not found. Please train first.")
                exit(1)
            self.model = keras.models.load_model(CNN_CONFIG['model_file'])
            with open(CNN_CONFIG['metadata_file'], 'rb') as f:
                self.metadata = pickle.load(f)
            self.image_size = self.metadata['image_size']
            print("✓ CNN model loaded")
            
        elif self.model_type == 'ann':
            if not os.path.exists(ANN_CONFIG['model_file']):
                print("❌ ANN model not found. Please train first.")
                exit(1)
            self.model = keras.models.load_model(ANN_CONFIG['model_file'])
            with open(ANN_CONFIG['scaler_file'], 'rb') as f:
                self.scaler = pickle.load(f)
            with open(ANN_CONFIG['metadata_file'], 'rb') as f:
                self.metadata = pickle.load(f)
            print("✓ ANN model loaded")
            
        else:  # knn
            self.model = cv2.ml.KNearest_load(KNN_CONFIG['model_file'])
            with open(KNN_CONFIG['metadata_file'], 'rb') as f:
                self.metadata = pickle.load(f)
            print("✓ KNN model loaded")
    
    def count_fingers(self, hand_landmarks):
        """Count extended fingers"""
        finger_tips = [8, 12, 16, 20]
        thumb_tip = 4
        
        fingers_up = 0
        
        if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_tip - 1].x:
            fingers_up += 1
        
        for tip in finger_tips:
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                fingers_up += 1
        
        return fingers_up
    
    def extract_hand_image(self, frame, hand_landmarks):
        """Extract hand region for CNN"""
        if self.model_type != 'cnn':
            return None
        
        h, w = frame.shape[:2]
        
        # Get bounding box
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        # Crop and preprocess
        hand_roi = frame[y_min:y_max, x_min:x_max]
        
        if hand_roi.size == 0:
            return None
        
        hand_gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
        hand_resized = cv2.resize(hand_gray, (self.image_size, self.image_size))
        hand_normalized = hand_resized.astype('float32') / 255.0
        hand_input = hand_normalized.reshape(1, self.image_size, self.image_size, 1)
        
        return hand_input
    
    def extract_landmarks(self, hand_landmarks):
        """Extract normalized landmarks for ANN/KNN"""
        wrist_x = hand_landmarks.landmark[0].x
        wrist_y = hand_landmarks.landmark[0].y
        
        features = []
        for lm in hand_landmarks.landmark:
            features.append(lm.x - wrist_x)
            features.append(lm.y - wrist_y)
        
        return np.array([features], dtype=np.float32)
    
    def predict_static_gesture(self, frame, hand_landmarks):
        """Predict static gesture using loaded model"""
        if self.model_type == 'cnn':
            features = self.extract_hand_image(frame, hand_landmarks)
            if features is None:
                return "NO_HAND", 0.0
        else:
            features = self.extract_landmarks(hand_landmarks)
            if self.model_type == 'ann':
                features = self.scaler.transform(features)
        
        # Get prediction
        if self.model_type == 'knn':
            ret, results, neighbours, dist = self.model.findNearest(features, k=3)
            predicted_id = int(results[0][0])
            confidence = 1.0 - (np.mean(dist) / 1000)  # Rough confidence
        else:
            predictions = self.model.predict(features, verbose=0)[0]
            predicted_id = np.argmax(predictions)
            confidence = predictions[predicted_id]
        
        gesture_name = STATIC_GESTURES.get(predicted_id, "UNKNOWN")
        
        return gesture_name, confidence, features
    
    def handle_two_hand_gestures(self, hands_data):
        """Handle two-handed gestures including follow mode"""
        if len(hands_data) != 2:
            return None
        
        fingers_1 = self.count_fingers(hands_data[0])
        fingers_2 = self.count_fingers(hands_data[1])
        
        # TAKEOFF - Two open palms
        if fingers_1 == 5 and fingers_2 == 5:
            if not self.is_flying:
                self.drone.takeoff()
                self.is_flying = True
                return "TAKEOFF"
        
        # EMERGENCY - Two fists
        elif fingers_1 == 0 and fingers_2 == 0:
            self.drone.emergency()
            self.is_flying = False
            return "EMERGENCY"
        
        # FOLLOW MODE - Left fist + Right open
        elif fingers_1 == 0 and fingers_2 == 5:
            if self.is_flying:
                # Use right hand position as target
                wrist_pos = (hands_data[1].landmark[0].x, hands_data[1].landmark[0].y)
                self.drone.enable_follow_mode(wrist_pos)
                return "FOLLOW_MODE"
        
        # Disable follow mode if different gesture
        elif self.drone.follow_mode:
            self.drone.disable_follow_mode()
        
        return None
    
    def handle_flight_gesture(self, gesture, hand_orientation=None):
        """Execute drone command with hand orientation support"""
        if not self.is_flying:
            return
        
        # Get movement distance (can be modified by hand distance)
        move_dist = DRONE_CONTROL['move_distance']
        
        # Execute gesture command
        if gesture == "UP":
            self.drone.move_up(move_dist)
        elif gesture == "DOWN":
            self.drone.move_down(move_dist)
        elif gesture == "LEFT":
            self.drone.move_left(move_dist)
        elif gesture == "RIGHT":
            self.drone.move_right(move_dist)
        elif gesture == "FORWARD":
            self.drone.move_forward(move_dist)
        elif gesture == "BACKWARD":
            self.drone.move_back(move_dist)
        elif gesture == "LAND":
            self.drone.land()
            self.is_flying = False
        elif gesture == "FLIP":
            current_time = time.time()
            if current_time - self.last_flip_time > DRONE_CONTROL['flip_cooldown']:
                self.drone.flip_back()
                self.last_flip_time = current_time
        elif gesture == "HOVER":
            pass
        
        # Hand orientation control
        if hand_orientation and HAND_POSE['estimate_orientation']:
            self._handle_orientation_control(hand_orientation)
    
    def _handle_orientation_control(self, orientation):
        """Control drone rotation based on hand tilt"""
        roll = orientation.get('roll', 0)
        
        # Rotate if hand is tilted significantly
        threshold = 30  # degrees
        if abs(roll) > threshold:
            angle = int(abs(roll) / 10)
            if roll > 0:
                self.drone.rotate_clockwise(angle)
            else:
                self.drone.rotate_counterclockwise(angle)
    
    def handle_dynamic_gesture(self, gesture_name, confidence):
        """Handle dynamic gestures"""
        if not self.is_flying:
            return
        
        print(f"🎬 Dynamic gesture: {gesture_name} ({confidence:.2f})")
        
        if gesture_name == "CIRCLE":
            print("🚁 [DRONE] 🔄 ORBIT MODE")
            # Could implement orbit around a point
        
        elif "SWIPE" in gesture_name:
            direction = gesture_name.split("_")[1]
            if direction == "LEFT":
                self.drone.move_left(DRONE_CONTROL['speed_fast'])
            elif direction == "RIGHT":
                self.drone.move_right(DRONE_CONTROL['speed_fast'])
            elif direction == "UP":
                self.drone.move_up(DRONE_CONTROL['speed_fast'])
            elif direction == "DOWN":
                self.drone.move_down(DRONE_CONTROL['speed_fast'])
        
        elif gesture_name == "OPEN_CLOSE":
            # Capture image
            current_time = time.time()
            if current_time - self.last_capture_time > 2.0:
                print("📸 [DRONE] CAPTURING IMAGE")
                self.last_capture_time = current_time
        
        elif gesture_name == "WAVE":
            print("👋 [DRONE] WAVE DETECTED - RETURN TO HOME")
            # Reset position
            self.drone.position = {"x": 0, "y": 100, "z": 0}
    
    def run(self):
        """Main control loop"""
        cap = cv2.VideoCapture(CAMERA['camera_id'])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA['height'])
        
        print("\n" + "=" * 70)
        print("ADVANCED GESTURE DRONE CONTROLLER")
        print("=" * 70)
        print("\n✨ Features enabled:")
        print("   • Dynamic gestures (Circle, Swipe, Open/Close, Wave)")
        print("   • Hand pose estimation")
        print("   • Follow mode (Left fist + Right open)")
        print("   • AR overlay")
        print("   • Online learning")
        print("\n🎮 Controls:")
        print("   Two open palms (✋✋) = TAKEOFF")
        print("   Two fists (✊✊) = EMERGENCY")
        print("   Left fist + Right open = FOLLOW MODE")
        print("   'c' = Enable correction mode")
        print("   'q' = Quit")
        print("\n" + "=" * 70 + "\n")
        
        current_gesture = "NONE"
        confidence = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hands
            results = self.hands.process(rgb_frame)
            
            safety_status = None
            dynamic_gesture = None
            
            if results.multi_hand_landmarks:
                hands_data = results.multi_hand_landmarks
                
                # Check two-hand gestures (with error handling)
                try:
                    safety_status = self.handle_two_hand_gestures(hands_data)
                except Exception as e:
                    print(f"Warning: Two-hand detection error: {e}")
                    safety_status = None
                
                # Process single hand
                if len(hands_data) == 1 and not safety_status:
                    hand = hands_data[0]
                    
                    try:
                        # Update dynamic gesture detector
                        self.dynamic_detector.add_frame(hand, 
                            self.pose_estimator.get_hand_state(hand))
                        
                        # Get hand pose
                        hand_orientation = self.pose_estimator.estimate_hand_rotation(hand)
                        hand_pos_3d = self.pose_estimator.get_hand_3d_position(hand, w, h)
                        
                        # Add to AR trajectory
                        wrist = hand.landmark[0]
                        self.ar_overlay.add_trajectory_point((wrist.x * w, wrist.y * h))
                    except Exception as e:
                        print(f"Warning: Hand processing error: {e}")
                        hand_orientation = None
                        hand_pos_3d = None
                    
                    if self.is_flying:
                        try:
                            # Check for dynamic gesture
                            dynamic_gesture, dyn_conf = self.dynamic_detector.detect_all()
                            
                            if dynamic_gesture:
                                self.handle_dynamic_gesture(dynamic_gesture, dyn_conf)
                                if AR_OVERLAY['enabled']:
                                    self.ar_overlay.draw_dynamic_gesture_trail(frame, dynamic_gesture)
                            else:
                                # Static gesture
                                current_gesture, confidence, features = self.predict_static_gesture(
                                    frame, hand)
                                
                                if confidence > RECOGNITION['confidence_threshold']:
                                    if current_gesture != self.last_gesture:
                                        self.handle_flight_gesture(current_gesture, hand_orientation)
                                        self.last_gesture = current_gesture
                        except Exception as e:
                            print(f"Warning: Gesture recognition error: {e}")
                        
                        # Follow mode
                        try:
                            if self.drone.follow_mode:
                                self.drone.follow_target((wrist.x, wrist.y), w, h)
                                if AR_OVERLAY['enabled']:
                                    self.ar_overlay.draw_follow_mode_indicator(
                                        frame, (wrist.x, wrist.y), 
                                        DRONE_CONTROL['follow_mode_distance']
                                    )
                        except Exception as e:
                            print(f"Warning: Follow mode error: {e}")
                    
                    # Draw AR overlays
                    try:
                        if AR_OVERLAY['enabled']:
                            if AR_OVERLAY['show_hand_skeleton']:
                                self.ar_overlay.draw_hand_skeleton(
                                    frame, hand, self.mp_hands, self.mp_draw)
                            if AR_OVERLAY['show_trajectory']:
                                frame = self.ar_overlay.draw_trajectory(frame)
                            if AR_OVERLAY['show_confidence'] and confidence:
                                frame = self.ar_overlay.draw_confidence_meter(frame, confidence)
                            if AR_OVERLAY['show_position']:
                                self.ar_overlay.add_drone_position(self.drone.position)
                                frame = self.ar_overlay.draw_drone_position_3d(
                                    frame, self.drone.position)
                    except Exception as e:
                        print(f"Warning: AR overlay error: {e}")
                
                # Draw landmarks for all hands
                for hand_landmarks in hands_data:
                    if not AR_OVERLAY['show_hand_skeleton']:
                        self.mp_draw.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Draw UI
            frame = self._draw_main_ui(frame, current_gesture, confidence, 
                                       safety_status, dynamic_gesture)
            
            # Correction interface
            if self.learning_manager and self.correction_interface.correction_mode:
                frame = self.learning_manager.display_correction_ui(
                    frame, current_gesture, list(STATIC_GESTURES.values()))
            
            # FPS
            if VISUALIZATION['show_fps']:
                frame = self.ar_overlay.draw_fps(frame, self.fps)
            
            cv2.imshow(VISUALIZATION['window_name'], frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c') and self.learning_manager:
                # Enable correction mode
                if current_gesture != "NONE":
                    self.correction_interface.enable_correction_mode(
                        features if 'features' in locals() else None,
                        current_gesture
                    )
            elif self.correction_interface and self.correction_interface.correction_mode:
                # Check for correction
                corrected_label, corr_features = self.correction_interface.check_for_correction(key)
                if corrected_label is not None and corr_features is not None:
                    predicted_id = list(STATIC_GESTURES.values()).index(current_gesture)
                    self.learning_manager.add_correction(
                        corr_features, predicted_id, corrected_label)
            
            # Update FPS
            self.frame_count += 1
            if self.frame_count % 10 == 0:
                current_time = time.time()
                self.fps = 10 / (current_time - self.fps_start_time)
                self.fps_start_time = current_time
        
        # Cleanup
        if self.is_flying:
            self.drone.land()
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Controller shutdown complete")
    
    def _draw_main_ui(self, frame, gesture, confidence, safety, dynamic):
        """Draw main UI elements"""
        h, w = frame.shape[:2]
        
        # Status panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Title
        cv2.putText(frame, "ADVANCED CONTROLLER", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Status
        status_text = "FLYING" if self.is_flying else "GROUNDED"
        status_color = (0, 255, 0) if self.is_flying else (0, 165, 255)
        cv2.putText(frame, f"Status: {status_text}", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Gesture
        if dynamic:
            cv2.putText(frame, f"Dynamic: {dynamic}", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)
        else:
            cv2.putText(frame, f"Gesture: {gesture}", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Safety
        if safety:
            cv2.putText(frame, f"Action: {safety}", (20, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Follow mode indicator
        if self.drone.follow_mode:
            cv2.putText(frame, "FOLLOW MODE", (w//2 - 100, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
        return frame

# ===================================================================
#                              MAIN
# ===================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Gesture Drone Controller')
    parser.add_argument('--model', type=str, default='cnn',
                       choices=['knn', 'ann', 'cnn'],
                       help='Model type to use')
    
    args = parser.parse_args()
    
    try:
        controller = AdvancedDroneController(model_type=args.model)
        controller.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
