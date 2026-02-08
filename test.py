import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque
import os
import pickle
from tensorflow import keras

class Config:
    STATIC_GESTURES = {
        0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "HOVER",
        5: "LAND", 6: "FORWARD", 7: "BACKWARD", 8: "FLIP", 9: "ROCK"
    }
    COOLDOWN_TWO_HAND = 2.0
    COOLDOWN_GESTURE = 0.5
    COOLDOWN_DYNAMIC = 2.0
    COOLDOWN_MODE_SWITCH = 1.5
    CONFIDENCE_THRESHOLD = 0.75
    DYNAMIC_GESTURE_MIN_FRAMES = 15
    MOVE_DISTANCE = 20
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    CAMERA_ID = 0
    COLOR_GREEN = (0, 255, 0)
    COLOR_RED = (0, 0, 255)
    COLOR_BLUE = (255, 0, 0)
    COLOR_YELLOW = (0, 255, 255)
    COLOR_ORANGE = (0, 165, 255)
    COLOR_WHITE = (255, 255, 255)
    MODEL_KNN_PATH = 'models/gesture_model_knn.yml'
    MODEL_CNN_PATH = 'models/gesture_model_cnn.h5'
    MODEL_ANN_PATH = 'models/gesture_model_ann.pkl'

class MockDrone:
    def __init__(self):
        self.is_flying = False
        self.position = {"x": 0, "y": 100, "z": 0}
        self.follow_mode = False
        self.battery = 100
        print("MOCK DRONE Initialized")
    
    def connect(self):
        print("DRONE Connected")
        return True
    
    def takeoff(self):
        if not self.is_flying:
            print("\nTAKEOFF - Drone is now FLYING!\n")
            self.is_flying = True
            self.position["y"] = 100
            return True
        return False
    
    def land(self):
        if self.is_flying:
            print("\nLANDING - Drone has landed\n")
            self.is_flying = False
            self.follow_mode = False
            self.position = {"x": 0, "y": 0, "z": 0}
            return True
        return False
    
    def emergency(self):
        print("\nEMERGENCY STOP!\n")
        self.is_flying = False
        self.follow_mode = False
        self.position = {"x": 0, "y": 0, "z": 0}
    
    def move(self, direction, distance=20):
        if not self.is_flying:
            return False
        if direction == "UP":
            self.position["y"] += distance
            print(f"UP → Height: {self.position['y']}cm")
        elif direction == "DOWN":
            self.position["y"] = max(20, self.position["y"] - distance)
            print(f"DOWN → Height: {self.position['y']}cm")
        elif direction == "LEFT":
            self.position["x"] -= distance
            print(f"LEFT → X: {self.position['x']}cm")
        elif direction == "RIGHT":
            self.position["x"] += distance
            print(f"RIGHT → X: {self.position['x']}cm")
        elif direction == "FORWARD":
            self.position["z"] += distance
            print(f"FORWARD → Z: {self.position['z']}cm")
        elif direction == "BACKWARD":
            self.position["z"] -= distance
            print(f"BACKWARD → Z: {self.position['z']}cm")
        return True
    
    def flip(self):
        if self.is_flying:
            print("BACKFLIP!")
            time.sleep(0.5)
    
    def enable_follow_mode(self):
        if self.is_flying:
            self.follow_mode = True
            print("\nFOLLOW MODE ACTIVATED\n")
    
    def disable_follow_mode(self):
        if self.follow_mode:
            self.follow_mode = False
            print("\nFOLLOW MODE DEACTIVATED\n")

class RoboticHandHUD:
    def __init__(self, position=(20, 500), size=150):
        self.x, self.y = position
        self.size = size
        self.COLOR_BONE = (0, 255, 255)
        self.COLOR_JOINT = (0, 165, 255)
        self.COLOR_TIP = (0, 0, 255)
        self.COLOR_BG = (0, 20, 0)
        self.CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17), (0, 5), (0, 17)
        ]
    
    def draw(self, frame, multi_hand_landmarks):
        if not multi_hand_landmarks:
            return frame
        box_width = (self.size * len(multi_hand_landmarks)) + 20
        box_height = self.size + 20
        overlay = frame.copy()
        cv2.rectangle(overlay, (self.x, self.y),
                     (self.x + box_width, self.y + box_height),
                     self.COLOR_BG, -1)
        cv2.rectangle(overlay, (self.x, self.y),
                     (self.x + box_width, self.y + box_height),
                     (0, 255, 0), 1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, "HUD LINK", (self.x + 5, self.y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        for i, landmarks in enumerate(multi_hand_landmarks):
            x_offset = self.x + 10 + (i * self.size)
            y_offset = self.y + 10
            self._draw_single_hand(frame, landmarks, x_offset, y_offset)
        return frame
    
    def _draw_single_hand(self, frame, landmarks, x_start, y_start):
        x_coords = [lm.x for lm in landmarks.landmark]
        y_coords = [lm.y for lm in landmarks.landmark]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        hand_width = max_x - min_x
        hand_height = max_y - min_y
        if hand_width == 0 or hand_height == 0:
            return
        scale = (self.size - 40) / max(hand_width, hand_height)
        points = {}
        for idx, lm in enumerate(landmarks.landmark):
            rel_x = lm.x - min_x
            rel_y = lm.y - min_y
            draw_x = int(x_start + 20 + (rel_x * scale))
            draw_y = int(y_start + 20 + (rel_y * scale))
            points[idx] = (draw_x, draw_y)
        for start_idx, end_idx in self.CONNECTIONS:
            if start_idx in points and end_idx in points:
                cv2.line(frame, points[start_idx], points[end_idx], self.COLOR_BONE, 1)
        for idx, point in points.items():
            radius = 2
            color = self.COLOR_JOINT
            if idx in [4, 8, 12, 16, 20]:
                radius = 4
                color = self.COLOR_TIP
            cv2.circle(frame, point, radius, color, -1)

class DynamicGestureDetector:
    def __init__(self, buffer_size=20):
        self.buffer_size = buffer_size
        self.position_history = deque(maxlen=buffer_size)
        self.hand_state_history = deque(maxlen=buffer_size)
        self.last_photo_state = None
    
    def add_frame(self, x, y, is_open):
        self.position_history.append((x, y))
        self.hand_state_history.append(1 if is_open else 0)
    
    def detect_circle(self):
        if len(self.position_history) < self.buffer_size:
            return False
        positions = np.array(list(self.position_history))
        std_dev = np.std(positions, axis=0)
        if std_dev[0] < 0.05 or std_dev[1] < 0.05:
            return False
        center = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - center, axis=1)
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        if std_distance / mean_distance > 0.25:
            return False
        return True
    
    def detect_swipe(self):
        if len(self.position_history) < 10:
            return False, None
        recent_pos = list(self.position_history)[-10:]
        start = np.array(recent_pos[0])
        end = np.array(recent_pos[-1])
        displacement = end - start
        distance = np.linalg.norm(displacement)
        if distance < 0.2:
            return False, None
        angle = np.degrees(np.arctan2(displacement[1], displacement[0]))
        if -45 <= angle < 45:
            return True, "RIGHT"
        elif 45 <= angle < 135:
            return True, "DOWN"
        elif -135 <= angle < -45:
            return True, "UP"
        else:
            return True, "LEFT"
    
    def detect_photo_transition(self):
        if len(self.hand_state_history) < 15:
            return False, None
        states = list(self.hand_state_history)
        mid = len(states) // 2
        early_avg = sum(states[:mid]) / len(states[:mid])
        late_avg = sum(states[mid:]) / len(states[mid:])
        if early_avg < 0.2 and late_avg > 0.8:
            if self.last_photo_state != "OPEN":
                self.last_photo_state = "OPEN"
                return True, "OPEN"
        elif early_avg > 0.8 and late_avg < 0.2:
            if self.last_photo_state != "CLOSE":
                self.last_photo_state = "CLOSE"
                return True, "CLOSE"
        return False, None
    
    def detect_wave(self, min_oscillations=3):
        if len(self.position_history) < self.buffer_size:
            return False
        open_ratio = sum(self.hand_state_history) / len(self.hand_state_history)
        if open_ratio < 0.8:
            return False
        x_positions = [pos[0] for pos in self.position_history]
        min_x, max_x = min(x_positions), max(x_positions)
        if (max_x - min_x) < 0.2:
            return False
        peaks = 0
        troughs = 0
        for i in range(1, len(x_positions) - 1):
            if x_positions[i] > x_positions[i-1] and x_positions[i] > x_positions[i+1]:
                peaks += 1
            elif x_positions[i] < x_positions[i-1] and x_positions[i] < x_positions[i+1]:
                troughs += 1
        return (peaks + troughs) >= min_oscillations
    
    def clear(self):
        self.position_history.clear()
        self.hand_state_history.clear()
        self.last_photo_state = None

class HandPoseEstimator:
    def __init__(self):
        self.palm_size_baseline = None
    
    def get_hand_openness(self, hand_landmarks):
        palm_center = hand_landmarks.landmark[0]
        fingertips = [8, 12, 16, 20]
        distances = []
        for tip_idx in fingertips:
            tip = hand_landmarks.landmark[tip_idx]
            distance = np.sqrt(
                (tip.x - palm_center.x)**2 +
                (tip.y - palm_center.y)**2
            )
            distances.append(distance)
        avg_distance = np.mean(distances)
        openness = min(1.0, avg_distance / 0.2)
        return openness
    
    def get_hand_state(self, hand_landmarks):
        openness = self.get_hand_openness(hand_landmarks)
        return "open" if openness > 0.5 else "closed"

class MLGestureClassifier:
    def __init__(self):
        self.model_knn = None
        self.model_cnn = None
        self.model_ann = None
        self.model_type = None
        self.load_models()
    
    def load_models(self):
        if os.path.exists(Config.MODEL_KNN_PATH):
            try:
                self.model_knn = cv2.ml.KNearest_load(Config.MODEL_KNN_PATH)
                self.model_type = 'KNN'
                print(f"Loaded KNN model from {Config.MODEL_KNN_PATH}")
            except Exception as e:
                print(f"Failed to load KNN model: {e}")
        
        if os.path.exists(Config.MODEL_CNN_PATH):
            try:
                self.model_cnn = keras.models.load_model(Config.MODEL_CNN_PATH)
                if self.model_type is None:
                    self.model_type = 'CNN'
                print(f"Loaded CNN model from {Config.MODEL_CNN_PATH}")
            except Exception as e:
                print(f"Failed to load CNN model: {e}")
        
        if os.path.exists(Config.MODEL_ANN_PATH):
            try:
                with open(Config.MODEL_ANN_PATH, 'rb') as f:
                    self.model_ann = pickle.load(f)
                if self.model_type is None:
                    self.model_type = 'ANN'
                print(f"Loaded ANN model from {Config.MODEL_ANN_PATH}")
            except Exception as e:
                print(f"Failed to load ANN model: {e}")
        
        if self.model_type is None:
            print("No trained models found. Using fallback finger counting.")
            self.model_type = 'FALLBACK'
    
    def count_fingers(self, hand_landmarks):
        fingertips = [8, 12, 16, 20]
        thumb_tip = 4
        count = 0
        if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_tip - 1].x:
            count += 1
        for tip_idx in fingertips:
            if hand_landmarks.landmark[tip_idx].y < hand_landmarks.landmark[tip_idx - 2].y:
                count += 1
        return count
    
    def detect_rock_gesture(self, hand_landmarks):
        index_up = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
        pinky_up = hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y
        middle_down = hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y
        ring_down = hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y
        return index_up and pinky_up and middle_down and ring_down
    
    def extract_features(self, hand_landmarks):
        wrist = hand_landmarks.landmark[0]
        features = []
        for lm in hand_landmarks.landmark:
            features.append(lm.x - wrist.x)
            features.append(lm.y - wrist.y)
        return np.array(features, dtype=np.float32)
    
    def extract_hand_image(self, frame, hand_landmarks):
        h, w = frame.shape[:2]
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
        x_min = max(0, int(min(x_coords)) - 20)
        x_max = min(w, int(max(x_coords)) + 20)
        y_min = max(0, int(min(y_coords)) - 20)
        y_max = min(h, int(max(y_coords)) + 20)
        hand_roi = frame[y_min:y_max, x_min:x_max]
        if hand_roi.size == 0:
            return None
        hand_gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
        hand_resized = cv2.resize(hand_gray, (64, 64))
        hand_normalized = hand_resized / 255.0
        return hand_normalized.reshape(1, 64, 64, 1)
    
    def classify_gesture(self, hand_landmarks, frame=None):
        if self.detect_rock_gesture(hand_landmarks):
            return "ROCK", 0.95
        
        if self.model_type == 'KNN' and self.model_knn is not None:
            try:
                features = self.extract_features(hand_landmarks).reshape(1, -1)
                ret, results, neighbours, dist = self.model_knn.findNearest(features, k=3)
                predicted_id = int(results[0][0])
                confidence = 1.0 - (np.mean(dist) / 1000)
                if predicted_id in Config.STATIC_GESTURES:
                    return Config.STATIC_GESTURES[predicted_id], confidence
            except Exception as e:
                print(f"KNN prediction error: {e}")
        
        elif self.model_type == 'CNN' and self.model_cnn is not None and frame is not None:
            try:
                hand_image = self.extract_hand_image(frame, hand_landmarks)
                if hand_image is not None:
                    predictions = self.model_cnn.predict(hand_image, verbose=0)
                    predicted_id = np.argmax(predictions[0])
                    confidence = float(predictions[0][predicted_id])
                    if predicted_id in Config.STATIC_GESTURES:
                        return Config.STATIC_GESTURES[predicted_id], confidence
            except Exception as e:
                print(f"CNN prediction error: {e}")
        
        elif self.model_type == 'ANN' and self.model_ann is not None:
            try:
                features = self.extract_features(hand_landmarks).reshape(1, -1)
                predictions = self.model_ann.predict_proba(features)
                predicted_id = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_id])
                if predicted_id in Config.STATIC_GESTURES:
                    return Config.STATIC_GESTURES[predicted_id], confidence
            except Exception as e:
                print(f"ANN prediction error: {e}")
        
        fingers_up = self.count_fingers(hand_landmarks)
        if fingers_up == 1:
            return "UP", 0.7
        elif fingers_up == 5:
            return "LAND", 0.8
        elif fingers_up == 0:
            return "HOVER", 0.8
        elif fingers_up == 2:
            return "FLIP", 0.7
        else:
            return "HOVER", 0.5

class AROverlay:
    def __init__(self):
        self.trajectory_history = deque(maxlen=30)
        self.colors = Config()
    
    def add_trajectory_point(self, point):
        self.trajectory_history.append(point)
    
    def draw_trajectory(self, frame):
        if len(self.trajectory_history) < 2:
            return frame
        points = list(self.trajectory_history)
        for i in range(1, len(points)):
            alpha = i / len(points)
            color = tuple(int(c * alpha) for c in self.colors.COLOR_YELLOW)
            pt1 = (int(points[i-1][0]), int(points[i-1][1]))
            pt2 = (int(points[i][0]), int(points[i][1]))
            thickness = max(1, int(3 * alpha))
            cv2.line(frame, pt1, pt2, color, thickness)
        return frame
    
    def draw_3d_position(self, frame, position, size=(200, 200)):
        h, w = frame.shape[:2]
        x_offset = w - size[0] - 20
        y_offset = h - size[1] - 20
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_offset, y_offset),
                     (x_offset + size[0], y_offset + size[1]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        center_x = x_offset + size[0] // 2
        center_y = y_offset + size[1] // 2
        for i in range(-2, 3):
            y = center_y + i * 30
            cv2.line(frame, (x_offset + 20, y), (x_offset + size[0] - 20, y), (100, 100, 100), 1)
            x = center_x + i * 30
            cv2.line(frame, (x, y_offset + 20), (x, y_offset + size[1] - 20), (100, 100, 100), 1)
        drone_x = int(center_x + position['x'] * 0.5)
        drone_y = int(center_y - position['y'] * 0.3)
        cv2.circle(frame, (drone_x, drone_y), 10, self.colors.COLOR_GREEN, -1)
        cv2.circle(frame, (drone_x, drone_y), 12, self.colors.COLOR_WHITE, 2)
        cv2.putText(frame, "DRONE TELEMETRY", (x_offset + 10, y_offset + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors.COLOR_WHITE, 1)
        cv2.putText(frame, f"X:{position['x']:.0f}", (x_offset + 10, y_offset + size[1] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors.COLOR_WHITE, 1)
        cv2.putText(frame, f"Y:{position['y']:.0f}", (x_offset + 10, y_offset + size[1] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors.COLOR_WHITE, 1)
        cv2.putText(frame, f"Z:{position['z']:.0f}", (x_offset + 10, y_offset + size[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors.COLOR_WHITE, 1)
        return frame

class GestureDroneController:
    def __init__(self):
        print("Initializing Gesture Drone Controller with ML Models")
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        self.drone = MockDrone()
        self.drone.connect()
        self.dynamic_detector = DynamicGestureDetector()
        self.pose_estimator = HandPoseEstimator()
        self.gesture_classifier = MLGestureClassifier()
        self.ar_overlay = AROverlay()
        self.robotic_hud = RoboticHandHUD(position=(30, 480), size=150)
        self.is_flying = False
        self.mode = "STATIC"
        self.last_two_hand_time = 0
        self.last_gesture_time = 0
        self.last_dynamic_time = 0
        self.last_mode_switch_time = 0
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = time.time()
        print(f"Controller initialized with {self.gesture_classifier.model_type} model")
    
    def count_fingers(self, hand_landmarks):
        return self.gesture_classifier.count_fingers(hand_landmarks)
    
    def is_hand_open(self, hand_landmarks):
        return self.count_fingers(hand_landmarks) >= 4
    
    def switch_mode(self):
        current_time = time.time()
        if current_time - self.last_mode_switch_time < Config.COOLDOWN_MODE_SWITCH:
            return
        if self.mode == "STATIC":
            self.mode = "DYNAMIC"
            self.dynamic_detector.clear()
            print("\nDYNAMIC MODE ACTIVATED")
            print("Drone will HOVER. Use Circle, Swipe, Wave, or Photo gestures.")
        else:
            self.mode = "STATIC"
            self.dynamic_detector.clear()
            print("\nSTATIC MODE ACTIVATED")
            print("Use gestures for precise control.")
        self.last_mode_switch_time = current_time
    
    def handle_two_hand_gestures(self, hands_data):
        if len(hands_data) != 2:
            return None
        current_time = time.time()
        if current_time - self.last_two_hand_time < Config.COOLDOWN_TWO_HAND:
            return None
        fingers_1 = self.count_fingers(hands_data[0])
        fingers_2 = self.count_fingers(hands_data[1])
        action = None
        if fingers_1 == 5 and fingers_2 == 5 and not self.is_flying:
            self.drone.takeoff()
            self.is_flying = True
            action = "TAKEOFF"
        elif fingers_1 == 0 and fingers_2 == 0:
            self.drone.emergency()
            self.is_flying = False
            action = "EMERGENCY"
        elif (fingers_1 == 0 and fingers_2 == 5) or (fingers_1 == 5 and fingers_2 == 0):
            if self.is_flying:
                if not self.drone.follow_mode:
                    self.drone.enable_follow_mode()
                else:
                    self.drone.disable_follow_mode()
                action = "FOLLOW_TOGGLE"
        if action:
            self.last_two_hand_time = current_time
        return action
    
    def handle_static_gesture(self, gesture_name):
        if not self.is_flying:
            return
        if self.mode != "STATIC":
            return
        current_time = time.time()
        if current_time - self.last_gesture_time < Config.COOLDOWN_GESTURE:
            return
        if gesture_name == "LAND":
            self.drone.land()
            self.is_flying = False
        elif gesture_name == "ROCK":
            self.switch_mode()
        elif gesture_name == "FLIP":
            self.drone.flip()
        else:
            self.drone.move(gesture_name, Config.MOVE_DISTANCE)
        self.last_gesture_time = current_time
    
    def handle_dynamic_gesture(self, frame):
        if self.mode != "DYNAMIC" or not self.is_flying:
            return None
        current_time = time.time()
        if current_time - self.last_dynamic_time < Config.COOLDOWN_DYNAMIC:
            return None
        if self.dynamic_detector.detect_circle():
            print("CIRCLE detected - Orbit mode")
            self.dynamic_detector.clear()
            self.last_dynamic_time = current_time
            return "CIRCLE"
        is_swipe, direction = self.dynamic_detector.detect_swipe()
        if is_swipe:
            print(f"SWIPE {direction} - Fast movement")
            if direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
                self.drone.move(direction, Config.MOVE_DISTANCE * 2)
            self.dynamic_detector.clear()
            self.last_dynamic_time = current_time
            return f"SWIPE_{direction}"
        is_transition, transition_type = self.dynamic_detector.detect_photo_transition()
        if is_transition and transition_type == "OPEN":
            folder_name = "captured_images"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            timestamp = int(time.time())
            filename = os.path.join(folder_name, f"gesture_photo_{timestamp}.png")
            if frame is not None:
                cv2.imwrite(filename, frame)
                print(f"PHOTO CAPTURED & SAVED: {filename}")
            self.dynamic_detector.clear()
            self.last_dynamic_time = current_time
            return "PHOTO"
        if self.dynamic_detector.detect_wave():
            print("WAVE detected - Return to home")
            self.drone.position = {"x": 0, "y": 100, "z": 0}
            self.dynamic_detector.clear()
            self.last_dynamic_time = current_time
            return "WAVE"
        return None
    
    def draw_ui(self, frame, gesture, safety_action, dynamic_gesture, num_hands, confidence):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (10, 10), (400, 190), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 190), (0, 255, 0), 1)
        status = "FLYING" if self.is_flying else "GROUNDED"
        color = Config.COLOR_GREEN if self.is_flying else (100, 100, 100)
        cv2.putText(frame, f"STATUS: {status}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        mode_color = Config.COLOR_ORANGE if self.mode == "DYNAMIC" else Config.COLOR_GREEN
        cv2.putText(frame, f"MODE: {self.mode}", (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        if dynamic_gesture:
            cv2.putText(frame, f"CMD: {dynamic_gesture}", (20, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, Config.COLOR_ORANGE, 2)
        elif safety_action:
            cv2.putText(frame, f"CMD: {safety_action}", (20, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, Config.COLOR_RED, 2)
        else:
            cv2.putText(frame, f"CMD: {gesture}", (20, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, Config.COLOR_YELLOW, 2)
        cv2.putText(frame, f"CONF: {confidence*100:.1f}%", (20, 145),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, Config.COLOR_WHITE, 1)
        cv2.putText(frame, f"MODEL: {self.gesture_classifier.model_type}", (20, 175),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, Config.COLOR_WHITE, 1)
        if self.mode == "DYNAMIC":
            cv2.putText(frame, "DYNAMIC ACTIVE", (w//2 - 100, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, Config.COLOR_ORANGE, 2)
        if self.drone.follow_mode:
            cv2.putText(frame, "FOLLOW MODE", (w//2 - 100, h - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, Config.COLOR_GREEN, 3)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (w - 120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(frame, "'q'=Quit | 'd'=Toggle Mode | Rock(🤘)=Switch",
                   (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return frame
    
    def run(self):
        cap = cv2.VideoCapture(Config.CAMERA_ID)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        print("\nSYSTEM READY. SHOW HANDS TO BEGIN.")
        gesture_text = "NONE"
        safety_text = None
        dynamic_text = None
        confidence = 0.0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            num_hands = 0
            safety_text = None
            dynamic_text = None
            if results.multi_hand_landmarks:
                hands_data = results.multi_hand_landmarks
                num_hands = len(hands_data)
                for hand in hands_data:
                    self.mp_draw.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)
                if num_hands >= 2:
                    safety_text = self.handle_two_hand_gestures(hands_data)
                elif num_hands == 1:
                    hand = hands_data[0]
                    wrist = hand.landmark[0]
                    if self.gesture_classifier.detect_rock_gesture(hand):
                        if time.time() - self.last_mode_switch_time >= Config.COOLDOWN_MODE_SWITCH:
                            self.switch_mode()
                    is_open = self.is_hand_open(hand)
                    self.dynamic_detector.add_frame(wrist.x, wrist.y, is_open)
                    self.ar_overlay.add_trajectory_point((wrist.x * w, wrist.y * h))
                    if self.mode == "STATIC":
                        gesture_text, confidence = self.gesture_classifier.classify_gesture(hand, frame)
                        if confidence > Config.CONFIDENCE_THRESHOLD:
                            self.handle_static_gesture(gesture_text)
                    else:
                        gesture_text = "HOVER (Dynamic)"
                        confidence = 1.0
                        dynamic_text = self.handle_dynamic_gesture(frame)
            frame = self.draw_ui(frame, gesture_text, safety_text, dynamic_text, num_hands, confidence)
            frame = self.ar_overlay.draw_trajectory(frame)
            frame = self.ar_overlay.draw_3d_position(frame, self.drone.position)
            if results.multi_hand_landmarks:
                frame = self.robotic_hud.draw(frame, results.multi_hand_landmarks)
            cv2.imshow("Gesture Drone Controller - ML Enhanced", frame)
            self.frame_count += 1
            if self.frame_count % 10 == 0:
                self.fps = 10 / (time.time() - self.fps_start_time)
                self.fps_start_time = time.time()
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                self.switch_mode()
        if self.is_flying:
            self.drone.land()
        cap.release()
        cv2.destroyAllWindows()
        print("\nController shutdown complete")

if __name__ == "__main__":
    try:
        controller = GestureDroneController()
        controller.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()