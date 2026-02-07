"""
Gesture Detection Utilities
============================
Advanced gesture detection including dynamic gestures,
hand pose estimation, and motion analysis.
"""

import numpy as np
import cv2
from collections import deque
import math

class DynamicGestureDetector:
    """Detect dynamic gestures from hand motion sequences"""
    
    def __init__(self, sequence_length=15):
        self.sequence_length = sequence_length
        self.position_history = deque(maxlen=sequence_length)
        self.hand_state_history = deque(maxlen=sequence_length)
        
    def add_frame(self, hand_landmarks, hand_state=None):
        """
        Add a frame to the sequence
        hand_state: 'open' or 'closed'
        """
        # Get center of hand (wrist position)
        if hand_landmarks:
            wrist = hand_landmarks.landmark[0]
            self.position_history.append((wrist.x, wrist.y))
            self.hand_state_history.append(hand_state)
        
    def detect_circle(self, threshold=0.7):
        """
        Detect circular motion
        Returns: (is_circle, confidence, direction)
        """
        if len(self.position_history) < self.sequence_length:
            return False, 0.0, None
        
        positions = np.array(list(self.position_history))
        
        # Calculate center
        center = np.mean(positions, axis=0)
        
        # Calculate radius variations
        radii = np.linalg.norm(positions - center, axis=1)
        radius_std = np.std(radii)
        radius_mean = np.mean(radii)
        
        # Check if radius is relatively constant (circular motion)
        if radius_mean < 0.01 or radius_std / radius_mean > 0.3:
            return False, 0.0, None
        
        # Calculate angles
        angles = []
        for pos in positions:
            angle = math.atan2(pos[1] - center[1], pos[0] - center[0])
            angles.append(angle)
        
        # Calculate cumulative angle change
        angle_changes = np.diff(angles)
        
        # Handle angle wrapping
        angle_changes = np.array([
            ac if abs(ac) < math.pi else ac - 2*math.pi*np.sign(ac)
            for ac in angle_changes
        ])
        
        total_rotation = np.sum(angle_changes)
        
        # Check if total rotation is close to 2π (full circle)
        expected_rotation = 2 * math.pi
        rotation_error = abs(abs(total_rotation) - expected_rotation)
        
        confidence = max(0, 1 - (rotation_error / math.pi))
        
        if confidence > threshold:
            direction = 'clockwise' if total_rotation > 0 else 'counterclockwise'
            return True, confidence, direction
        
        return False, confidence, None
    
    def detect_swipe(self, velocity_threshold=50):
        """
        Detect swipe gesture
        Returns: (is_swipe, direction, velocity)
        """
        if len(self.position_history) < 5:
            return False, None, 0
        
        positions = np.array(list(self.position_history))
        
        # Calculate displacement
        start = positions[0]
        end = positions[-1]
        displacement = end - start
        
        # Calculate velocity (pixels per frame)
        velocity = np.linalg.norm(displacement)
        
        if velocity < velocity_threshold / 1000:  # Normalized
            return False, None, velocity
        
        # Determine direction
        angle = math.atan2(displacement[1], displacement[0])
        angle_deg = math.degrees(angle)
        
        # Classify direction
        if -45 <= angle_deg < 45:
            direction = 'RIGHT'
        elif 45 <= angle_deg < 135:
            direction = 'DOWN'
        elif -135 <= angle_deg < -45:
            direction = 'UP'
        else:
            direction = 'LEFT'
        
        return True, direction, velocity
    
    def detect_open_close(self, threshold=0.6):
        """
        Detect fist-to-open or open-to-fist transition
        Returns: (is_transition, transition_type, confidence)
        """
        if len(self.hand_state_history) < 10:
            return False, None, 0.0
        
        states = list(self.hand_state_history)
        
        # Count state changes
        first_half = states[:len(states)//2]
        second_half = states[len(states)//2:]
        
        first_open_count = first_half.count('open')
        second_open_count = second_half.count('open')
        
        first_ratio = first_open_count / len(first_half) if first_half else 0
        second_ratio = second_open_count / len(second_half) if second_half else 0
        
        # Detect fist -> open
        if first_ratio < (1 - threshold) and second_ratio > threshold:
            confidence = second_ratio - first_ratio
            return True, 'OPEN', confidence
        
        # Detect open -> fist
        elif first_ratio > threshold and second_ratio < (1 - threshold):
            confidence = first_ratio - second_ratio
            return True, 'CLOSE', confidence
        
        return False, None, 0.0
    
    def detect_wave(self, min_oscillations=2):
        """
        Detect waving motion (side-to-side oscillation)
        Returns: (is_wave, frequency, amplitude)
        """
        if len(self.position_history) < self.sequence_length:
            return False, 0, 0
        
        positions = np.array(list(self.position_history))
        x_positions = positions[:, 0]
        
        # Find peaks and troughs
        peaks = []
        troughs = []
        
        for i in range(1, len(x_positions) - 1):
            if x_positions[i] > x_positions[i-1] and x_positions[i] > x_positions[i+1]:
                peaks.append(i)
            elif x_positions[i] < x_positions[i-1] and x_positions[i] < x_positions[i+1]:
                troughs.append(i)
        
        oscillations = len(peaks) + len(troughs)
        
        if oscillations < min_oscillations * 2:
            return False, 0, 0
        
        # Calculate amplitude
        if peaks and troughs:
            amplitude = np.mean([x_positions[p] for p in peaks]) - \
                       np.mean([x_positions[t] for t in troughs])
        else:
            amplitude = 0
        
        frequency = oscillations / self.sequence_length
        
        return oscillations >= min_oscillations * 2, frequency, amplitude
    
    def clear(self):
        """Clear all history"""
        self.position_history.clear()
        self.hand_state_history.clear()


class HandPoseEstimator:
    """Estimate 3D hand pose and orientation"""
    
    def __init__(self):
        self.palm_size_baseline = None
        
    def estimate_hand_openness(self, hand_landmarks):
        """
        Estimate how open the hand is (0 = closed fist, 1 = fully open)
        """
        # Get fingertip and base positions
        fingertips = [8, 12, 16, 20]  # Index, middle, ring, pinky
        finger_bases = [5, 9, 13, 17]
        
        # Calculate average distance from fingertips to palm center
        palm_center = hand_landmarks.landmark[0]  # Wrist
        
        distances = []
        for tip in fingertips:
            tip_pos = hand_landmarks.landmark[tip]
            dist = math.sqrt(
                (tip_pos.x - palm_center.x)**2 +
                (tip_pos.y - palm_center.y)**2
            )
            distances.append(dist)
        
        avg_distance = np.mean(distances)
        
        # Normalize (assuming max distance is ~0.3 for open hand)
        openness = min(1.0, avg_distance / 0.25)
        
        return openness
    
    def get_hand_state(self, hand_landmarks):
        """
        Determine if hand is 'open' or 'closed'
        """
        openness = self.estimate_hand_openness(hand_landmarks)
        return 'open' if openness > 0.5 else 'closed'
    
    def estimate_hand_rotation(self, hand_landmarks):
        """
        Estimate hand rotation angles (roll, pitch, yaw)
        Returns: dict with 'roll', 'pitch', 'yaw' in degrees
        """
        # Use specific landmarks to estimate orientation
        wrist = hand_landmarks.landmark[0]
        middle_mcp = hand_landmarks.landmark[9]  # Middle finger base
        index_mcp = hand_landmarks.landmark[5]   # Index finger base
        pinky_mcp = hand_landmarks.landmark[17]  # Pinky base
        
        # Calculate vectors
        palm_vector = np.array([
            middle_mcp.x - wrist.x,
            middle_mcp.y - wrist.y,
            middle_mcp.z - wrist.z if hasattr(middle_mcp, 'z') else 0
        ])
        
        finger_vector = np.array([
            index_mcp.x - pinky_mcp.x,
            index_mcp.y - pinky_mcp.y,
            index_mcp.z - pinky_mcp.z if hasattr(index_mcp, 'z') else 0
        ])
        
        # Calculate roll (rotation around palm axis)
        roll = math.degrees(math.atan2(finger_vector[1], finger_vector[0]))
        
        # Calculate pitch (hand tilted forward/back)
        pitch = math.degrees(math.atan2(palm_vector[1], palm_vector[0]))
        
        # Calculate yaw (hand turned left/right)
        yaw = math.degrees(math.atan2(palm_vector[0], palm_vector[2])) if len(palm_vector) > 2 else 0
        
        return {
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw
        }
    
    def estimate_hand_distance(self, hand_landmarks, frame_width):
        """
        Estimate distance of hand from camera based on palm size
        Returns: distance in arbitrary units (0-1, closer = higher)
        """
        # Calculate palm size
        wrist = hand_landmarks.landmark[0]
        middle_mcp = hand_landmarks.landmark[9]
        
        palm_size = math.sqrt(
            (middle_mcp.x - wrist.x)**2 +
            (middle_mcp.y - wrist.y)**2
        ) * frame_width
        
        # Set baseline on first call
        if self.palm_size_baseline is None:
            self.palm_size_baseline = palm_size
        
        # Normalize (larger palm = closer)
        distance_factor = palm_size / self.palm_size_baseline
        
        return distance_factor
    
    def get_hand_3d_position(self, hand_landmarks, frame_width, frame_height):
        """
        Get estimated 3D position of hand
        Returns: dict with 'x', 'y', 'z' (normalized 0-1)
        """
        wrist = hand_landmarks.landmark[0]
        
        return {
            'x': wrist.x,
            'y': wrist.y,
            'z': self.estimate_hand_distance(hand_landmarks, frame_width)
        }


class GestureClassifier:
    """Classify gestures combining static and dynamic detection"""
    
    def __init__(self):
        self.dynamic_detector = DynamicGestureDetector()
        self.pose_estimator = HandPoseEstimator()
        
    def classify_static(self, hand_landmarks):
        """
        Classify static gesture based on finger positions
        Returns: gesture_name or None
        """
        # Count extended fingers
        fingers_up = self.count_fingers(hand_landmarks)
        
        # Simple rule-based classification
        if fingers_up == 1:
            # Check which finger is up
            index_tip = hand_landmarks.landmark[8]
            index_pip = hand_landmarks.landmark[6]
            
            if index_tip.y < index_pip.y:
                return "UP"
        
        elif fingers_up == 5:
            return "LAND"
        
        elif fingers_up == 0:
            return "HOVER"
        
        return None
    
    def count_fingers(self, hand_landmarks):
        """Count number of extended fingers"""
        finger_tips = [8, 12, 16, 20]
        thumb_tip = 4
        
        fingers_up = 0
        
        # Check thumb
        if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_tip - 1].x:
            fingers_up += 1
        
        # Check other fingers
        for tip in finger_tips:
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                fingers_up += 1
        
        return fingers_up
    
    def update_dynamic(self, hand_landmarks):
        """Update dynamic gesture detector with new frame"""
        hand_state = self.pose_estimator.get_hand_state(hand_landmarks)
        self.dynamic_detector.add_frame(hand_landmarks, hand_state)
    
    def detect_dynamic_gesture(self):
        """
        Detect any dynamic gesture
        Returns: (gesture_name, confidence) or (None, 0)
        """
        # Check for circle
        is_circle, conf, direction = self.dynamic_detector.detect_circle()
        if is_circle:
            return "CIRCLE", conf
        
        # Check for swipe
        is_swipe, direction, velocity = self.dynamic_detector.detect_swipe()
        if is_swipe:
            return f"SWIPE_{direction}", velocity / 100  # Normalize
        
        # Check for open/close
        is_transition, trans_type, conf = self.dynamic_detector.detect_open_close()
        if is_transition:
            return "OPEN_CLOSE", conf
        
        # Check for wave
        is_wave, freq, amp = self.dynamic_detector.detect_wave()
        if is_wave:
            return "WAVE", min(1.0, freq * amp * 10)
        
        return None, 0.0
