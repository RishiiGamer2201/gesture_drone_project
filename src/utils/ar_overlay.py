"""
Augmented Reality Overlay Utilities
====================================
Visualization tools for AR overlays on video feed.
"""

import cv2
import numpy as np
from collections import deque
import math

class AROverlay:
    """Augmented Reality overlay for gesture visualization"""
    
    def __init__(self, config):
        self.config = config
        self.trajectory_history = deque(maxlen=config.get('trajectory_length', 30))
        self.drone_position_history = deque(maxlen=50)
        self.colors = config.get('colors', {})
        
    def add_trajectory_point(self, point):
        """Add point to trajectory"""
        self.trajectory_history.append(point)
    
    def add_drone_position(self, position):
        """Add drone position to history"""
        self.drone_position_history.append(position)
    
    def draw_trajectory(self, frame):
        """Draw hand movement trajectory"""
        if len(self.trajectory_history) < 2:
            return frame
        
        points = list(self.trajectory_history)
        
        for i in range(1, len(points)):
            # Fade older points
            alpha = i / len(points)
            color = tuple(int(c * alpha) for c in self.colors.get('trajectory', (0, 255, 255)))
            
            pt1 = (int(points[i-1][0]), int(points[i-1][1]))
            pt2 = (int(points[i][0]), int(points[i][1]))
            
            thickness = max(1, int(3 * alpha))
            cv2.line(frame, pt1, pt2, color, thickness)
        
        return frame
    
    def draw_hand_skeleton(self, frame, hand_landmarks, mp_hands, mp_draw):
        """Draw enhanced hand skeleton with depth"""
        if not hand_landmarks:
            return frame
        
        # Draw connections with varying thickness based on depth
        for connection in mp_hands.HAND_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            
            start = hand_landmarks.landmark[start_idx]
            end = hand_landmarks.landmark[end_idx]
            
            # Calculate depth-based thickness
            avg_z = (start.z + end.z) / 2 if hasattr(start, 'z') else 0
            thickness = max(1, int(3 - avg_z * 5))
            
            h, w = frame.shape[:2]
            pt1 = (int(start.x * w), int(start.y * h))
            pt2 = (int(end.x * w), int(end.y * h))
            
            cv2.line(frame, pt1, pt2, self.colors.get('skeleton', (0, 255, 0)), thickness)
        
        # Draw landmarks
        for idx, landmark in enumerate(hand_landmarks.landmark):
            h, w = frame.shape[:2]
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            
            # Highlight fingertips
            if idx in [4, 8, 12, 16, 20]:
                cv2.circle(frame, (cx, cy), 8, (255, 0, 255), -1)
                cv2.circle(frame, (cx, cy), 10, (255, 255, 255), 2)
            else:
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        
        return frame
    
    def draw_confidence_meter(self, frame, confidence, position=(20, 100)):
        """Draw confidence meter bar"""
        x, y = position
        bar_width = 200
        bar_height = 20
        
        # Background
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (50, 50, 50), -1)
        
        # Confidence bar
        conf_width = int(bar_width * confidence)
        
        if confidence > 0.9:
            color = self.colors.get('confidence_high', (0, 255, 0))
        elif confidence > 0.75:
            color = self.colors.get('confidence_medium', (0, 255, 255))
        else:
            color = self.colors.get('confidence_low', (0, 165, 255))
        
        cv2.rectangle(frame, (x, y), (x + conf_width, y + bar_height), color, -1)
        
        # Border
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (255, 255, 255), 2)
        
        # Text
        text = f"Confidence: {confidence*100:.1f}%"
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def draw_drone_position_3d(self, frame, position, size=(200, 200)):
        """Draw 3D visualization of drone position"""
        h, w = frame.shape[:2]
        
        # Position in bottom-right corner
        x_offset = w - size[0] - 20
        y_offset = h - size[1] - 20
        
        # Create semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_offset, y_offset), 
                     (x_offset + size[0], y_offset + size[1]), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw 3D grid
        center_x = x_offset + size[0] // 2
        center_y = y_offset + size[1] // 2
        
        # Grid lines
        for i in range(-2, 3):
            # Horizontal lines
            y = center_y + i * 30
            cv2.line(frame, (x_offset + 20, y), (x_offset + size[0] - 20, y), (100, 100, 100), 1)
            # Vertical lines
            x = center_x + i * 30
            cv2.line(frame, (x, y_offset + 20), (x, y_offset + size[1] - 20), (100, 100, 100), 1)
        
        # Draw drone position
        drone_x = int(center_x + position['x'])
        drone_y = int(center_y - position['y'] / 2)  # Y is inverted and scaled
        
        # Drone representation
        cv2.circle(frame, (drone_x, drone_y), 10, (0, 255, 0), -1)
        cv2.circle(frame, (drone_x, drone_y), 12, (255, 255, 255), 2)
        
        # Altitude indicator
        altitude_bar_height = int(position['y'] / 2)
        if altitude_bar_height > 0:
            cv2.rectangle(frame, 
                         (x_offset + 10, drone_y), 
                         (x_offset + 15, center_y),
                         (0, 255, 255), -1)
        
        # Labels
        cv2.putText(frame, "3D Position", (x_offset + 10, y_offset + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"X:{position['x']:.0f}", (x_offset + 10, y_offset + size[1] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Y:{position['y']:.0f}", (x_offset + 10, y_offset + size[1] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Z:{position['z']:.0f}", (x_offset + 10, y_offset + size[1] - 0),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def draw_gesture_info(self, frame, gesture_name, gesture_type="static"):
        """Draw gesture information panel"""
        h, w = frame.shape[:2]
        
        # Panel background
        panel_height = 80
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h - panel_height - 10), (400, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Gesture name
        cv2.putText(frame, f"Gesture: {gesture_name}", (20, h - panel_height + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Gesture type
        type_color = (255, 128, 0) if gesture_type == "dynamic" else (0, 255, 0)
        cv2.putText(frame, f"Type: {gesture_type.upper()}", (20, h - panel_height + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, type_color, 1)
        
        return frame
    
    def draw_follow_mode_indicator(self, frame, target_position, current_distance):
        """Draw follow mode visualization"""
        h, w = frame.shape[:2]
        
        # Draw target reticle at hand position
        if target_position:
            tx, ty = int(target_position[0] * w), int(target_position[1] * h)
            
            # Crosshair
            size = 30
            cv2.line(frame, (tx - size, ty), (tx + size, ty), (0, 255, 0), 2)
            cv2.line(frame, (tx, ty - size), (tx, ty + size), (0, 255, 0), 2)
            cv2.circle(frame, (tx, ty), size, (0, 255, 0), 2)
            
            # Distance indicator
            cv2.putText(frame, f"Distance: {current_distance:.1f}cm", 
                       (tx + 40, ty),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Follow mode text
        cv2.putText(frame, "FOLLOW MODE ACTIVE", (w//2 - 150, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
        return frame
    
    def draw_dynamic_gesture_trail(self, frame, gesture_type):
        """Draw visual trail for dynamic gestures"""
        if gesture_type == "CIRCLE":
            self._draw_circle_indicator(frame)
        elif "SWIPE" in gesture_type:
            self._draw_swipe_indicator(frame, gesture_type)
        elif gesture_type == "WAVE":
            self._draw_wave_indicator(frame)
        
        return frame
    
    def _draw_circle_indicator(self, frame):
        """Draw circular motion indicator"""
        if len(self.trajectory_history) < 3:
            return
        
        points = list(self.trajectory_history)
        
        # Draw circle guide
        if len(points) >= 5:
            # Fit circle to points
            points_array = np.array(points, dtype=np.float32)
            center, radius = cv2.minEnclosingCircle(points_array)
            
            cv2.circle(frame, (int(center[0]), int(center[1])), 
                      int(radius), (255, 0, 255), 2, cv2.LINE_AA)
    
    def _draw_swipe_indicator(self, frame, gesture_type):
        """Draw swipe direction indicator"""
        if len(self.trajectory_history) < 2:
            return
        
        start = self.trajectory_history[0]
        end = self.trajectory_history[-1]
        
        # Draw arrow
        cv2.arrowedLine(frame, 
                       (int(start[0]), int(start[1])),
                       (int(end[0]), int(end[1])),
                       (255, 255, 0), 3, tipLength=0.3)
    
    def _draw_wave_indicator(self, frame):
        """Draw wave motion indicator"""
        if len(self.trajectory_history) < 3:
            return
        
        points = list(self.trajectory_history)
        
        # Draw wavy line
        for i in range(1, len(points)):
            pt1 = (int(points[i-1][0]), int(points[i-1][1]))
            pt2 = (int(points[i][0]), int(points[i][1]))
            
            cv2.line(frame, pt1, pt2, (0, 255, 255), 3)
    
    def draw_fps(self, frame, fps):
        """Draw FPS counter"""
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame
    
    def clear_trajectory(self):
        """Clear trajectory history"""
        self.trajectory_history.clear()
