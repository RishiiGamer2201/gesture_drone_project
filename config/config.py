"""
Gesture Drone Project Configuration
====================================
Central configuration file for all project settings.
"""

import os

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
TRAINING_DATA_DIR = os.path.join(DATA_DIR, 'training_data')
IMAGE_DATA_DIR = os.path.join(DATA_DIR, 'hand_images')
SEQUENCE_DATA_DIR = os.path.join(DATA_DIR, 'sequences')
ONLINE_LEARNING_DIR = os.path.join(DATA_DIR, 'online_learning')

# Model directories
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# Logs directory
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

# KNN Configuration
KNN_CONFIG = {
    'k': 3,
    'model_file': os.path.join(MODELS_DIR, 'gesture_model_knn.yml'),
    'metadata_file': os.path.join(MODELS_DIR, 'gesture_metadata_knn.pkl')
}

# ANN Configuration
ANN_CONFIG = {
    'hidden_layers': [128, 64, 32],
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'batch_size': 8,
    'epochs': 100,
    'model_file': os.path.join(MODELS_DIR, 'gesture_model_ann.h5'),
    'scaler_file': os.path.join(MODELS_DIR, 'gesture_scaler.pkl'),
    'metadata_file': os.path.join(MODELS_DIR, 'gesture_metadata_ann.pkl')
}

# CNN Configuration
CNN_CONFIG = {
    'image_size': 64,
    'conv_filters': [32, 64, 128],
    'dropout_rate': 0.25,
    'learning_rate': 0.001,
    'batch_size': 16,
    'epochs': 50,
    'model_file': os.path.join(MODELS_DIR, 'gesture_model_cnn.h5'),
    'metadata_file': os.path.join(MODELS_DIR, 'gesture_metadata_cnn.pkl')
}

# Temporal CNN Configuration (for dynamic gestures)
TEMPORAL_CNN_CONFIG = {
    'sequence_length': 15,
    'image_size': 64,
    'lstm_units': 128,
    'learning_rate': 0.0005,
    'batch_size': 8,
    'epochs': 50,
    'model_file': os.path.join(MODELS_DIR, 'gesture_model_temporal.h5'),
    'metadata_file': os.path.join(MODELS_DIR, 'gesture_metadata_temporal.pkl')
}

# ============================================================================
# GESTURE DEFINITIONS
# ============================================================================

# Static gestures (single frame)
# Current (Missing ID 9)
STATIC_GESTURES = {
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

# Dynamic gestures (motion-based)
DYNAMIC_GESTURES = {
    10: "CIRCLE",
    11: "SWIPE_LEFT",
    12: "SWIPE_RIGHT",
    13: "SWIPE_UP",
    14: "SWIPE_DOWN",
    15: "OPEN_CLOSE",  # Fist to open palm
    16: "WAVE"
}

# Combined gesture mapping
ALL_GESTURES = {**STATIC_GESTURES, **DYNAMIC_GESTURES}

# Two-hand gestures
TWO_HAND_GESTURES = {
    'TAKEOFF': {'left': 5, 'right': 5},  # Both open palms
    'EMERGENCY': {'left': 0, 'right': 0},  # Both fists
    'FOLLOW_MODE': {'left': 4, 'right': 5}  # Left fist, right open
}

# ============================================================================
# HAND TRACKING SETTINGS
# ============================================================================

HAND_TRACKING = {
    'max_num_hands': 2,
    'min_detection_confidence': 0.7,
    'min_tracking_confidence': 0.5,
    'static_image_mode': False
}

# ============================================================================
# DRONE CONTROL SETTINGS
# ============================================================================

DRONE_CONTROL = {
    'move_distance': 20,  # cm per command
    'rotation_angle': 15,  # degrees per rotation command
    'speed_slow': 10,
    'speed_medium': 20,
    'speed_fast': 40,
    'flip_cooldown': 3.0,  # seconds
    'command_delay': 0.1,  # seconds between commands
    'follow_mode_distance': 150  # cm to maintain in follow mode
}

# ============================================================================
# GESTURE RECOGNITION SETTINGS
# ============================================================================

RECOGNITION = {
    'confidence_threshold': 0.75,
    'buffer_size': 5,  # frames to smooth predictions
    'dynamic_gesture_frames': 15,  # frames for temporal gestures
    'circle_detection_threshold': 0.7,
    'swipe_velocity_threshold': 50,  # pixels per frame
    'open_close_threshold': 0.6  # change in hand openness
}

# ============================================================================
# CAMERA SETTINGS
# ============================================================================

CAMERA = {
    'width': 1280,
    'height': 720,
    'fps': 30,
    'camera_id': 0
}

# ============================================================================
# ONLINE LEARNING SETTINGS
# ============================================================================

ONLINE_LEARNING = {
    'enabled': True,
    'buffer_size': 100,  # corrections before retraining
    'retrain_threshold': 50,  # minimum corrections
    'save_interval': 20,  # save buffer every N corrections
    'learning_rate_multiplier': 0.5,  # reduce LR for fine-tuning
    'correction_file': os.path.join(ONLINE_LEARNING_DIR, 'corrections.pkl')
}

# ============================================================================
# AUGMENTED REALITY SETTINGS
# ============================================================================

AR_OVERLAY = {
    'enabled': True,
    'show_trajectory': True,
    'show_hand_skeleton': True,
    'show_confidence': True,
    'show_position': True,
    'trajectory_length': 30,  # frames
    'colors': {
        'trajectory': (0, 255, 255),
        'skeleton': (0, 255, 0),
        'bbox': (255, 128, 0),
        'text': (255, 255, 255),
        'confidence_high': (0, 255, 0),
        'confidence_medium': (0, 255, 255),
        'confidence_low': (0, 165, 255)
    }
}

# ============================================================================
# HAND POSE ESTIMATION
# ============================================================================

HAND_POSE = {
    'estimate_orientation': True,
    'estimate_3d_position': True,
    'rotation_sensitivity': 0.5,  # 0-1, how much rotation affects control
    'distance_sensitivity': 0.3,  # 0-1, how much distance affects speed
    'min_hand_distance': 30,  # cm
    'max_hand_distance': 200  # cm
}

# ============================================================================
# DATA COLLECTION SETTINGS
# ============================================================================

DATA_COLLECTION = {
    'min_samples_per_gesture': 50,
    'recommended_samples': 100,
    'image_format': 'png',
    'sequence_format': 'npy',
    'auto_save_interval': 10  # samples
}

# ============================================================================
# LOGGING SETTINGS
# ============================================================================

LOGGING = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': os.path.join(LOGS_DIR, 'gesture_drone.log'),
    'console_output': True
}

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

VISUALIZATION = {
    'window_name': 'Gesture Drone Controller',
    'show_fps': True,
    'show_hand_preview': True,
    'hand_preview_size': 150,
    'ui_font': 'FONT_HERSHEY_SIMPLEX',
    'ui_font_scale': 0.6,
    'ui_thickness': 2
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ensure_directories():
    """Create all necessary directories if they don't exist"""
    dirs = [
        DATA_DIR,
        TRAINING_DATA_DIR,
        IMAGE_DATA_DIR,
        SEQUENCE_DATA_DIR,
        ONLINE_LEARNING_DIR,
        MODELS_DIR,
        LOGS_DIR
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
    
    # Create subdirectories for image data
    for gesture_id, gesture_name in ALL_GESTURES.items():
        gesture_dir = os.path.join(IMAGE_DATA_DIR, f"{gesture_id}_{gesture_name}")
        os.makedirs(gesture_dir, exist_ok=True)

def get_gesture_name(gesture_id):
    """Get gesture name from ID"""
    return ALL_GESTURES.get(gesture_id, "UNKNOWN")

def get_gesture_id(gesture_name):
    """Get gesture ID from name"""
    for gid, gname in ALL_GESTURES.items():
        if gname == gesture_name:
            return gid
    return None

def is_static_gesture(gesture_id):
    """Check if gesture is static"""
    return gesture_id in STATIC_GESTURES

def is_dynamic_gesture(gesture_id):
    """Check if gesture is dynamic"""
    return gesture_id in DYNAMIC_GESTURES

# Initialize directories on import
ensure_directories()
