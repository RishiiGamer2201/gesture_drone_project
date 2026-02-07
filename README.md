# 🚁 Advanced Gesture-Controlled Drone System

A complete gesture recognition system for drone control with machine learning, featuring dynamic gestures, hand pose estimation, AR overlay, and online learning.

## ✨ Features

### Core Features
- **Three ML Models**: KNN, ANN, and CNN implementations
- **Static Gestures**: 10 hand gestures for basic control (UP, DOWN, LEFT, RIGHT, FORWARD, BACKWARD, HOVER, LAND, FLIP, ROCK)
- **Dynamic Gestures**: CIRCLE, SWIPE (4 directions), OPEN/CLOSE, WAVE
- **Hand Pose Estimation**: 3D position and orientation tracking
- **Two-Hand Coordination**: Follow mode - drone follows your hand
- **AR Overlay**: Real-time visualization with trajectory and 3D position
- **Online Learning**: Adaptive model improvement during runtime
- **Mock Drone**: Safe testing without hardware

### Advanced Capabilities
- **Follow Mode**: Left fist + right open palm activates drone following
- **Image Capture**: Open/close gesture triggers camera
- **Gesture Sequences**: Combine gestures for complex commands
- **Confidence Scoring**: Visual feedback for gesture recognition quality
- **Adaptive Speed**: Hand distance controls movement speed
- **Rotation Control**: Hand tilt controls drone rotation

## 📁 Project Structure

```
gesture_drone_project/
├── config/
│   └── config.py                 # Central configuration
├── src/
│   ├── data_collection/
│   │   ├── collect_static.py     # Collect static gesture data
│   │   ├── collect_dynamic.py    # Collect dynamic gesture sequences
│   │   └── collect_images.py     # Collect images for CNN
│   ├── training/
│   │   ├── train_knn.py          # Train KNN model
│   │   ├── train_ann.py          # Train ANN model
│   │   └── train_cnn.py          # Train CNN model
│   ├── controllers/
│   │   └── advanced_controller.py # Main advanced controller
│   └── utils/
│       ├── gesture_detection.py   # Dynamic gesture detection
│       ├── ar_overlay.py          # AR visualization
│       └── online_learning.py     # Adaptive learning
├── models/                        # Trained models (generated)
├── data/                          # Training data (generated)
├── logs/                          # System logs
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Collect Training Data

```bash
# For KNN/ANN (landmarks)
python src/data_collection/collect_static.py

# For CNN (images)
python src/data_collection/collect_images.py

# For dynamic gestures
python src/data_collection/collect_dynamic.py
```

**Recommendation**: Collect 50-100 samples per gesture

### 3. Train Models

```bash
# Train KNN (fastest)
python src/training/train_knn.py

# Train ANN (best accuracy)
python src/training/train_ann.py

# Train CNN (most robust)
python src/training/train_cnn.py
```

### 4. Run Controller

```bash
# With CNN (recommended)
python src/controllers/advanced_controller.py --model cnn

# With ANN
python src/controllers/advanced_controller.py --model ann

# With KNN
python src/controllers/advanced_controller.py --model knn
```

## 🎮 Gesture Controls

### Static Gestures (Single Frame)
| Gesture | Command | Description |
|---------|---------|-------------|
| 👆 Index Up | UP | Move drone up |
| 👇 Index Down | DOWN | Move drone down |
| 👈 Index Left | LEFT | Move drone left |
| 👉 Index Right | RIGHT | Move drone right |
| 👍 Thumbs Up | FORWARD | Move forward |
| 👎 Thumbs Down | BACKWARD | Move backward |
| ✌️ Peace Sign | FLIP | Backflip |
| ✊ Fist | HOVER | Hover in place |
| ✋ Open Palm | LAND | Land drone |
| 🤘 Rock | CHANGE MODE | Change mode from static to dyanamic and vice-verse |

### Dynamic Gestures (Motion-Based)
| Gesture | Command | Description |
|---------|---------|-------------|
| ⭕ Circle Motion | CIRCLE | Orbit mode |
| ← Swipe Left | SWIPE_LEFT | Fast left |
| → Swipe Right | SWIPE_RIGHT | Fast right |
| ↑ Swipe Up | SWIPE_UP | Fast up |
| ↓ Swipe Down | SWIPE_DOWN | Fast down |
| ✊→✋ Fist to Open | OPEN_CLOSE | Capture image |
| 👋 Wave | WAVE | Return to home |

### Two-Hand Gestures
| Gesture | Command | Description |
|---------|---------|-------------|
| ✋✋ Two Open Palms | TAKEOFF | Takeoff |
| ✊✊ Two Fists | EMERGENCY | Emergency stop |
| ✊✋ Left Fist + Right Open | FOLLOW MODE | Drone follows hand |

## 🎯 Model Comparison

| Feature | KNN | ANN | CNN |
|---------|-----|-----|-----|
| **Training Time** | <1s | 30-60s | 60-120s |
| **Prediction Speed** | 1ms | 5ms | 8ms |
| **Accuracy** | 85-95% | 90-98% | 92-99% |
| **Data Needed** | 20+ | 30+ | 50+ |
| **Best For** | Learning | Production | Research |

## 🔧 Configuration

Edit `config/config.py` to customize:

- **Gesture sensitivity**
- **Movement distances**
- **Confidence thresholds**
- **AR overlay settings**
- **Online learning parameters**

## 📊 Advanced Features

### Hand Pose Estimation
- Estimates 3D hand position
- Tracks hand orientation (roll, pitch, yaw)
- Controls drone rotation with hand tilt
- Adjusts speed based on hand distance

### Follow Mode
- Activate with left fist + right open palm
- Drone maintains distance from target hand
- Auto-adjusts position as you move
- Deactivates automatically on other gestures

### Online Learning
- Corrects mispredictions in real-time
- Press 'c' to enter correction mode
- Model auto-updates after 50 corrections
- Personalizes to your gesture style

### AR Overlay
- Hand trajectory visualization
- 3D drone position display
- Confidence meter
- Gesture type indicators
- FPS counter

## 🐛 Troubleshooting

**Camera not detected:**
```bash
# Test camera
python -c "import cv2; print('OK' if cv2.VideoCapture(0).read()[0] else 'FAIL')"
```

**Low accuracy:**
- Collect more diverse training data
- Ensure good lighting
- Make distinct gestures
- Use CNN model for best results

**Gestures not recognized:**
- Check confidence threshold in config
- Ensure entire hand is visible
- Improve lighting conditions
- Retrain with more samples

## 📈 Performance Tips

1. **Lighting**: Bright, even lighting works best
2. **Background**: Simple, uncluttered background
3. **Distance**: Keep hand 30-100cm from camera
4. **Gestures**: Make clear, exaggerated gestures
5. **Position**: Keep hand centered in frame

## 🔬 For Real Drone (DJI Tello)

1. Install djitellopy:
```bash
pip install djitellopy
```

2. Replace `AdvancedMockDrone` with:
```python
from djitellopy import Tello
self.drone = Tello()
```

3. Connect to Tello WiFi network

4. Test in safe, open area!

## 📝 License

This project is for educational purposes.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- More dynamic gestures
- Voice command integration
- Multi-drone control
- Mobile deployment
- Better visualization

## ⚠️ Safety

**IMPORTANT** when using real drone:
- Always fly in open, safe areas
- Keep drone in sight
- Have manual override ready
- Follow local regulations
- Never fly near people/animals
- Test extensively in mock mode first

## Video Demonstration

[Click here to watch the Project Demo Video](https://drive.google.com/file/d/1U7Jknz6b5JID3wG8v70Un_lgvapMVmsT/view?usp=sharing)

---

**Ready to fly! 🚁✨**

*Built with ❤️ using Python, OpenCV, MediaPipe, and TensorFlow*
