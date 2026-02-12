# 🚁 Gesture-Controlled Drone System

A real-time hand gesture recognition system for controlling drones using computer vision and machine learning. Control your drone with simple hand gestures - no controller needed!

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 Features

### Core Capabilities
- ✅ **10 Static Gestures** - UP, DOWN, LEFT, RIGHT, FORWARD, BACKWARD, HOVER, LAND, FLIP, ROCK
- ✅ **Dynamic Gestures** - Circle, Swipe (4 directions), Photo Capture, Wave
- ✅ **Two-Hand Controls** - Takeoff, Emergency Stop, Follow Mode
- ✅ **Mode Switching** - Toggle between Static and Dynamic modes
- ✅ **ML Model Support** - KNN, CNN, and ANN models for accurate recognition
- ✅ **Real-time Hand Tracking** - 60+ FPS performance with MediaPipe
- ✅ **AR Overlay** - Hand trajectory visualization and drone telemetry
- ✅ **Robotic HUD** - Advanced hand skeleton display

### Advanced Features
- 🎮 **Dual Mode System** - Prevent gesture confusion with exclusive modes
- 🤖 **ML Model Integration** - Automatic model detection and loading
- 📸 **Photo Capture** - Take photos using hand gestures
- 👤 **Follow Mode** - Drone follows your hand movements
- 📊 **Live Telemetry** - Real-time position and status display
- 🎨 **Custom HUD** - Robotic-style hand visualization

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8 or higher
Webcam or external camera
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/RishiiGamer2201/gesture_drone_project.git
cd gesture_drone_project
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
python test.py
```

That's it! The system will automatically detect and load any trained models from the `models/` folder.

## 🎮 Controls

### Two-Hand Gestures (Always Active)
| Gesture | Action | Description |
|---------|--------|-------------|
| ✋✋ Two open palms | **TAKEOFF** | Launch the drone |
| ✊✊ Two fists | **EMERGENCY** | Immediate stop and land |
| ✊✋ Fist + Open | **FOLLOW MODE** | Toggle follow mode |

### Static Mode (Default)
| Gesture | Action | How to Perform |
|---------|--------|----------------|
| ☝️ One finger up | **UP** | Point index finger upward |
| 👇 One finger down | **DOWN** | Point index finger downward |
| 👈 Point left | **LEFT** | Point index finger left |
| 👉 Point right | **RIGHT** | Point index finger right |
| 👍 Thumbs up | **FORWARD** | Thumb pointing up |
| 👎 Thumbs down | **BACKWARD** | Thumb pointing down |
| ✊ Closed fist | **HOVER** | Make a fist |
| ✋ Open palm | **LAND** | Show all 5 fingers |
| ✌️ Peace sign | **FLIP** | Index + middle finger up |
| 🤘 Rock sign | **MODE SWITCH** | Index + pinky up |

### Dynamic Mode (After switching)
| Gesture | Action | How to Perform |
|---------|--------|----------------|
| 🔄 Circle | **ORBIT** | Draw a circle with hand |
| 👆 Swipe Up | **FAST UP** | Quick upward motion |
| 👇 Swipe Down | **FAST DOWN** | Quick downward motion |
| 👈 Swipe Left | **FAST LEFT** | Quick left motion |
| 👉 Swipe Right | **FAST RIGHT** | Quick right motion |
| 📸 Fist → Open | **PHOTO** | Close then open hand |
| 👋 Wave | **RETURN HOME** | Wave hand side to side |

### Keyboard Shortcuts
| Key | Action |
|-----|--------|
| `d` | Toggle between Static and Dynamic modes |
| `q` | Quit application |

## 📁 Project Structure

```
gesture_drone_project/
├── test.py                          # Main application (START HERE)
├── test1.py
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── PROJECT_STRUCTURE.txt           # Detailed structure
│
├── models/                          # Trained ML models
│   ├── gesture_model_knn.yml       # K-Nearest Neighbors model
│   ├── gesture_model_cnn.h5        # Convolutional Neural Network
│   └── gesture_model_ann.pkl       # Artificial Neural Network
│
├── data/                            # Training and collected data
│   ├── hand_images/                # CNN training images (by gesture)
│   ├── training_data/              # KNN/ANN training data
│   └── sequences/                  # Dynamic gesture sequences
│
├── captured_images/                 # Photos taken with gestures
│
├── src/
│   ├── controllers/                # Drone controllers
│   │   ├── final_controller.py    # Production controller
│   │   ├── simple_controller.py   # Basic controller
│   │   └── advanced_controller.py # Full-featured controller
│   │
│   ├── data_collection/            # Data collection tools
│   │   ├── collect_images.py      # Collect hand images for CNN
│   │   └── collect_static.py      # Collect landmark data for KNN/ANN
│   │
│   ├── training/                   # Model training scripts
│   │   ├── train_knn.py           # Train KNN model
│   │   ├── train_ann.py           # Train ANN model
│   │   └── train_cnn.py           # Train CNN model
│   │
│   └── utils/                      # Utility modules
│       ├── gesture_detection.py   # Gesture detection logic
│       ├── ar_overlay.py          # AR visualization
│       └── online_learning.py     # Adaptive learning
│
└── config/
    └── config.py                   # Configuration settings

```

## 🛠️ How It Works

### 1. Hand Detection
Uses Google's **MediaPipe** to detect hands and extract 21 landmark points per hand in real-time.

### 2. Gesture Classification
Three ML models available (automatically selected):
- **KNN** - Fast, lightweight, good for real-time (Priority 1)
- **CNN** - Most accurate, best for static gestures (Priority 2)
- **ANN** - Balanced speed and accuracy (Priority 3)
- **Fallback** - Simple finger counting if no models found

### 3. Dynamic Gesture Detection
Analyzes hand motion over time (15-20 frames) to detect:
- Circular motions
- Directional swipes
- Hand opening/closing transitions
- Waving patterns

### 4. Mode System
**STATIC MODE** (Default):
- Only static gestures work
- Full drone control
- Can land normally

**DYNAMIC MODE** (Activated by ROCK gesture or 'd' key):
- Only dynamic gestures work
- Drone auto-hovers
- Must switch back to Static to land

This prevents confusion between static and dynamic gestures!

### 5. Drone Control
Currently uses a **MockDrone** for safe testing. Replace with real drone API (DJI Tello) for actual flight:
```python
# Replace MockDrone with:
from djitellopy import Tello
drone = Tello()
drone.connect()
```

## 📊 Training Your Own Models

### Step 1: Collect Data

**For CNN (Image-based):**
```bash
python src/data_collection/collect_images.py
```
- Press 0-9 to capture gestures
- Collect 50-100 images per gesture
- Images saved to `data/hand_images/`

**For KNN/ANN (Landmark-based):**
```bash
python src/data_collection/collect_static.py
```
- Press 0-9 to capture gestures
- Press 's' to save
- Data saved to `data/training_data/`

### Step 2: Train Models

**Train KNN:**
```bash
python src/training/train_knn.py
```

**Train CNN:**
```bash
python src/training/train_cnn.py
```

**Train ANN:**
```bash
python src/training/train_ann.py
```

Models are automatically saved to `models/` and will be loaded on next run.

## 🎯 Performance

- **FPS**: 25-30 on average hardware
- **Latency**: 30-50ms gesture recognition
- **Accuracy**: 
  - KNN: 85-90%
  - CNN: 92-95%
  - ANN: 88-92%
  - Fallback: 70-80%

## 🔧 Configuration

Edit `config/config.py` to customize:
```python
CAMERA_ID = 0                  # Camera index
CAMERA_WIDTH = 1280            # Resolution
CAMERA_HEIGHT = 720
MOVE_DISTANCE = 20             # Movement distance (cm)
CONFIDENCE_THRESHOLD = 0.75    # Minimum confidence
COOLDOWN_GESTURE = 0.5         # Gesture cooldown (seconds)
```

## 🐛 Troubleshooting

### Camera Not Opening
```python
# Try different camera IDs in config/config.py
CAMERA_ID = 0  # Try 0, 1, 2, etc.
```

### Gestures Not Detected
1. Ensure good lighting
2. Hand fully visible to camera
3. Check confidence threshold
4. Train models with your own hand data

### Mode Not Switching
1. Verify ROCK gesture is correct:
   - Index finger: UP
   - Pinky finger: UP
   - Middle finger: DOWN
   - Ring finger: DOWN
2. Try 'd' key instead
3. Wait for cooldown (1.5 seconds)

### Low Accuracy
1. Collect more training data
2. Train with your specific hand
3. Adjust lighting conditions
4. Increase confidence threshold

## 🚀 Advanced Usage

### Using with Real Drone (DJI Tello)

1. Install DJI Tello SDK:
```bash
pip install djitellopy
```

2. Modify `test.py`:
```python
# Replace line ~40:
from djitellopy import Tello

# Replace MockDrone() with:
self.drone = Tello()
self.drone.connect()
```

3. **Safety First**:
- Test in open area
- Have manual override ready
- Start with low altitude
- Follow local regulations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Rishii Kumar Singh**
- GitHub: [@RishiiGamer2201](https://github.com/RishiiGamer2201)
- Project: [gesture_drone_project](https://github.com/RishiiGamer2201/gesture_drone_project)

## 📽️ Video Demonstration
[Click here to watch the Project Demo Video](https://drive.google.com/file/d/1U7Jknz6b5JID3wG8v70Un_lgvapMVmsT/view?usp=sharing)

## 🙏 Acknowledgments

- **MediaPipe** by Google for hand tracking
- **OpenCV** for computer vision
- **TensorFlow** for deep learning
- **DJI Tello** for drone platform

## 📞 Support

For issues, questions, or suggestions:
1. Open an issue on GitHub
2. Review existing issues for solutions

## 🎓 Learn More

This project demonstrates:
- Computer Vision with OpenCV
- Hand tracking with MediaPipe
- Machine Learning (KNN, CNN, ANN)
- Real-time gesture recognition
- State management
- AR visualization
- Drone control systems

Perfect for learning CV, ML, and robotics!

---

**⚠️ Safety Warning**: Always test with MockDrone first. Follow all safety guidelines and local regulations when using real drones.

**🎯 Ready to fly?** Run `python test.py` and start controlling with your hands!
