# 🚀 Setup Guide - Gesture Drone Project

## Step-by-Step Installation & Usage

### Step 1: System Requirements

- **Python**: 3.8 or higher
- **Webcam**: Built-in or USB camera
- **OS**: Windows, Linux, or macOS
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space

### Step 2: Installation

```bash
# 1. Extract the project
unzip gesture_drone_project.zip
cd gesture_drone_project

# 2. Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Step 3: Collect Training Data

```bash
# Option A: Use launcher
./run.sh
# Select option 1

# Option B: Direct command
python3 src/data_collection/collect_static.py
```

**Data Collection Tips:**
- Collect 50-100 samples per gesture
- Vary hand position (left, center, right of frame)
- Change hand orientation slightly
- Use consistent lighting
- Press number keys (0-8) to capture
- Press 's' to save, 'q' to quit

### Step 4: Train a Model

```bash
# Start with KNN (fastest)
./run.sh  # Select option 3
# OR
python3 src/training/train_knn.py
```

**Training Time:**
- KNN: <1 second
- ANN: 30-60 seconds
- CNN: 60-120 seconds (needs image data)

### Step 5: Run the Controller

```bash
# With launcher
./run.sh  # Select option 6, 7, or 8

# Direct command
python3 src/controllers/advanced_controller.py --model knn
```

### Step 6: Use Gestures

**Basic Controls:**
1. Show **two open palms** (✋✋) to takeoff
2. Use **single-hand gestures** to control
3. Show **two fists** (✊✊) for emergency stop

**Advanced Features:**
- **Follow Mode**: Left fist + right open palm
- **Correction Mode**: Press 'c' to correct predictions
- **Quit**: Press 'q'

## 🎯 Complete Workflow

### Beginner Path (1-2 hours)
```bash
# 1. Collect data (15 min)
python3 src/data_collection/collect_static.py

# 2. Train KNN (instant)
python3 src/training/train_knn.py

# 3. Test (30 min)
python3 src/controllers/advanced_controller.py --model knn
```

### Intermediate Path (3-4 hours)
```bash
# 1. Collect more data (30 min)
python3 src/data_collection/collect_static.py

# 2. Train ANN (1 min)
python3 src/training/train_ann.py

# 3. Test and compare (1 hour)
python3 src/controllers/advanced_controller.py --model ann

# 4. Use online learning to improve
# Press 'c' during runtime to correct gestures
```

### Advanced Path (1-2 days)
```bash
# 1. Collect image data for CNN (1 hour)
python3 src/data_collection/collect_images.py

# 2. Collect dynamic gesture sequences (30 min)
python3 src/data_collection/collect_dynamic.py

# 3. Train CNN (2 min)
python3 src/training/train_cnn.py

# 4. Test all features
python3 src/controllers/advanced_controller.py --model cnn
```

## 🐛 Troubleshooting

### Camera Not Working
```bash
# Test camera
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.read()[0] else 'FAIL')"

# Try different camera index
# Edit config/config.py: CAMERA['camera_id'] = 1
```

### Import Errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Check Python version
python3 --version  # Should be 3.8+
```

### Low Accuracy
- Collect more diverse samples (100+ per gesture)
- Ensure good lighting
- Make distinct, exaggerated gestures
- Use CNN model for best results

### Model Not Found
```bash
# Make sure you trained the model first
python3 src/training/train_knn.py
```

## 📊 Expected Results

### Accuracy by Model
- **KNN**: 85-95% (fast, simple)
- **ANN**: 90-98% (balanced)
- **CNN**: 92-99% (best, requires more data)

### Performance
- **FPS**: 25-30 fps on average laptop
- **Latency**: <100ms gesture to command
- **Training**: Minutes, not hours

## 🎓 Learning Path

**Day 1 - Basics:**
- Install and setup
- Collect initial data (50 samples/gesture)
- Train KNN
- Test basic gestures

**Day 2 - Improve:**
- Collect more data (100 samples/gesture)
- Train ANN
- Compare with KNN
- Try online learning

**Day 3 - Advanced:**
- Collect image data
- Train CNN
- Test dynamic gestures
- Try follow mode

**Day 4 - Master:**
- Fine-tune models
- Test all features
- Prepare for real drone

## 🚁 Using with Real Drone

**For DJI Tello:**

1. Install djitellopy:
```bash
pip install djitellopy
```

2. Edit `src/controllers/advanced_controller.py`:
```python
# Replace:
from djitellopy import Tello
self.drone = Tello()
# Instead of:
self.drone = AdvancedMockDrone()
```

3. Connect to Tello WiFi

4. Test in safe area!

## ⚡ Quick Tips

- **Best lighting**: Bright, even, from front
- **Best distance**: 40-80cm from camera
- **Best background**: Simple, solid color
- **Best gestures**: Exaggerated, slow movements
- **Best data**: Varied positions and angles

## 📞 Support

If you encounter issues:
1. Check this guide
2. Read README.md
3. Check config/config.py settings
4. Verify Python and package versions

## ✅ Checklist

Before first use:
- [ ] Python 3.8+ installed
- [ ] All dependencies installed
- [ ] Camera working
- [ ] Collected training data (50+ samples/gesture)
- [ ] Trained at least one model
- [ ] Tested in mock mode

Ready for real drone:
- [ ] Tested extensively in mock mode
- [ ] All gestures work reliably
- [ ] Emergency stop tested
- [ ] Safe flying area prepared
- [ ] Backup control method ready

---

**Happy Flying! 🚁✨**
