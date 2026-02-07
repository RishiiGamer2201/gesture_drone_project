# 🔧 Quick Fix Guide

## Issues Fixed

### 1. Two-Hand Detection Not Working
**Problem:** MediaPipe wasn't configured to detect 2 hands
**Fix:** Changed `max_num_hands=2` in MediaPipe initialization

### 2. Program Crashing After Takeoff
**Problem:** Missing `detect_all()` method in DynamicGestureDetector
**Fix:** Added the method to gesture_detection.py

### 3. AttributeError on detect_dynamic_gesture
**Problem:** Method name mismatch
**Fix:** Changed to `detect_all()` in controller

## How to Use

### Option 1: Use Simple Controller (Recommended)
```bash
python src/controllers/simple_controller.py --model knn
```

This is a simplified, robust version that:
- ✅ Detects 2 hands properly
- ✅ Has full error handling
- ✅ Works reliably
- ✅ Includes debug output

### Option 2: Use Advanced Controller (All Features)
```bash
python src/controllers/advanced_controller.py --model knn
```

This has all features but may show warnings (can ignore them).

## Testing Two-Hand Detection

1. Run the simple controller
2. Show BOTH hands to camera clearly
3. You should see: `DEBUG: Detected 2 hand(s)`
4. Make two open palms for TAKEOFF
5. Make two fists for EMERGENCY

## Debugging

If two-hand detection still doesn't work:

1. **Check camera:** Make sure both hands are clearly visible
2. **Lighting:** Ensure good, even lighting
3. **Distance:** Keep hands 40-80cm from camera
4. **Check debug output:** Look for "DEBUG: Detected X hand(s)" messages

## Quick Test

```python
# Test MediaPipe two-hand detection
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        print(f"Hands detected: {len(results.multi_hand_landmarks)}")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
```

## Support

If issues persist:
1. Use simple_controller.py (most reliable)
2. Check debug output for hand count
3. Ensure both hands are visible in frame
4. Try adjusting camera angle/lighting
