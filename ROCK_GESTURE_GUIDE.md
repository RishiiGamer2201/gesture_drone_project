# 🤘 ROCK Gesture Collection Guide

## Quick Fix Applied ✅

The data collection scripts now support **10 gestures** (0-9) including ROCK!

---

## 🎯 How to Collect ROCK Gesture

### Step 1: Run Image Collector
```bash
python src/data_collection/collect_images.py
```

### Step 2: Show ROCK Gesture
Make the rock/metal sign (🤘):
- **Index finger**: UP
- **Pinky finger**: UP  
- **Middle finger**: DOWN
- **Ring finger**: DOWN
- **Thumb**: Can be up or down (flexible)

### Step 3: Press '9' Key
- Press and hold '9' on keyboard
- Each press captures one image
- Collect **50-100 images** for best results

### Step 4: Vary Your Hand
- Different angles (slightly left, right, tilted)
- Different distances from camera
- Different hand positions in frame
- Different lighting

---

## 📸 Collection Tips

### Good ROCK Gestures:
✅ Clear separation between fingers
✅ Index and pinky fully extended
✅ Middle and ring clearly down
✅ Hand clearly visible
✅ Good lighting

### What to Avoid:
❌ Fingers too close together
❌ Partially extended middle/ring
❌ Hand too far or too close
❌ Blurry images
❌ Hand partially out of frame

---

## 🎮 Testing ROCK Gesture Recognition

### After Training:
```bash
python src/controllers/final_controller.py
```

**To test:**
1. Takeoff with two palms
2. Show ROCK gesture (🤘)
3. Should see: "DYNAMIC MODE ACTIVATED"
4. Show ROCK again
5. Should see: "STATIC MODE ACTIVATED"

---

## 📊 Collection Progress

You should see this when collecting:

```
Samples Collected:
0:UP   50    ✓ Good
1:DOW  50    ✓ Good
2:LEF  50    ✓ Good
3:RIG  50    ✓ Good
4:HOV  50    ✓ Good
5:LAN  50    ✓ Good
6:FOR  50    ✓ Good
7:BAC  50    ✓ Good
8:FLI  50    ✓ Good
9:ROC  50    ✓ Good   ← ROCK gesture!
```

---

## 🔧 Troubleshooting

### ROCK Gesture Not Showing
**Problem:** Only see 0-8 in collector
**Solution:** ✅ FIXED! Now shows 0-9

### Can't Press '9'
**Problem:** Key not responding
**Solution:** Click on the OpenCV window first, then press '9'

### ROCK Not Detected in Controller
**Problem:** Shows ROCK but doesn't switch modes
**Solution:** 
1. Make sure you trained with ROCK data
2. Check fingers: Index + Pinky UP, Middle + Ring DOWN
3. Hold gesture for 0.5-1 second
4. Try 'd' key as alternative

---

## ✅ Complete Workflow

### 1. Collect All Gestures
```bash
python src/data_collection/collect_images.py
```
- Press 0-9 for each gesture
- Collect 50-100 images per gesture
- Press 'q' when done

### 2. Train CNN Model
```bash
python src/training/train_cnn.py
```
- Will train on all 10 gestures
- Includes ROCK gesture

### 3. Test Mode Switching
```bash
python src/controllers/final_controller.py
```
- Show ROCK to switch modes
- Or press 'd' key

---

## 🎓 ROCK Gesture Examples

### Correct ROCK (🤘):
```
     👆 Index UP
    /
   |
  | 👇 Middle DOWN
  | 👇 Ring DOWN
   \
    👆 Pinky UP
```

### Wrong Gestures:
- ✌️ Peace sign (index + middle up) - NOT rock
- 👌 OK sign (thumb + index circle) - NOT rock
- 🖖 Vulcan salute (all up, split) - NOT rock

---

## 💡 Pro Tips

1. **Collect in batches** - 10 images at a time
2. **Change position** - Move hand around frame
3. **Change angle** - Slight rotations help
4. **Change distance** - Near and far from camera
5. **Good lighting** - Consistent, bright light

---

## 📝 Quick Reference

| Key | Gesture | Description |
|-----|---------|-------------|
| 0 | UP | Index pointing up |
| 1 | DOWN | Index pointing down |
| 2 | LEFT | Index pointing left |
| 3 | RIGHT | Index pointing right |
| 4 | HOVER | Closed fist |
| 5 | LAND | Open palm |
| 6 | FORWARD | Thumbs up |
| 7 | BACKWARD | Thumbs down |
| 8 | FLIP | Peace sign ✌️ |
| 9 | ROCK | Rock sign 🤘 |

---

**Now you can collect all 10 gestures including ROCK!** 🤘🎉
