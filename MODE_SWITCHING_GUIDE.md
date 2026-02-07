# 🎮 Mode Switching System - Complete Guide

## 🎯 NEW FEATURE: Exclusive Mode System

The controller now has **TWO EXCLUSIVE MODES** that never interfere with each other:

### **STATIC MODE** (Default)
- Static gestures work (UP, DOWN, LEFT, RIGHT, etc.)
- Dynamic gestures DISABLED
- Full drone control
- Can land normally

### **DYNAMIC MODE** (Opt-in)
- Dynamic gestures work (Circle, Swipe, Photo)
- Static gestures DISABLED
- Drone automatically HOVERS
- Cannot land (must switch to static first)

---

## 🔄 How to Switch Modes

### Method 1: ROCK Gesture (🤘)
Make the rock/metal sign:
- Index finger UP
- Pinky finger UP
- Middle & ring fingers DOWN
- Hold for 0.5 seconds

### Method 2: Press 'd' Key
Simply press 'd' on your keyboard

**Cooldown:** 1.5 seconds between mode switches

---

## 📋 Step-by-Step Usage

### Starting Up
```bash
python src/controllers/final_controller.py
```

**Initial State:** STATIC MODE (green display)

### Basic Flight (Static Mode)
1. Show two open palms → TAKEOFF
2. Use single hand gestures:
   - Point up → Move UP
   - Point down → Move DOWN
   - Point left → Move LEFT
   - etc.
3. Show open palm → LAND

### Using Dynamic Features
1. **While flying**, show ROCK gesture (🤘) or press 'd'
2. **Mode switches to DYNAMIC** (orange banner appears)
3. **Drone enters HOVER** (won't respond to static gestures)
4. Perform dynamic gestures:
   - Draw circle → Orbit
   - Swipe hand → Fast movement
   - Fist → Open → Photo capture
5. **To exit:** Show ROCK (🤘) or press 'd' again
6. **Mode switches to STATIC**
7. Now you can control or land normally

---

## 🎯 Why This System?

### Problem Solved:
- **Before:** Model confused circles/swipes with static gestures
- **Before:** Photo capture triggered on both open and close
- **Before:** Couldn't tell if user wants dynamic or static

### Solution:
- **Now:** Modes are exclusive - no confusion
- **Now:** Photo only on fist→open, not open→fist
- **Now:** Clear mode indication on screen

---

## 📊 Visual Indicators

### On Screen:

**STATIC MODE:**
```
Mode: STATIC            [Green text]
Status: FLYING ✈️
Hands: 1
Gesture: UP
```

**DYNAMIC MODE:**
```
Mode: DYNAMIC           [Orange text]
╔════════════════════════════════╗
║      DYNAMIC MODE              ║ [Orange banner]
║  Show ROCK (🤘) or press 'd'   ║
╚════════════════════════════════╝
Status: FLYING ✈️
Hands: 1
Dynamic: CIRCLE
Gesture: HOVER (Dynamic Mode)
```

---

## 🎮 Complete Control Reference

### TWO-HAND GESTURES (Always Active in Both Modes)
| Gesture | Action |
|---------|--------|
| ✋✋ Two open palms | TAKEOFF |
| ✊✊ Two fists | EMERGENCY STOP |
| ✊✋ Fist + Open | Toggle FOLLOW MODE |

### STATIC MODE (Default)
| Gesture | Action |
|---------|--------|
| Point UP | Move up |
| Point DOWN | Move down |
| Point LEFT | Move left |
| Point RIGHT | Move right |
| Thumbs UP | Move forward |
| Thumbs DOWN | Move backward |
| Fist | Hover |
| Open palm | Land |
| 🤘 ROCK | **Switch to DYNAMIC** |

### DYNAMIC MODE (After switching)
| Gesture | Action |
|---------|--------|
| Draw circle | Orbit mode |
| Swipe LEFT | Fast left |
| Swipe RIGHT | Fast right |
| Swipe UP | Fast up |
| Swipe DOWN | Fast down |
| Fist → Open | Capture photo |
| 🤘 ROCK | **Switch to STATIC** |

### KEYBOARD (Always Available)
| Key | Action |
|-----|--------|
| 'd' | Switch modes |
| 'q' | Quit program |

---

## ⚠️ Important Notes

### To Land the Drone:
1. **If in DYNAMIC mode:** Switch to STATIC first (ROCK or 'd')
2. **Then** show open palm to land
3. **Cannot land directly from DYNAMIC mode**

### Photo Capture (Dynamic Mode Only):
- Make a **tight fist**
- Hold for 0.5 seconds
- **Open hand fully**
- Hold for 0.5 seconds
- Photo captured!
- **Closing hand again won't trigger** (fixed!)

### Mode Switch Cooldown:
- 1.5 seconds between switches
- Prevents accidental mode changes
- Clear console messages when switching

---

## 🧪 Testing the New System

### Test 1: Mode Switching
```
1. Start program (STATIC mode)
2. Takeoff with two palms
3. Show ROCK gesture 🤘
   → Should see "DYNAMIC MODE ACTIVATED" 
   → Orange banner appears
4. Show ROCK again 🤘
   → Should see "STATIC MODE ACTIVATED"
   → Banner disappears
```

### Test 2: Static Gestures
```
1. In STATIC mode
2. Point UP → Drone moves up ✓
3. Point DOWN → Drone moves down ✓
4. Switch to DYNAMIC mode
5. Point UP → Nothing happens ✓ (static disabled)
```

### Test 3: Dynamic Gestures
```
1. In STATIC mode
2. Draw circle → Nothing happens ✓ (dynamic disabled)
3. Switch to DYNAMIC mode
4. Draw circle → "CIRCLE detected" ✓
5. Swipe hand → Fast movement ✓
```

### Test 4: Photo Capture
```
1. Switch to DYNAMIC mode
2. Make fist → wait
3. Open hand → "📸 PHOTO CAPTURED!" ✓
4. Close hand → No photo ✓ (fixed!)
```

---

## 🐛 Troubleshooting

### Mode Won't Switch
- **Check:** 1.5 second cooldown hasn't passed
- **Try:** Use 'd' key instead of gesture
- **Verify:** Console shows mode switch message

### ROCK Gesture Not Detected
- **Make sure:** Index and pinky UP, middle and ring DOWN
- **Hold:** Keep gesture for 0.5-1 second
- **Try:** 'd' key as alternative

### Can't Land in Dynamic Mode
- **This is intentional!**
- **Solution:** Switch to STATIC first, then land
- **Reason:** Prevents accidental landing during dynamic operations

### Photo Triggers Twice
- **Should be fixed!**
- **Now:** Only triggers on fist→open
- **Not:** On open→fist (closing)

---

## 💡 Pro Tips

1. **Start in Static** - Get drone in the air first
2. **Switch to Dynamic** - Only when you need those features
3. **Switch back** - Return to static for landing
4. **Use 'd' key** - Fastest, most reliable mode switch
5. **Watch the banner** - Orange banner = Dynamic mode
6. **Check console** - Clear messages on every mode switch

---

## 🎓 Advanced Usage

### Workflow Example:
```
1. TAKEOFF (two palms)
2. STATIC mode: Fly to position
3. ROCK gesture: Enter DYNAMIC
4. Circle gesture: Orbit around object
5. Fist→Open: Take photo
6. ROCK gesture: Exit to STATIC
7. Fly to another position
8. Repeat steps 3-6 as needed
9. Open palm: LAND
```

---

## ✅ Success Checklist

- [ ] Can switch modes with ROCK gesture
- [ ] Can switch modes with 'd' key
- [ ] Orange banner appears in dynamic mode
- [ ] Static gestures work in static mode
- [ ] Static gestures DON'T work in dynamic mode
- [ ] Dynamic gestures work in dynamic mode
- [ ] Dynamic gestures DON'T work in static mode
- [ ] Photo only triggers on fist→open, not open→fist
- [ ] Can land after switching back to static

---

**This system eliminates all confusion between static and dynamic gestures!** 🎉
