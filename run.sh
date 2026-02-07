#!/bin/bash
# Gesture Drone Project Launcher

echo "🚁 Gesture Drone Control System"
echo "==============================="
echo ""
echo "Choose an option:"
echo "1. Collect Static Gesture Data"
echo "2. Collect Dynamic Gesture Sequences  
echo "3. Train KNN Model"
echo "4. Train ANN Model"
echo "5. Train CNN Model"
echo "6. Run Advanced Controller (KNN)"
echo "7. Run Advanced Controller (ANN)"
echo "8. Run Advanced Controller (CNN)"
echo "9. Exit"
echo ""
read -p "Enter choice [1-9]: " choice

case $choice in
    1) python3 src/data_collection/collect_static.py ;;
    2) python3 src/data_collection/collect_dynamic.py ;;
    3) python3 src/training/train_knn.py ;;
    4) python3 src/training/train_ann.py ;;
    5) python3 src/training/train_cnn.py ;;
    6) python3 src/controllers/advanced_controller.py --model knn ;;
    7) python3 src/controllers/advanced_controller.py --model ann ;;
    8) python3 src/controllers/advanced_controller.py --model cnn ;;
    9) echo "Goodbye!"; exit 0 ;;
    *) echo "Invalid choice"; exit 1 ;;
esac
