"""
Online Learning Module
=======================
Enables adaptive learning and model improvement during runtime.
Collects user corrections and fine-tunes the model.
"""

import numpy as np
import pickle
import os
from datetime import datetime
from collections import deque
import cv2

class OnlineLearningManager:
    """Manage online learning and model adaptation"""
    
    def __init__(self, config, model, model_type='ann'):
        self.config = config
        self.model = model
        self.model_type = model_type
        
        # Correction buffer
        self.correction_buffer = deque(maxlen=config['buffer_size'])
        self.correction_file = config['correction_file']
        
        # Statistics
        self.total_corrections = 0
        self.gesture_correction_counts = {}
        
        # Load existing corrections if available
        self._load_corrections()
        
    def _load_corrections(self):
        """Load previously saved corrections"""
        if os.path.exists(self.correction_file):
            try:
                with open(self.correction_file, 'rb') as f:
                    data = pickle.load(f)
                    self.correction_buffer = deque(data['corrections'], 
                                                   maxlen=self.config['buffer_size'])
                    self.total_corrections = data.get('total_corrections', 0)
                    self.gesture_correction_counts = data.get('counts', {})
                print(f"✓ Loaded {len(self.correction_buffer)} previous corrections")
            except Exception as e:
                print(f"⚠️  Could not load corrections: {e}")
    
    def add_correction(self, features, predicted_label, correct_label):
        """
        Add a user correction to the buffer
        
        Args:
            features: Input features (landmarks or image)
            predicted_label: What the model predicted
            correct_label: What the user says it should be
        """
        if predicted_label == correct_label:
            return  # No correction needed
        
        correction = {
            'features': features,
            'predicted': predicted_label,
            'correct': correct_label,
            'timestamp': datetime.now()
        }
        
        self.correction_buffer.append(correction)
        self.total_corrections += 1
        
        # Update statistics
        if correct_label not in self.gesture_correction_counts:
            self.gesture_correction_counts[correct_label] = 0
        self.gesture_correction_counts[correct_label] += 1
        
        print(f"✓ Correction added: {predicted_label} → {correct_label}")
        
        # Auto-save periodically
        if len(self.correction_buffer) % self.config['save_interval'] == 0:
            self._save_corrections()
        
        # Check if we should retrain
        if len(self.correction_buffer) >= self.config['retrain_threshold']:
            print(f"\n📚 Correction buffer full ({len(self.correction_buffer)} samples)")
            print("Ready for model fine-tuning!")
            return True  # Signal that retraining is recommended
        
        return False
    
    def _save_corrections(self):
        """Save corrections to disk"""
        os.makedirs(os.path.dirname(self.correction_file), exist_ok=True)
        
        data = {
            'corrections': list(self.correction_buffer),
            'total_corrections': self.total_corrections,
            'counts': self.gesture_correction_counts,
            'last_updated': datetime.now()
        }
        
        with open(self.correction_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"💾 Saved {len(self.correction_buffer)} corrections")
    
    def fine_tune_model(self, original_train_data=None):
        """
        Fine-tune the model using corrections
        
        Args:
            original_train_data: Optional original training data to include
        
        Returns:
            fine_tuned_model: Updated model
        """
        if len(self.correction_buffer) < self.config['retrain_threshold']:
            print("⚠️  Not enough corrections for fine-tuning")
            return self.model
        
        print("\n" + "=" * 60)
        print("FINE-TUNING MODEL WITH USER CORRECTIONS")
        print("=" * 60)
        
        # Extract features and labels from corrections
        X_corrections = []
        y_corrections = []
        
        for correction in self.correction_buffer:
            X_corrections.append(correction['features'])
            y_corrections.append(correction['correct'])
        
        X_corrections = np.array(X_corrections)
        y_corrections = np.array(y_corrections)
        
        print(f"\n📊 Correction dataset:")
        print(f"   Total corrections: {len(X_corrections)}")
        
        unique_labels, counts = np.unique(y_corrections, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"   Gesture {label}: {count} corrections")
        
        # Fine-tune based on model type
        if self.model_type == 'ann' or self.model_type == 'cnn':
            self._fine_tune_neural_network(X_corrections, y_corrections)
        elif self.model_type == 'knn':
            self._fine_tune_knn(X_corrections, y_corrections)
        
        print("\n✓ Fine-tuning complete!")
        print("=" * 60)
        
        # Clear correction buffer after fine-tuning
        self.correction_buffer.clear()
        
        return self.model
    
    def _fine_tune_neural_network(self, X, y):
        """Fine-tune neural network (ANN or CNN)"""
        from tensorflow import keras
        
        # Prepare data
        if self.model_type == 'cnn':
            # Ensure correct shape for CNN
            if len(X.shape) == 3:
                X = X.reshape(-1, X.shape[1], X.shape[2], 1)
        
        # Convert labels to categorical
        num_classes = len(np.unique(y))
        y_cat = keras.utils.to_categorical(y, num_classes=num_classes)
        
        # Reduce learning rate for fine-tuning
        original_lr = keras.backend.get_value(self.model.optimizer.learning_rate)
        new_lr = original_lr * self.config['learning_rate_multiplier']
        keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
        
        print(f"\n🔧 Fine-tuning with learning rate: {new_lr}")
        
        # Fine-tune with few epochs
        history = self.model.fit(
            X, y_cat,
            epochs=10,
            batch_size=4,
            verbose=1,
            validation_split=0.2 if len(X) > 10 else 0
        )
        
        # Restore original learning rate
        keras.backend.set_value(self.model.optimizer.learning_rate, original_lr)
        
        print(f"✓ Final accuracy: {history.history['accuracy'][-1]*100:.2f}%")
    
    def _fine_tune_knn(self, X, y):
        """Fine-tune KNN by adding new training samples"""
        print("\n🔧 Adding corrections to KNN training set")
        
        # For KNN, we simply retrain with combined data
        # This requires access to original training data
        # For now, we'll just inform the user
        print("⚠️  KNN fine-tuning requires full retraining")
        print("   Save corrections and retrain with original data + corrections")
    
    def get_statistics(self):
        """Get correction statistics"""
        return {
            'total_corrections': self.total_corrections,
            'buffer_size': len(self.correction_buffer),
            'gesture_counts': self.gesture_correction_counts,
            'ready_for_retrain': len(self.correction_buffer) >= self.config['retrain_threshold']
        }
    
    def display_correction_ui(self, frame, current_gesture, available_gestures):
        """
        Display UI for user to correct predictions
        
        Args:
            frame: Video frame
            current_gesture: Currently predicted gesture
            available_gestures: List of all gesture names
        
        Returns:
            corrected_gesture: User-selected correct gesture (or None)
        """
        h, w = frame.shape[:2]
        
        # Draw correction panel
        panel_width = 400
        panel_height = 300
        panel_x = w - panel_width - 20
        panel_y = 20
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "Correction Mode", (panel_x + 10, panel_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Current prediction
        cv2.putText(frame, f"Predicted: {current_gesture}", (panel_x + 10, panel_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # Instructions
        cv2.putText(frame, "Press number key for correct gesture:", 
                   (panel_x + 10, panel_y + 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # List gestures with key bindings
        y_offset = 120
        for i, gesture in enumerate(available_gestures[:9]):
            color = (0, 255, 0) if gesture == current_gesture else (200, 200, 200)
            cv2.putText(frame, f"{i}: {gesture}", 
                       (panel_x + 10, panel_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 25
        
        # Statistics
        stats = self.get_statistics()
        cv2.putText(frame, f"Corrections: {stats['buffer_size']}/{self.config['retrain_threshold']}", 
                   (panel_x + 10, panel_y + panel_height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return frame
    
    def export_corrections_for_retraining(self, output_file):
        """
        Export corrections in format suitable for retraining
        
        Args:
            output_file: Path to save corrections
        """
        if len(self.correction_buffer) == 0:
            print("No corrections to export")
            return
        
        # Extract features and labels
        features = []
        labels = []
        
        for correction in self.correction_buffer:
            features.append(correction['features'])
            labels.append(correction['correct'])
        
        features = np.array(features)
        labels = np.array(labels)
        
        # Save as numpy arrays
        np.savez(output_file, features=features, labels=labels)
        
        print(f"✓ Exported {len(features)} corrections to {output_file}")
    
    def __del__(self):
        """Save corrections on cleanup"""
        if len(self.correction_buffer) > 0:
            self._save_corrections()


class CorrectionInterface:
    """Interactive interface for collecting corrections"""
    
    def __init__(self, gesture_names):
        self.gesture_names = gesture_names
        self.correction_mode = False
        self.last_features = None
        self.last_prediction = None
    
    def enable_correction_mode(self, features, prediction):
        """Enable correction mode with current prediction"""
        self.correction_mode = True
        self.last_features = features
        self.last_prediction = prediction
    
    def check_for_correction(self, key_press):
        """
        Check if user pressed a key to correct the gesture
        
        Args:
            key_press: Key pressed by user
        
        Returns:
            (corrected_label, features) or (None, None)
        """
        if not self.correction_mode:
            return None, None
        
        # Check if key is a number 0-9
        if key_press >= ord('0') and key_press <= ord('9'):
            gesture_id = key_press - ord('0')
            
            if gesture_id < len(self.gesture_names):
                self.correction_mode = False
                return gesture_id, self.last_features
        
        # Press 'c' to cancel correction
        elif key_press == ord('c'):
            self.correction_mode = False
        
        return None, None
