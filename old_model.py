"""
Real-time ASL Recognition System
Uses MediaPipe Tasks API + TensorFlow for robust sign language detection
Implements state machine to prevent false positives
WITH preprocessing support for A-Z recognition
"""

import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from enum import Enum
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import os
import json
import pickle


class HandState(Enum):
    """State machine for hand detection pipeline"""
    NO_HAND = 0
    HAND_DETECTED = 1
    HAND_CONFIRMED = 2


class ASLRecognizer:
    def __init__(self, model_path, labels_path, hand_landmarker_path, 
                 config_path=None, scaler_path=None):
        """
        Initialize ASL recognition system
        
        Args:
            model_path: Path to TensorFlow .keras model
            labels_path: Path to labels text file
            hand_landmarker_path: Path to hand_landmarker.task
            config_path: Path to model_config.json (optional)
            scaler_path: Path to scaler.pkl (optional, for preprocessing)
        """
        # Verify files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        if not os.path.exists(hand_landmarker_path):
            raise FileNotFoundError(f"Hand landmarker file not found: {hand_landmarker_path}")
        
        print(f"Loading model from: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        print(f"✓ Model loaded successfully")
        
        # Load labels
        print(f"Loading labels from: {labels_path}")
        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
        print(f"✓ Loaded {len(self.labels)} labels: {self.labels}")
        
        # Load config if available
        self.preprocessing_enabled = False
        if config_path and os.path.exists(config_path):
            print(f"Loading config from: {config_path}")
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.preprocessing_enabled = config.get("preprocessing_enabled", False)
            print(f"✓ Preprocessing: {'ENABLED' if self.preprocessing_enabled else 'DISABLED'}")
        
        # Load scaler if preprocessing is enabled
        self.scaler = None
        if self.preprocessing_enabled:
            if scaler_path and os.path.exists(scaler_path):
                print(f"Loading scaler from: {scaler_path}")
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"✓ Scaler loaded successfully")
            else:
                print(f"⚠ Warning: Preprocessing enabled but scaler not found!")
                print(f"   Expected at: {scaler_path}")
                print(f"   Predictions may be inaccurate!")
        
        # Initialize MediaPipe Hand Landmarker
        print(f"Initializing MediaPipe Hand Landmarker: {hand_landmarker_path}")
        base_options = python.BaseOptions(model_asset_path=hand_landmarker_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        print("✓ Hand Landmarker initialized successfully")
        
        # State machine
        self.state = HandState.NO_HAND
        self.current_prediction = None
        self.saved_label = None
        
        # Buffers for stability checking
        self.landmark_buffer = deque(maxlen=15)  # ~0.5s at 30fps
        self.prediction_buffer = deque(maxlen=30)  # ~1s at 30fps
        
        # Stability thresholds
        self.STABILITY_THRESHOLD = 0.02  # Max movement between frames
        self.MIN_STABLE_FRAMES = 15  # Frames needed for stability
        self.CONFIDENCE_THRESHOLD = 0.80  # Minimum prediction confidence (higher for 25 classes)
        self.MAJORITY_THRESHOLD = 0.70  # 70% agreement for prediction
        
        # Timing
        self.last_prediction_time = 0
        
        # Prediction history for display
        self.prediction_history = []
        
        print("\n" + "="*60)
        print("ASL Recognition System Initialized")
        print(f"Ready to recognize {len(self.labels)} signs!")
        print("="*60)
        
    def extract_landmarks(self, hand_landmarks):
        """
        Extract normalized x,y coordinates from hand landmarks
        
        Args:
            hand_landmarks: MediaPipe hand landmarks object
            
        Returns:
            numpy array of shape (42,) containing flattened x,y coordinates
        """
        landmarks = []
        for landmark in hand_landmarks:
            landmarks.extend([landmark.x, landmark.y])
        return np.array(landmarks, dtype=np.float32)
    
    def preprocess_landmarks(self, landmarks):
        """
        Apply same preprocessing as training
        
        Args:
            landmarks: Raw landmark array (42,)
            
        Returns:
            Preprocessed landmarks
        """
        if not self.preprocessing_enabled or self.scaler is None:
            return landmarks
        
        # Center around wrist (landmark 0)
        processed = landmarks.copy()
        wrist_x = processed[0]
        wrist_y = processed[1]
        processed[::2] -= wrist_x  # Center x coordinates
        processed[1::2] -= wrist_y  # Center y coordinates
        
        # Standardize using saved scaler
        processed = processed.reshape(1, -1)
        processed = self.scaler.transform(processed)
        processed = processed.flatten()
        
        return processed
    
    def check_hand_stability(self):
        """
        Check if hand landmarks are stable (minimal movement)
        
        Returns:
            bool: True if hand is stable enough for prediction
        """
        if len(self.landmark_buffer) < self.MIN_STABLE_FRAMES:
            return False
        
        # Calculate variance across recent frames
        landmarks_array = np.array(self.landmark_buffer)
        movement = np.std(landmarks_array, axis=0)
        max_movement = np.max(movement)
        
        return max_movement < self.STABILITY_THRESHOLD
    
    def get_majority_prediction(self):
        """
        Get prediction with majority voting from buffer
        
        Returns:
            tuple: (label, confidence) or (None, 0) if no majority
        """
        if len(self.prediction_buffer) < self.MIN_STABLE_FRAMES:
            return None, 0.0
        
        # Count occurrences of each prediction
        predictions = list(self.prediction_buffer)
        unique, counts = np.unique(predictions, return_counts=True)
        
        # Get most common prediction
        max_idx = np.argmax(counts)
        most_common = unique[max_idx]
        confidence = counts[max_idx] / len(predictions)
        
        # Check if it meets majority threshold
        if confidence >= self.MAJORITY_THRESHOLD:
            return most_common, confidence
        
        return None, 0.0
    
    def predict_sign(self, landmarks):
        """
        Predict ASL sign from landmarks
        
        Args:
            landmarks: numpy array of shape (42,) - RAW landmarks
            
        Returns:
            tuple: (predicted_label, confidence)
        """
        # Preprocess landmarks (applies centering + scaling if enabled)
        processed_landmarks = self.preprocess_landmarks(landmarks)
        
        # Reshape for model input
        landmarks_input = processed_landmarks.reshape(1, -1)
        
        # Get prediction
        predictions = self.model.predict(landmarks_input, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        
        # Only return if confidence is high enough
        if confidence >= self.CONFIDENCE_THRESHOLD:
            return self.labels[class_idx], float(confidence)
        
        return None, 0.0
    
    def process_frame(self, frame):
        """
        Process single frame through the detection pipeline
        
        Args:
            frame: OpenCV BGR image
            
        Returns:
            tuple: (processed_frame, status_message)
        """
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect hand landmarks
        detection_result = self.landmarker.detect(mp_image)
        
        # Check if hand is detected
        if not detection_result.hand_landmarks or len(detection_result.hand_landmarks) == 0:
            # NO HAND DETECTED - Reset everything
            if self.state == HandState.HAND_CONFIRMED:
                # Hand was removed after confirmation - ready for next sign
                self.reset_state()
                status = "Sign saved! Show next sign"
            else:
                # No hand present
                self.reset_state()
                status = "No hand detected"
            
            return frame, status
        
        # HAND IS PRESENT
        hand_landmarks = detection_result.hand_landmarks[0]
        
        # Draw landmarks on frame
        h, w, _ = frame.shape
        
        # Draw connections first
        connections = [
            (0,1),(1,2),(2,3),(3,4),  # Thumb
            (0,5),(5,6),(6,7),(7,8),  # Index
            (0,9),(9,10),(10,11),(11,12),  # Middle
            (0,13),(13,14),(14,15),(15,16),  # Ring
            (0,17),(17,18),(18,19),(19,20),  # Pinky
            (5,9),(9,13),(13,17)  # Palm
        ]
        for start, end in connections:
            start_point = (int(hand_landmarks[start].x * w), int(hand_landmarks[start].y * h))
            end_point = (int(hand_landmarks[end].x * w), int(hand_landmarks[end].y * h))
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
        # Draw landmarks
        for landmark in hand_landmarks:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
        
        # Extract landmarks (RAW - preprocessing happens in predict_sign)
        landmarks = self.extract_landmarks(hand_landmarks)
        self.landmark_buffer.append(landmarks)
        
        # STATE MACHINE LOGIC
        
        if self.state == HandState.NO_HAND:
            # Transition to HAND_DETECTED
            self.state = HandState.HAND_DETECTED
            status = "Hold hand steady..."
            
        elif self.state == HandState.HAND_DETECTED:
            # Check if hand is stable
            if self.check_hand_stability():
                # Make prediction
                predicted_label, confidence = self.predict_sign(landmarks)
                
                if predicted_label:
                    self.prediction_buffer.append(predicted_label)
                    
                    # Check for majority vote
                    majority_label, majority_conf = self.get_majority_prediction()
                    
                    if majority_label:
                        # CONFIRMED PREDICTION
                        self.state = HandState.HAND_CONFIRMED
                        self.saved_label = majority_label
                        self.last_prediction_time = time.time()
                        
                        # Add to history
                        self.prediction_history.append(majority_label)
                        if len(self.prediction_history) > 10:
                            self.prediction_history.pop(0)
                        
                        status = f"✓ Saved: {majority_label} ({majority_conf:.0%})"
                    else:
                        status = f"Detecting: {predicted_label} ({confidence:.0%})"
                else:
                    status = "Hold hand steady..."
            else:
                status = "Hold hand steady..."
                
        elif self.state == HandState.HAND_CONFIRMED:
            # Prediction locked - wait for hand removal
            status = f"✓ Saved: {self.saved_label} — Remove hand for next sign"
        
        return frame, status
    
    def reset_state(self):
        """Reset all buffers and state"""
        self.state = HandState.NO_HAND
        self.current_prediction = None
        self.landmark_buffer.clear()
        self.prediction_buffer.clear()
    
    def draw_ui(self, frame, status):
        """Draw enhanced UI on frame"""
        h, w, _ = frame.shape
        
        # Main status box
        cv2.rectangle(frame, (10, 10), (800, 80), (0, 0, 0), -1)
        cv2.putText(frame, status, (20, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        # State indicator
        state_color = {
            HandState.NO_HAND: (128, 128, 128),
            HandState.HAND_DETECTED: (0, 165, 255),
            HandState.HAND_CONFIRMED: (0, 255, 0)
        }
        cv2.circle(frame, (850, 45), 25, state_color[self.state], -1)
        
        # Preprocessing indicator
        if self.preprocessing_enabled:
            cv2.putText(frame, "PREP: ON", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Prediction history
        if self.prediction_history:
            history_text = "History: " + " → ".join(self.prediction_history)
            cv2.rectangle(frame, (10, h - 60), (min(len(history_text) * 15 + 20, w - 10), h - 10), (0, 0, 0), -1)
            cv2.putText(frame, history_text, (20, h - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Model info
        cv2.rectangle(frame, (10, 110), (350, 170), (0, 0, 0), -1)
        cv2.putText(frame, f"Model: {len(self.labels)} classes", (20, 135),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"Labels: {', '.join(self.labels[:5])}...", (20, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        return frame
    
    def run(self):
        """Main loop for real-time ASL recognition"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\n" + "="*60)
        print("ASL Recognition System Started")
        print("="*60)
        print("Controls:")
        print("  'ESC' - Quit")
        print("  'r' - Reset state")
        print("  'c' - Clear history")
        print("="*60 + "\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            processed_frame, status = self.process_frame(frame)
            
            # Draw UI
            processed_frame = self.draw_ui(processed_frame, status)
                      
            # Display frame
            cv2.imshow('ASL Recognition - A-Z', processed_frame)
           
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('r'):
                self.reset_state()
                print("System reset")
            elif key == ord('c'):
                self.prediction_history.clear()
                print("History cleared")
      
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.landmarker.close()
        
        # Print final history
        if self.prediction_history:
            print("\n" + "="*60)
            print("SESSION SUMMARY")
            print("="*60)
            print(f"Recognized signs: {' → '.join(self.prediction_history)}")
            print("="*60)


def main():
    """Entry point"""
    print("ASL Recognition System - Initialization")
    print("="*60)
    
    # File paths
    MODEL_PATH = "model.keras"
    LABELS_PATH = "labels.txt"
    HAND_LANDMARKER_PATH = "hand_landmarker.task"
    CONFIG_PATH = "model_config.json"
    SCALER_PATH = "scaler.pkl"
    
    # Check required files
    missing_files = []
    if not os.path.exists(MODEL_PATH):
        missing_files.append(f"Model file: {MODEL_PATH}")
    if not os.path.exists(LABELS_PATH):
        missing_files.append(f"Labels file: {LABELS_PATH}")
    if not os.path.exists(HAND_LANDMARKER_PATH):
        missing_files.append(f"Hand landmarker: {HAND_LANDMARKER_PATH}")
    
    if missing_files:
        print("\n❌ ERROR: Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print(f"\nCurrent directory: {os.getcwd()}")
        return
    
    # Initialize and run
    try:
        recognizer = ASLRecognizer(
            MODEL_PATH, 
            LABELS_PATH, 
            HAND_LANDMARKER_PATH,
            CONFIG_PATH,
            SCALER_PATH
        )
        recognizer.run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()