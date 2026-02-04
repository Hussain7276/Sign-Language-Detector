"""
ASL Dataset Collection Tool - Single Click Collection
Press a key once, captures ONE frame, then auto-generates 99 augmented samples
Much easier and faster!
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import json
from datetime import datetime


class ASLDataCollector:
    def __init__(self, hand_landmarker_path, output_dir="dataset"):
        """
        Initialize data collection system
        
        Args:
            hand_landmarker_path: Path to hand_landmarker.task
            output_dir: Directory to save collected data
        """
        # Create output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize MediaPipe Hand Landmarker
        print(f"Initializing MediaPipe Hand Landmarker: {hand_landmarker_path}")
        base_options = python.BaseOptions(model_asset_path=hand_landmarker_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        print("Hand Landmarker initialized")
        
        # Data storage
        self.all_landmarks = []
        self.all_labels = []
        
        # Current state
        self.current_label = None
        self.last_captured_landmarks = None
        
        # Collection settings
        self.samples_per_capture = 100  # Generate 100 samples from each capture
        
        # Sign definitions - CUSTOMIZE THIS FOR YOUR SIGNS
        self.sign_keys = {
            'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D', 'e': 'E',
            'f': 'F', 'g': 'G', 'h': 'H', 'i': 'I', 'j': 'J',
            'k': 'K', 'l': 'L', 'm': 'M', 'n': 'N', 'o': 'O',
            'p': 'P', 'q': 'Q', 'r': 'R', 's': 'S', 't': 'T',
            'u': 'U', 'v': 'V', 'w': 'W', 'x': 'X', 'y': 'Y',
            'z': 'Z',
            '0': 'nothing', '1': 'hello', '2': 'thanks', '3': 'yes', '4': 'no'
        }
        
        # Statistics per label
        self.label_counts = {}
        self.captures_per_label = {}
        
    def extract_landmarks(self, hand_landmarks):
        """Extract normalized x,y coordinates from hand landmarks"""
        landmarks = []
        for landmark in hand_landmarks:
            landmarks.extend([landmark.x, landmark.y])
        return np.array(landmarks, dtype=np.float32)
    
    def augment_landmarks(self, landmarks, num_samples=100):
        """
        Generate augmented samples from single landmark capture
        
        Args:
            landmarks: Original landmark array (42,)
            num_samples: Number of augmented samples to generate
            
        Returns:
            List of augmented landmark arrays
        """
        augmented = []
        
        # Always include the original
        augmented.append(landmarks.copy())
        
        # Generate augmented samples
        for i in range(num_samples - 1):
            aug_landmarks = landmarks.copy()
            
            # Random noise (small positional variations)
            noise = np.random.normal(0, 0.01, aug_landmarks.shape)
            aug_landmarks += noise
            
            # Random scaling (simulate hand size variation)
            scale = np.random.uniform(0.95, 1.05)
            # Scale around center point
            center_x = np.mean(aug_landmarks[::2])
            center_y = np.mean(aug_landmarks[1::2])
            aug_landmarks[::2] = center_x + (aug_landmarks[::2] - center_x) * scale
            aug_landmarks[1::2] = center_y + (aug_landmarks[1::2] - center_y) * scale
            
            # Random rotation (simulate hand angle variation)
            angle = np.random.uniform(-0.15, 0.15)  # radians
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            
            for j in range(0, len(aug_landmarks), 2):
                x = aug_landmarks[j] - center_x
                y = aug_landmarks[j+1] - center_y
                aug_landmarks[j] = center_x + (x * cos_angle - y * sin_angle)
                aug_landmarks[j+1] = center_y + (x * sin_angle + y * cos_angle)
            
            # Random translation (simulate hand position variation)
            tx = np.random.uniform(-0.05, 0.05)
            ty = np.random.uniform(-0.05, 0.05)
            aug_landmarks[::2] += tx
            aug_landmarks[1::2] += ty
            
            # Keep landmarks in valid range [0, 1]
            aug_landmarks = np.clip(aug_landmarks, 0, 1)
            
            augmented.append(aug_landmarks)
        
        return augmented
    
    def capture_and_augment(self, landmarks, label):
        """Capture landmarks and generate augmented samples"""
        print(f"\n📸 Captured {label}! Generating {self.samples_per_capture} samples...")
        
        # Generate augmented samples
        augmented_samples = self.augment_landmarks(landmarks, self.samples_per_capture)
        
        # Add all samples to dataset
        for aug_landmarks in augmented_samples:
            self.all_landmarks.append(aug_landmarks)
            self.all_labels.append(label)
        
        # Update statistics
        if label not in self.label_counts:
            self.label_counts[label] = 0
            self.captures_per_label[label] = 0
        
        self.label_counts[label] += len(augmented_samples)
        self.captures_per_label[label] += 1
        
        print(f"✓ Added {len(augmented_samples)} samples for '{label}'")
        print(f"  Total for '{label}': {self.captures_per_label[label]} captures = {self.label_counts[label]} samples")
        
    def save_dataset(self):
        """Save collected dataset to files"""
        if len(self.all_landmarks) == 0:
            print("No data to save!")
            return
        
        # Convert to numpy arrays
        landmarks_array = np.array(self.all_landmarks)
        labels_array = np.array(self.all_labels)
        
        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as .npy files
        landmarks_file = os.path.join(self.output_dir, f"landmarks_{timestamp}.npy")
        labels_file = os.path.join(self.output_dir, f"labels_{timestamp}.npy")
        
        np.save(landmarks_file, landmarks_array)
        np.save(labels_file, labels_array)
        
        # Save metadata
        metadata = {
            'total_samples': len(self.all_landmarks),
            'num_features': 42,
            'label_counts': self.label_counts,
            'captures_per_label': self.captures_per_label,
            'samples_per_capture': self.samples_per_capture,
            'collection_date': timestamp,
            'unique_labels': list(np.unique(labels_array))
        }
        
        metadata_file = os.path.join(self.output_dir, f"metadata_{timestamp}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Also save labels.txt for the recognition system
        labels_txt_file = "labels.txt"
        unique_labels = sorted(np.unique(labels_array))
        with open(labels_txt_file, 'w') as f:
            for label in unique_labels:
                f.write(f"{label}\n")
        
        print(f"\n{'='*60}")
        print(f"Dataset saved successfully!")
        print(f"{'='*60}")
        print(f"Landmarks: {landmarks_file}")
        print(f"Labels: {labels_file}")
        print(f"Metadata: {metadata_file}")
        print(f"Labels file: {labels_txt_file}")
        print(f"\nTotal samples: {len(self.all_landmarks)}")
        print(f"Captures and samples per label:")
        for label in sorted(self.captures_per_label.keys()):
            captures = self.captures_per_label[label]
            samples = self.label_counts[label]
            print(f"  {label}: {captures} captures = {samples} samples")
        print(f"{'='*60}")
        
    def process_frame(self, frame):
        """Process frame and display current state"""
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect hand landmarks
        detection_result = self.landmarker.detect(mp_image)
        
        # Status message
        status_lines = []
        hand_detected = False
        current_landmarks = None
        
        # Check if hand is detected
        if not detection_result.hand_landmarks or len(detection_result.hand_landmarks) == 0:
            status_lines.append("No hand detected")
            status_lines.append("Show your hand, then press a key")
        else:
            # Hand is present
            hand_detected = True
            hand_landmarks = detection_result.hand_landmarks[0]
            
            # Draw landmarks
            h, w, _ = frame.shape
            for landmark in hand_landmarks:
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            
            # Draw connections between landmarks
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
            
            # Extract landmarks
            current_landmarks = self.extract_landmarks(hand_landmarks)
            self.last_captured_landmarks = current_landmarks
            
            status_lines.append("✓ Hand detected - Ready!")
            status_lines.append("Press a key to capture this sign")
        
        # Draw UI
        self.draw_ui(frame, status_lines, hand_detected)
        
        return frame, current_landmarks
    
    def draw_ui(self, frame, status_lines, hand_detected):
        """Draw user interface on frame"""
        h, w, _ = frame.shape
        
        # Main status box
        box_height = 40 + (len(status_lines) * 40)
        cv2.rectangle(frame, (10, 10), (w - 10, box_height), (0, 0, 0), -1)
        
        y_offset = 50
        for line in status_lines:
            color = (0, 255, 0) if hand_detected else (0, 165, 255)
            cv2.putText(frame, line, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            y_offset += 40
        
        # Collection summary
        summary_y = h - 200
        cv2.rectangle(frame, (10, summary_y), (400, h - 10), (0, 0, 0), -1)
        
        cv2.putText(frame, "Collection Summary:", (20, summary_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        y = summary_y + 65
        if self.captures_per_label:
            for label in sorted(self.captures_per_label.keys())[:4]:  # Show first 4
                captures = self.captures_per_label[label]
                samples = self.label_counts[label]
                text = f"{label}: {captures}x = {samples} samples"
                cv2.putText(frame, text, (30, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                y += 30
            
            if len(self.captures_per_label) > 4:
                cv2.putText(frame, f"...and {len(self.captures_per_label) - 4} more", 
                           (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        else:
            cv2.putText(frame, "No captures yet", (30, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        
        # Total counter
        cv2.putText(frame, f"TOTAL: {len(self.all_landmarks)} samples", 
                   (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Controls
        controls_y = h - 200
        cv2.rectangle(frame, (w - 420, controls_y), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.putText(frame, "Controls:", (w - 410, controls_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "a-z, 0-4: Capture sign", (w - 400, controls_y + 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, "  (100 samples per capture)", (w - 400, controls_y + 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(frame, "SPACE: Capture current label", (w - 400, controls_y + 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, "s: Save dataset", (w - 400, controls_y + 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, "ESC: Quit", (w - 400, controls_y + 165),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    def run(self):
        """Main collection loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\n" + "="*60)
        print("ASL Dataset Collection Tool - Quick Capture Mode")
        print("="*60)
        print(f"Each capture generates {self.samples_per_capture} augmented samples")
        print("\nHow to use:")
        print("1. Make a sign with your hand")
        print("2. Press the corresponding key (a-z, 0-4)")
        print("3. Repeat for different variations (angles, positions)")
        print("4. Press 's' to save when done")
        print("\nAvailable signs:")
        for key, label in sorted(self.sign_keys.items()):
            print(f"  '{key}' -> {label}")
        print("\nRecommendation: Capture each sign 5-10 times from different")
        print("angles/positions for best results (500-1000 samples per sign)")
        print("="*60 + "\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            processed_frame, current_landmarks = self.process_frame(frame)
            
            cv2.imshow('ASL Dataset Collection - Quick Capture', processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Handle key presses
            if key == 27:  # ESC key
                print("\nQuitting...")
                break
            elif key == ord('s'):
                self.save_dataset()
            elif chr(key) in self.sign_keys and current_landmarks is not None:
                # Capture and augment
                label = self.sign_keys[chr(key)]
                self.capture_and_augment(current_landmarks, label)
                self.current_label = label
            elif key == 32 and self.current_label and current_landmarks is not None:  # SPACE
                # Capture another sample of the current label
                self.capture_and_augment(current_landmarks, self.current_label)
        
        cap.release()
        cv2.destroyAllWindows()
        self.landmarker.close()


def main():
    """Entry point"""
    # Check for hand landmarker file
    HAND_LANDMARKER_PATH = "hand_landmarker.task"
    
    if not os.path.exists(HAND_LANDMARKER_PATH):
        print("="*60)
        print("ERROR: hand_landmarker.task not found!")
        print("="*60)
        print("Please download it from:")
        print("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        print(f"\nPlace it in: {os.getcwd()}")
        print("="*60)
        return
    
    # Start collection
    collector = ASLDataCollector(HAND_LANDMARKER_PATH, output_dir="dataset")
    collector.run()


if __name__ == "__main__":
    main()