"""
Train ASL Model from Collected Dataset
WITH optional preprocessing for better accuracy
Automatically handles ANY number of classes (A-Z, 1-100, etc.)
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import os
import glob
import json
import pickle


def find_latest_dataset(dataset_dir="dataset"):
    """Find the most recent dataset files"""
    landmark_files = glob.glob(os.path.join(dataset_dir, "landmarks_*.npy"))
    
    if not landmark_files:
        raise FileNotFoundError(f"No dataset files found in {dataset_dir}/")
    
    latest_landmarks = max(landmark_files, key=os.path.getctime)
    timestamp = latest_landmarks.split("landmarks_")[1].split(".npy")[0]
    latest_labels = os.path.join(dataset_dir, f"labels_{timestamp}.npy")
    
    print(f"Found dataset:")
    print(f"  Landmarks: {latest_landmarks}")
    print(f"  Labels: {latest_labels}")
    
    return latest_landmarks, latest_labels


def preprocess_landmarks(landmarks, scaler=None, normalize=True, center=True):
    """
    Optional preprocessing for landmarks
    
    Args:
        landmarks: Array of shape (N, 42)
        scaler: Pre-fitted StandardScaler (for test data)
        normalize: Whether to apply standardization
        center: Whether to center landmarks around wrist
        
    Returns:
        Processed landmarks, scaler
    """
    processed = landmarks.copy()
    
    if center:
        # Center all landmarks relative to wrist (landmark 0)
        # This makes the model rotation/translation invariant
        for i in range(len(processed)):
            wrist_x = processed[i, 0]
            wrist_y = processed[i, 1]
            processed[i, ::2] -= wrist_x  # Center x coordinates
            processed[i, 1::2] -= wrist_y  # Center y coordinates
    
    if normalize:
        # Standardize features (zero mean, unit variance)
        if scaler is None:
            scaler = StandardScaler()
            processed = scaler.fit_transform(processed)
        else:
            processed = scaler.transform(processed)
    
    return processed, scaler


def train_model(landmarks_file, labels_file, output_model="model.keras", 
                use_preprocessing=True):
    """
    Train ASL recognition model
    
    Args:
        landmarks_file: Path to landmarks .npy file
        labels_file: Path to labels .npy file
        output_model: Output model filename
        use_preprocessing: Whether to apply preprocessing
    """
    
    print("\n" + "="*60)
    print("ASL Model Training")
    print("="*60)
    print(f"Preprocessing: {'ENABLED' if use_preprocessing else 'DISABLED'}")
    
    # Load data
    print("\nLoading data...")
    landmarks = np.load(landmarks_file)
    labels = np.load(labels_file)
    
    print(f"✓ Loaded {len(landmarks)} samples")
    print(f"✓ Feature shape: {landmarks.shape}")
    print(f"✓ Unique labels: {np.unique(labels)}")
    
    # Count samples per label
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nDataset contains {len(unique)} classes")
    print("Samples per label:")
    for label, count in zip(unique, counts):
        print(f"  {label}: {count}")
    
    # Check for class imbalance
    min_samples = counts.min()
    max_samples = counts.max()
    if max_samples / min_samples > 2:
        print(f"\n⚠ Warning: Class imbalance detected!")
        print(f"  Min samples: {min_samples}, Max samples: {max_samples}")
        print(f"  Consider collecting more data for underrepresented classes")
    
    # Encode labels
    print("\nEncoding labels...")
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    
    print(f"✓ Number of classes: {num_classes}")
    print(f"✓ Class mapping:")
    for idx, label in enumerate(label_encoder.classes_):
        print(f"  {idx}: {label}")
    
    # Save label names
    labels_txt = "labels.txt"
    with open(labels_txt, 'w') as f:
        for label in label_encoder.classes_:
            f.write(f"{label}\n")
    print(f"✓ Saved label mapping to: {labels_txt}")
    
    # Split data BEFORE preprocessing to avoid data leakage
    print("\nSplitting data...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        landmarks, encoded_labels, test_size=0.15, random_state=42, stratify=encoded_labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.15, random_state=42, stratify=y_train_val
    )
    
    print(f"✓ Train set: {len(X_train)} samples")
    print(f"✓ Validation set: {len(X_val)} samples")
    print(f"✓ Test set: {len(X_test)} samples")
    
    # Preprocessing
    scaler = None
    if use_preprocessing:
        print("\nApplying preprocessing...")
        X_train, scaler = preprocess_landmarks(X_train, normalize=True, center=True)
        X_val, _ = preprocess_landmarks(X_val, scaler=scaler, normalize=True, center=True)
        X_test, _ = preprocess_landmarks(X_test, scaler=scaler, normalize=True, center=True)
        
        # Save scaler for inference
        scaler_file = "scaler.pkl"
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"✓ Saved scaler to: {scaler_file}")
        print(f"✓ Landmarks centered and normalized")
    
    # Build model - ADJUSTED for more classes
    print("\nBuilding model...")
    
    # For 26 classes (A-Z), use a deeper/wider network
    if num_classes > 10:
        print(f"✓ Using larger architecture for {num_classes} classes")
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(42,)),
            
            # First hidden layer - larger for more classes
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            
            # Second hidden layer
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            
            # Third hidden layer
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            # Fourth hidden layer
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            # Output layer
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
    else:
        # Smaller model for fewer classes
        print(f"✓ Using standard architecture for {num_classes} classes")
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(42,)),
            
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✓ Model built")
    model.summary()
    
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,  # Increased patience for more classes
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=0.00001,
        verbose=1
    )
    
    # Model checkpoint to save best model
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_model_checkpoint.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Train model
    print("\nTraining model...")
    print("="*60)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,  # More epochs for more classes
        batch_size=32,
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1
    )
    
    print("="*60)
    print("✓ Training complete!")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"✓ Test Loss: {test_loss:.4f}")
    print(f"✓ Test Accuracy: {test_accuracy*100:.2f}%")
    
    # Per-class accuracy
    print("\nPer-class accuracy:")
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    class_accuracies = []
    for i, label in enumerate(label_encoder.classes_):
        mask = y_test == i
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred_classes[mask] == y_test[mask])
            class_accuracies.append(class_acc)
            print(f"  {label}: {class_acc*100:.1f}% ({np.sum(mask)} samples)")
    
    # Show worst performing classes
    if len(class_accuracies) > 5:
        print("\nLowest accuracy classes:")
        sorted_indices = np.argsort(class_accuracies)[:5]
        for idx in sorted_indices:
            label = label_encoder.classes_[idx]
            acc = class_accuracies[idx]
            print(f"  {label}: {acc*100:.1f}% - consider collecting more data")
    
    # Confusion matrix for classes with low accuracy
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    
    # Save model
    print(f"\nSaving model to: {output_model}")
    model.save(output_model)
    print("✓ Model saved successfully!")
    
    # Save config
    config = {
        "preprocessing_enabled": use_preprocessing,
        "num_classes": num_classes,
        "classes": label_encoder.classes_.tolist(),
        "test_accuracy": float(test_accuracy),
        "class_accuracies": {label: float(acc) for label, acc in zip(label_encoder.classes_, class_accuracies)}
    }
    config_file = "model_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Config saved to: {config_file}")
    
    # Plot training history
    print("\nGenerating training plots...")
    plt.figure(figsize=(15, 5))
    
    # Accuracy plot
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'Model Accuracy ({num_classes} classes)')
    plt.grid(True)
    
    # Loss plot
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Model Loss')
    plt.grid(True)
    
    # Per-class accuracy bar chart
    plt.subplot(1, 3, 3)
    plt.bar(range(len(class_accuracies)), [acc*100 for acc in class_accuracies])
    plt.xlabel('Class Index')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-Class Test Accuracy')
    plt.axhline(y=test_accuracy*100, color='r', linestyle='--', label=f'Overall: {test_accuracy*100:.1f}%')
    plt.legend()
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plot_file = "training_history.png"
    plt.savefig(plot_file, dpi=150)
    print(f"✓ Training plot saved to: {plot_file}")
    
    # Save confusion matrix visualization for reference
    if num_classes <= 26:  # Only for reasonable number of classes
        plt.figure(figsize=(12, 10))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title(f'Confusion Matrix ({num_classes} classes)')
        plt.colorbar()
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, label_encoder.classes_, rotation=45)
        plt.yticks(tick_marks, label_encoder.classes_)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        confusion_file = "confusion_matrix.png"
        plt.savefig(confusion_file, dpi=150)
        print(f"✓ Confusion matrix saved to: {confusion_file}")
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Model: {output_model}")
    print(f"Labels: {labels_txt}")
    print(f"Config: {config_file}")
    if use_preprocessing:
        print(f"Scaler: scaler.pkl")
    print(f"Classes: {num_classes} ({', '.join(label_encoder.classes_)})")
    print(f"Total samples: {len(landmarks)}")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")
    print(f"Preprocessing: {'YES' if use_preprocessing else 'NO'}")
    
    # Recommendations
    if test_accuracy < 0.90:
        print("\n⚠ RECOMMENDATIONS:")
        print("  - Accuracy below 90%. Consider:")
        print("    • Collecting more data per class (aim for 1000+ samples)")
        print("    • Capturing more variations (angles, positions, lighting)")
        print("    • Checking for mislabeled data")
    elif test_accuracy < 0.95:
        print("\n✓ Good accuracy! To improve further:")
        print("  - Collect more diverse samples")
        print("  - Focus on classes with low accuracy")
    else:
        print("\n✓ Excellent accuracy!")
    
    print("="*60)


def main():
    """Entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ASL recognition model')
    parser.add_argument('--no-preprocessing', action='store_true',
                       help='Disable preprocessing (centering & normalization)')
    
    args = parser.parse_args()
    
    try:
        # Find latest dataset
        landmarks_file, labels_file = find_latest_dataset()
        
        # Train model
        train_model(
            landmarks_file, 
            labels_file, 
            output_model="model.keras",
            use_preprocessing=not args.no_preprocessing
        )
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()