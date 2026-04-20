import tensorflow as tf
import os
import numpy as np
from main import DeepfakeGenerator # Reusing your existing generator

def evaluate_model(model_path, data_dir):
    # 1. Load the saved model
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # 2. Prepare the Validation Generator
    # We use a batch_size of 1 for precise evaluation
    val_gen = DeepfakeGenerator(data_dir, split='val', batch_size=1)
    
    if len(val_gen.data) == 0:
        print("Validation data not found! Check your paths.")
        return

    print(f"Evaluating on {len(val_gen.data)} samples...")

    # 3. Run Evaluation
    results = model.evaluate(val_gen)
    
    print("\n--- Evaluation Results ---")
    print(f"Validation Loss: {results[0]:.4f}")
    print(f"Validation Accuracy: {results[1]*100:.2f}%")

    # 4. Optional: Manual Check (First 5 samples)
    print("\n--- Sample Predictions ---")
    for i in range(5):
        X, y_true = val_gen[i]
        y_pred = model.predict(X, verbose=0)
        label_true = "FAKE" if y_true[0] == 1 else "REAL"
        label_pred = "FAKE" if y_pred[0][0] > 0.5 else "REAL"
        confidence = y_pred[0][0] if label_pred == "FAKE" else 1 - y_pred[0][0]
        
        print(f"Sample {i+1}: Actual: {label_true} | Predicted: {label_pred} ({confidence*100:.2f}% confidence)")

if __name__ == "__main__":
    MODEL_FILE = "deepfake_detector_model.keras"
    PROCESSED_DIR = os.path.join(os.getcwd(), "processed_dataset")
    
    if os.path.exists(MODEL_FILE):
        evaluate_model(MODEL_FILE, PROCESSED_DIR)
    else:
        print(f"Model file {MODEL_FILE} not found. Did you save it?")