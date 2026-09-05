import os
import sys
import numpy as np
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.data.dataset import get_datasets

def audit_pipeline(data_dir):
    print("--- Starting Pipeline Verification ---")
    train_ds, val_ds = get_datasets(data_dir=data_dir, batch_size=4)
    
    train_batch, train_labels = next(iter(train_ds))
    val_batch, val_labels = next(iter(val_ds))
    
    print(f"Train Batch Shape: {train_batch.shape}, dtype: {train_batch.dtype}")
    print(f"Val Batch Shape: {val_batch.shape}, dtype: {val_batch.dtype}")
    
    print(f"Train Labels Shape: {train_labels.shape}, dtype: {train_labels.dtype}")
    print(f"Val Labels Shape: {val_labels.shape}, dtype: {val_labels.dtype}")
    
    print("\n--- Normalization Audit ---")
    print(f"Train Batch Min/Max: {tf.reduce_min(train_batch):.2f} / {tf.reduce_max(train_batch):.2f}")
    print(f"Val Batch Min/Max: {tf.reduce_min(val_batch):.2f} / {tf.reduce_max(val_batch):.2f}")
    if tf.reduce_max(train_batch) > 1.0:
        print("Note: Images are NOT pre-normalized in the dataset pipeline. They range [0, 255].")
        print("Normalization MUST be applied in the model (e.g. Rescaling layer).")
    
    print("\n--- Shuffling Audit ---")
    # Take two subsequent batches from train to ensure they differ
    train_batch_2, _ = next(iter(train_ds))
    if not tf.reduce_all(train_batch == train_batch_2):
        print("Train Dataset is properly shuffled (subsequent batches differ).")
    else:
        print("WARNING: Train Dataset batches are identical! Shuffling failed.")
        
    print("\n--- Model-Level Augmentation Audit ---")
    from src.models.baseline import build_baseline_cnn
    model = build_baseline_cnn()
    
    # Run in training mode
    train_mode_output = model(train_batch, training=True)
    # Run in inference mode
    infer_mode_output = model(train_batch, training=False)
    
    if not tf.reduce_all(train_mode_output == infer_mode_output):
        print("Model dropout/augmentation behaves differently in train vs inference. Correct.")
    else:
        print("WARNING: Model outputs are identical in train and inference mode!")

if __name__ == "__main__":
    audit_pipeline("All-IDB dataset")
