import os
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.data.dataset import get_datasets
from src.models.baseline import build_baseline_cnn

def train_model(data_dir, epochs=50, batch_size=32):
    print("Loading datasets...")
    train_ds, val_ds = get_datasets(data_dir=data_dir, batch_size=batch_size)
    
    print("Building model...")
    model = build_baseline_cnn(input_shape=(224, 224, 3))
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    model.summary()
    
    # Setup Callbacks
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "best_baseline.keras"),
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        TensorBoard(log_dir="logs")
    ]
    
    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    
    print("Training complete. Best weights restored from EarlyStopping.")
    
    # Final evaluation on validation set
    val_loss, val_acc, val_auc = model.evaluate(val_ds, verbose=0)
    print("\n--- Final Validation Metrics (Optimistic due to patient-leakage) ---")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")
    
    return history, val_loss, val_acc, val_auc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline CNN for Leukemia Detection")
    parser.add_argument("--data_dir", type=str, default="All-IDB dataset", help="Path to the dataset directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    
    args = parser.parse_args()
    
    train_model(args.data_dir, epochs=args.epochs, batch_size=args.batch_size)
