import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, auc

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.models.transfer import build_transfer_model

def load_data_from_manifest(manifest_path):
    df = pd.read_csv(manifest_path)
    # 1. ENFORCE strict removal of duplicates before any splitting
    df = df[df['is_duplicate'] == False].copy()
    
    X = df['filepath'].values
    y = df['label'].values
    return X, y

def parse_image(filename, label):
    # Read and decode image
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    
    # We use resize_with_pad to PRESERVE aspect ratio!
    # This prevents the geometric distortion we noted in Step 3.
    image = tf.image.resize_with_pad(image, 224, 224)
    
    # The image is currently in [0, 255] float32
    return image, label

def get_dataset(X, y, batch_size=16, is_training=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    
    if is_training:
        ds = ds.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)
        
    ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def run_kfold():
    manifest_path = os.path.join(os.path.dirname(__file__), '../data/manifest.csv')
    X, y = load_data_from_manifest(manifest_path)
    
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    models_to_evaluate = ['efficientnetb0', 'convnexttiny', 'resnet50']
    
    results = []
    
    for model_name in models_to_evaluate:
        print(f"\n{'='*50}\nEvaluating {model_name} with {n_splits}-Fold CV\n{'='*50}")
        
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\n--- {model_name} Fold {fold+1}/{n_splits} ---")
            
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            train_ds = get_dataset(X_train, y_train, is_training=True)
            val_ds = get_dataset(X_val, y_val, is_training=False)
            
            # Build new model for each fold
            # Note: For ConvNeXtTiny we might need to handle the fact that it was added recently,
            # but we checked and it is available in this Keras version.
            try:
                model = build_transfer_model(model_name, input_shape=(224, 224, 3), freeze_base=True)
            except Exception as e:
                print(f"Failed to build {model_name}: {e}")
                continue
                
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), # Higher LR for frozen base
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=['accuracy']
            )
            
            # Train for 10 epochs
            model.fit(train_ds, validation_data=val_ds, epochs=10, verbose=1)
            
            # Predict
            y_pred_prob = []
            y_true_fold = []
            for images, labels in val_ds:
                preds = model.predict(images, verbose=0)
                y_pred_prob.extend(preds.flatten())
                y_true_fold.extend(labels.numpy().flatten())
                
            y_pred_prob = np.array(y_pred_prob)
            y_true_fold = np.array(y_true_fold)
            
            y_pred_class = (y_pred_prob >= 0.5).astype(int)
            
            acc = accuracy_score(y_true_fold, y_pred_class)
            roc_auc = roc_auc_score(y_true_fold, y_pred_prob)
            f1 = f1_score(y_true_fold, y_pred_class)
            
            cm = confusion_matrix(y_true_fold, y_pred_class, labels=[0, 1])
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            else:
                sens = spec = 0
                
            print(f"Fold {fold+1} ROC-AUC: {roc_auc:.4f}, Acc: {acc:.4f}, Sens: {sens:.4f}, Spec: {spec:.4f}")
            
            fold_metrics.append({
                'model': model_name,
                'fold': fold + 1,
                'roc_auc': roc_auc,
                'accuracy': acc,
                'f1': f1,
                'sensitivity': sens,
                'specificity': spec
            })
            
            # Clear memory
            tf.keras.backend.clear_session()
            
        if not fold_metrics:
            continue
            
        # Calculate averages for the model
        df_fold = pd.DataFrame(fold_metrics)
        avg_metrics = df_fold.mean().to_dict()
        std_metrics = df_fold.std().to_dict()
        
        print(f"\n{model_name} Average Metrics:")
        print(f"ROC-AUC: {avg_metrics['roc_auc']:.4f} (+/- {std_metrics['roc_auc']:.4f})")
        print(f"Accuracy: {avg_metrics['accuracy']:.4f} (+/- {std_metrics['accuracy']:.4f})")
        print(f"Sensitivity: {avg_metrics['sensitivity']:.4f} (+/- {std_metrics['sensitivity']:.4f})")
        print(f"Specificity: {avg_metrics['specificity']:.4f} (+/- {std_metrics['specificity']:.4f})")
        
        results.extend(fold_metrics)
        
    os.makedirs('reports', exist_ok=True)
    df_results = pd.DataFrame(results)
    df_results.to_csv('reports/kfold_results.csv', index=False)
    print("\nResults saved to reports/kfold_results.csv")

if __name__ == "__main__":
    run_kfold()
