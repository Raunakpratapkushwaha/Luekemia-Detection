import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc, f1_score, accuracy_score, balanced_accuracy_score

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.data.dataset import get_datasets

def debug_model(data_dir, checkpoint_path):
    print("Loading Validation Dataset...")
    _, val_ds = get_datasets(data_dir=data_dir, batch_size=1)
    
    # We will use batch_size 1 to easily match paths to predictions
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, 'val'),
        labels='inferred',
        label_mode='binary',
        class_names=['nonblast', 'Blast'], # 0: nonblast, 1: Blast
        color_mode='rgb',
        batch_size=1,
        image_size=(224, 224),
        shuffle=False
    )
    
    file_paths = val_ds.file_paths
    
    print(f"\nLoading Model from {checkpoint_path}...")
    model = tf.keras.models.load_model(checkpoint_path)
    
    # Verify Sigmoid / Config
    print("\n--- Model Output Layer Inspection ---")
    output_layer = model.layers[-1]
    print(f"Output Layer Name: {output_layer.name}")
    print(f"Output Layer Activation: {output_layer.activation.__name__}")
    if output_layer.activation.__name__ != 'sigmoid':
        print("WARNING: Final layer is not using sigmoid!")
    
    y_true = []
    y_pred_prob = []
    
    print("\n--- Generating Predictions ---")
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.append(labels.numpy()[0][0])
        y_pred_prob.append(preds[0][0])
        
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)
    
    # Basic Distributions
    print("\n--- Prediction Distribution ---")
    print(f"Min probability: {y_pred_prob.min():.6f}")
    print(f"Max probability: {y_pred_prob.max():.6f}")
    print(f"Mean probability: {y_pred_prob.mean():.6f}")
    print(f"Std probability: {y_pred_prob.std():.6f}")
    
    prob_class_0 = y_pred_prob[y_true == 0]
    prob_class_1 = y_pred_prob[y_true == 1]
    
    print(f"\nMean probability for True Class 0 (nonblast): {prob_class_0.mean():.6f}")
    print(f"Mean probability for True Class 1 (Blast): {prob_class_1.mean():.6f}")
    
    # Metrics at threshold 0.5
    threshold = 0.5
    y_pred_class = (y_pred_prob >= threshold).astype(int)
    
    print(f"\n--- Metrics at Threshold {threshold} ---")
    cm = confusion_matrix(y_true, y_pred_class)
    print(f"Confusion Matrix:\n{cm}")
    # Extract TN, FP, FN, TP
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"True Negatives: {tn}, False Positives: {fp}")
        print(f"False Negatives: {fn}, True Positives: {tp}")
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        print(f"Sensitivity (Recall): {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"Precision: {precision:.4f}")
    
    acc = accuracy_score(y_true, y_pred_class)
    bal_acc = balanced_accuracy_score(y_true, y_pred_class)
    f1 = f1_score(y_true, y_pred_class)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # AUC metrics
    roc_auc = roc_auc_score(y_true, y_pred_prob)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_prob)
    pr_auc = auc(recall_curve, precision_curve)
    
    print(f"\n--- ROC & PR AUC ---")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    
    print("\n--- Explicit Examples ---")
    for i in range(10): # Print first 10
        print(f"File: {os.path.basename(file_paths[i])} | True Label: {y_true[i]} | Pred Prob: {y_pred_prob[i]:.6f} | Pred Class (0.5): {y_pred_class[i]}")

if __name__ == "__main__":
    data_dir = "All-IDB dataset"
    checkpoint_path = "checkpoints/best_baseline.keras"
    debug_model(data_dir, checkpoint_path)
