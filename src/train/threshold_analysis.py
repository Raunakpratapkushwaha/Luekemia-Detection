import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.data.dataset import get_datasets

def analyze_thresholds(data_dir, checkpoint_path, output_dir):
    print("Loading Validation Dataset...")
    _, val_ds = get_datasets(data_dir=data_dir, batch_size=32)
    
    print(f"Loading Model from {checkpoint_path}...")
    model = tf.keras.models.load_model(checkpoint_path)
    
    y_true = []
    y_pred_prob = []
    
    print("Generating Predictions...")
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().flatten())
        y_pred_prob.extend(preds.flatten())
        
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)
    
    thresholds = np.arange(0.05, 1.0, 0.05)
    results = []
    
    for t in thresholds:
        y_pred_class = (y_pred_prob >= t).astype(int)
        
        acc = accuracy_score(y_true, y_pred_class)
        bal_acc = balanced_accuracy_score(y_true, y_pred_class)
        f1 = f1_score(y_true, y_pred_class)
        
        cm = confusion_matrix(y_true, y_pred_class, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = sens
        else:
            sens = spec = prec = rec = 0
            
        results.append({
            'Threshold': t,
            'Accuracy': acc,
            'Balanced_Accuracy': bal_acc,
            'Sensitivity': sens,
            'Specificity': spec,
            'Precision': prec,
            'Recall': rec,
            'F1': f1
        })
        
    df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'threshold_analysis.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved threshold analysis to {csv_path}")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df['Threshold'], df['Accuracy'], label='Accuracy', marker='o')
    plt.plot(df['Threshold'], df['Sensitivity'], label='Sensitivity (Recall)', marker='s')
    plt.plot(df['Threshold'], df['Specificity'], label='Specificity', marker='^')
    plt.plot(df['Threshold'], df['F1'], label='F1 Score', marker='x')
    plt.title('Performance Metrics vs. Probability Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(output_dir, 'threshold_analysis.png')
    plt.savefig(plot_path)
    print(f"Saved threshold plot to {plot_path}")

if __name__ == "__main__":
    data_dir = "All-IDB dataset"
    checkpoint_path = "checkpoints/best_baseline.keras"
    output_dir = "reports"
    analyze_thresholds(data_dir, checkpoint_path, output_dir)
