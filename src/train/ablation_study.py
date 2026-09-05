import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.data.dataset import get_datasets
from src.models.baseline import build_baseline_cnn

def train_and_eval(model_name, norm_type, bn_momentum, train_ds, val_ds, epochs=10):
    print(f"\n{'='*50}\nTraining {model_name} (Norm: {norm_type}, Mom: {bn_momentum})\n{'='*50}")
    model = build_baseline_cnn(norm_type=norm_type, bn_momentum=bn_momentum)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )
    
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=1)
    
    # Predict normally (training=False)
    y_true = []
    y_pred_prob = []
    
    # Also capture predictions with training=True for BN 0.99
    y_pred_prob_train_mode = []
    
    for images, labels in val_ds:
        preds = model(images, training=False).numpy()
        y_true.extend(labels.numpy().flatten())
        y_pred_prob.extend(preds.flatten())
        
        if model_name == "Model_A_BN_0.99":
            preds_t = model(images, training=True).numpy()
            y_pred_prob_train_mode.extend(preds_t.flatten())
            
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)
    
    results = {}
    
    # If Model A, do the training=True vs training=False comparison
    if model_name == "Model_A_BN_0.99":
        y_pred_prob_train_mode = np.array(y_pred_prob_train_mode)
        print("\n--- BN 0.99: training=False (Inference Mode) ---")
        print(f"Min: {y_pred_prob.min():.4f}, Max: {y_pred_prob.max():.4f}, Mean: {y_pred_prob.mean():.4f}, Std: {y_pred_prob.std():.4f}")
        print(f"Mean (Class 0): {y_pred_prob[y_true==0].mean():.4f}, Mean (Class 1): {y_pred_prob[y_true==1].mean():.4f}")
        
        print("\n--- BN 0.99: training=True (Training Mode) ---")
        print(f"Min: {y_pred_prob_train_mode.min():.4f}, Max: {y_pred_prob_train_mode.max():.4f}, Mean: {y_pred_prob_train_mode.mean():.4f}, Std: {y_pred_prob_train_mode.std():.4f}")
        print(f"Mean (Class 0): {y_pred_prob_train_mode[y_true==0].mean():.4f}, Mean (Class 1): {y_pred_prob_train_mode[y_true==1].mean():.4f}")
        
        results['y_pred_prob_train_mode'] = y_pred_prob_train_mode
        
    y_pred_class = (y_pred_prob >= 0.5).astype(int)
    
    acc = accuracy_score(y_true, y_pred_class)
    bal_acc = balanced_accuracy_score(y_true, y_pred_class)
    f1 = f1_score(y_true, y_pred_class)
    roc_auc = roc_auc_score(y_true, y_pred_prob)
    
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_prob)
    pr_auc = auc(recall_curve, precision_curve)
    
    cm = confusion_matrix(y_true, y_pred_class, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = sens
    else:
        sens = spec = prec = rec = 0
        tn = fp = fn = tp = 0
        
    print(f"\nMetrics for {model_name} at threshold 0.5:")
    print(f"ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}, Acc: {acc:.4f}, Bal-Acc: {bal_acc:.4f}")
    print(f"Sens: {sens:.4f}, Spec: {spec:.4f}, Prec: {prec:.4f}, F1: {f1:.4f}")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Prob Min/Max: {y_pred_prob.min():.4f} / {y_pred_prob.max():.4f}")
    print(f"Prob Mean (0): {y_pred_prob[y_true==0].mean():.4f}, Mean (1): {y_pred_prob[y_true==1].mean():.4f}")
    
    results.update({
        'model': model_name,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'acc': acc,
        'bal_acc': bal_acc,
        'sens': sens,
        'spec': spec,
        'prec': prec,
        'f1': f1,
        'y_true': y_true,
        'y_pred_prob': y_pred_prob,
        'min_prob': y_pred_prob.min(),
        'max_prob': y_pred_prob.max(),
        'mean_prob_0': y_pred_prob[y_true==0].mean(),
        'mean_prob_1': y_pred_prob[y_true==1].mean(),
        'cm': cm
    })
    return results

def run_ablation():
    print("Loading datasets...")
    train_ds, val_ds = get_datasets(data_dir="All-IDB dataset", batch_size=16)
    
    # We must explicitly NOT load the dataset multiple times to save time,
    # just reuse train_ds and val_ds
    
    res_A = train_and_eval("Model_A_BN_0.99", "batchnorm", 0.99, train_ds, val_ds)
    res_B = train_and_eval("Model_B_LayerNorm", "layernorm", None, train_ds, val_ds)
    res_C = train_and_eval("Model_C_BN_0.90", "batchnorm", 0.90, train_ds, val_ds)
    
    # Plotting distributions
    os.makedirs("reports", exist_ok=True)
    
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    def plot_dist(ax, y_true, y_prob, title):
        ax.hist(y_prob[y_true==0], bins=20, alpha=0.5, label='Class 0 (Nonblast)', color='blue')
        ax.hist(y_prob[y_true==1], bins=20, alpha=0.5, label='Class 1 (Blast)', color='red')
        ax.set_title(title)
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.set_xlim([0, 1])
        
    plot_dist(axs[0], res_A['y_true'], res_A['y_pred_prob_train_mode'], "Model A (BN 0.99) training=True")
    plot_dist(axs[1], res_A['y_true'], res_A['y_pred_prob'], "Model A (BN 0.99) training=False")
    plot_dist(axs[2], res_B['y_true'], res_B['y_pred_prob'], "Model B (LayerNorm) training=False")
    plot_dist(axs[3], res_C['y_true'], res_C['y_pred_prob'], "Model C (BN 0.90) training=False")
    
    plt.tight_layout()
    plt.savefig("reports/ablation_distributions.png")
    print("Saved plots to reports/ablation_distributions.png")

if __name__ == "__main__":
    run_ablation()
