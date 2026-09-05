import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, auc

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.models.transfer import build_transfer_model
from src.models.baseline import build_baseline_cnn

def load_data_from_manifest(manifest_path):
    df = pd.read_csv(manifest_path)
    df = df[df['is_duplicate'] == False].copy()
    X = df['filepath'].values
    y = df['label'].values
    return X, y

def parse_image(filename, label):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.resize_with_pad(image, 224, 224)
    return image, label

def get_dataset(X, y, batch_size=16, is_training=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if is_training:
        ds = ds.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)
    ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def find_best_threshold(y_true, y_prob):
    best_f1 = -1
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.95, 0.05):
        y_pred = (y_prob >= thresh).astype(int)
        score = f1_score(y_true, y_pred)
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh
    return best_thresh

def evaluate_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_curve, precision_curve)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    else:
        sens = spec = prec = 0
        
    return {
        'roc_auc': roc_auc, 'pr_auc': pr_auc, 'accuracy': acc, 'bal_acc': bal_acc,
        'sensitivity': sens, 'specificity': spec, 'precision': prec, 'f1': f1, 'threshold': threshold
    }

def run_nested_kfold():
    manifest_path = os.path.join(os.path.dirname(__file__), '../data/manifest.csv')
    X, y = load_data_from_manifest(manifest_path)
    
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models_to_evaluate = ['baseline', 'resnet50', 'efficientnetb0', 'convnexttiny']
    
    all_results = []
    
    for model_name in models_to_evaluate:
        print(f"\n{'='*60}\nEvaluating {model_name} with Nested 5-Fold CV\n{'='*60}")
        fold_metrics = []
        
        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            print(f"\n--- {model_name} Outer Fold {fold+1}/5 ---")
            
            X_outer_train, y_outer_train = X[train_idx], y[train_idx]
            X_outer_test, y_outer_test = X[test_idx], y[test_idx]
            
            # Inner split for early stopping and threshold tuning
            # 25% of outer train (80% total) = 20% of total data for inner val
            X_inner_train, X_inner_val, y_inner_train, y_inner_val = train_test_split(
                X_outer_train, y_outer_train, test_size=0.25, stratify=y_outer_train, random_state=fold
            )
            
            inner_train_ds = get_dataset(X_inner_train, y_inner_train, is_training=True)
            inner_val_ds = get_dataset(X_inner_val, y_inner_val, is_training=False)
            outer_test_ds = get_dataset(X_outer_test, y_outer_test, is_training=False)
            
            if model_name == 'baseline':
                model = build_baseline_cnn(input_shape=(224, 224, 3), norm_type='layernorm')
            else:
                try:
                    model = build_transfer_model(model_name, input_shape=(224, 224, 3), freeze_base=True)
                except Exception as e:
                    print(f"Failed to build {model_name}: {e}")
                    continue
                    
            # 1. Initial Training (train head, or full baseline)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                          loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
                          
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            model.fit(inner_train_ds, validation_data=inner_val_ds, epochs=15, callbacks=[early_stop], verbose=1)
            
            # 2. Fine-tuning (only for transfer models)
            if model_name != 'baseline':
                print(f"Fine-tuning {model_name}...")
                # Unfreeze all layers
                model.trainable = True
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                              loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
                early_stop_ft = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
                model.fit(inner_train_ds, validation_data=inner_val_ds, epochs=10, callbacks=[early_stop_ft], verbose=1)
                
            # 3. Threshold Tuning on INNER VAL
            inner_val_preds = []
            inner_val_true = []
            for images, labels in inner_val_ds:
                preds = model.predict(images, verbose=0)
                inner_val_preds.extend(preds.flatten())
                inner_val_true.extend(labels.numpy().flatten())
                
            best_thresh = find_best_threshold(np.array(inner_val_true), np.array(inner_val_preds))
            print(f"Tuned Threshold on Inner Val: {best_thresh:.2f}")
            
            # 4. Final Evaluation on OUTER TEST (Untouched)
            outer_test_preds = []
            outer_test_true = []
            for images, labels in outer_test_ds:
                preds = model.predict(images, verbose=0)
                outer_test_preds.extend(preds.flatten())
                outer_test_true.extend(labels.numpy().flatten())
                
            metrics = evaluate_metrics(np.array(outer_test_true), np.array(outer_test_preds), best_thresh)
            print(f"Outer Test ROC-AUC: {metrics['roc_auc']:.4f}, Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
            
            metrics['model'] = model_name
            metrics['fold'] = fold + 1
            fold_metrics.append(metrics)
            tf.keras.backend.clear_session()
            
        if fold_metrics:
            df_fold = pd.DataFrame(fold_metrics)
            avg_metrics = df_fold.mean().to_dict()
            std_metrics = df_fold.std().to_dict()
            
            print(f"\n{model_name} Nested CV Average:")
            print(f"ROC-AUC: {avg_metrics['roc_auc']:.4f} (+/- {std_metrics['roc_auc']:.4f})")
            print(f"Accuracy: {avg_metrics['accuracy']:.4f} (+/- {std_metrics['accuracy']:.4f})")
            print(f"F1 Score: {avg_metrics['f1']:.4f} (+/- {std_metrics['f1']:.4f})")
            
            all_results.extend(fold_metrics)
            
    os.makedirs('reports', exist_ok=True)
    df_results = pd.DataFrame(all_results)
    df_results.to_csv('reports/nested_kfold_results.csv', index=False)
    
    # Generate final summary table
    summary = df_results.groupby('model').mean()[['roc_auc', 'pr_auc', 'accuracy', 'sensitivity', 'specificity', 'f1']].round(4)
    print("\nFINAL NESTED CV COMPARISON TABLE:")
    print(summary)
    summary.to_csv('reports/nested_kfold_summary.csv')

if __name__ == "__main__":
    run_nested_kfold()
