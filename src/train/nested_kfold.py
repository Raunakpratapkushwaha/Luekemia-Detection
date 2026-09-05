import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, auc

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.models.transfer import build_transfer_model
from src.models.baseline import build_baseline_cnn

def load_data_from_manifest(manifest_path):
    df = pd.read_csv(manifest_path)
    
    # 5. The known duplicate was excluded through the manifest.
    original_len = len(df)
    df = df[df['is_duplicate'] == False].copy()
    assert len(df) == original_len - 1, "Duplicate exclusion failed or multiple duplicates found!"
    
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
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh
    return best_thresh

def evaluate_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except:
        roc_auc = 0
    
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_curve, precision_curve)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = sens
    else:
        sens = spec = prec = rec = 0
        tn = fp = fn = tp = 0
        
    return {
        'roc_auc': roc_auc, 'pr_auc': pr_auc, 'accuracy': acc, 'bal_acc': bal_acc,
        'sensitivity': sens, 'specificity': spec, 'precision': prec, 'recall': rec, 'f1': f1, 'threshold': threshold,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'min_prob': np.min(y_prob), 'max_prob': np.max(y_prob), 
        'mean_prob': np.mean(y_prob), 'std_prob': np.std(y_prob),
        'mean_prob_class0': np.mean(y_prob[y_true == 0]) if np.any(y_true == 0) else 0,
        'mean_prob_class1': np.mean(y_prob[y_true == 1]) if np.any(y_true == 1) else 0
    }

def get_predictions(model, ds):
    preds, true_labels = [], []
    for images, labels in ds:
        p = model.predict(images, verbose=0)
        preds.extend(p.flatten())
        true_labels.extend(labels.numpy().flatten())
    return np.array(true_labels), np.array(preds)

def run_nested_kfold():
    manifest_path = os.path.join(os.path.dirname(__file__), '../data/manifest.csv')
    X, y = load_data_from_manifest(manifest_path)
    
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models_to_evaluate = ['baseline', 'resnet50', 'efficientnetb0', 'convnexttiny']
    
    all_results = []
    histories = {}
    
    for model_name in models_to_evaluate:
        print(f"\n{'='*60}\nEvaluating {model_name} with Nested 5-Fold CV\n{'='*60}")
        histories[model_name] = {}
        
        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            print(f"\n--- {model_name} Outer Fold {fold+1}/5 ---")
            
            X_outer_train, y_outer_train = X[train_idx], y[train_idx]
            X_outer_test, y_outer_test = X[test_idx], y[test_idx]
            
            # Inner split
            X_inner_train, X_inner_val, y_inner_train, y_inner_val = train_test_split(
                X_outer_train, y_outer_train, test_size=0.25, stratify=y_outer_train, random_state=fold
            )
            
            # --- STRICT PROGRAMMATIC AUDIT ---
            # 1. No outer-test image used for training
            assert len(set(X_outer_test).intersection(set(X_inner_train))) == 0
            # 2. No outer-test image used for early stopping
            assert len(set(X_outer_test).intersection(set(X_inner_val))) == 0
            
            print(f"Sample Counts | Outer Train: {len(X_outer_train)} | Inner Train: {len(X_inner_train)} | Inner Val: {len(X_inner_val)} | Outer Test: {len(X_outer_test)}")
            
            inner_train_ds = get_dataset(X_inner_train, y_inner_train, is_training=True)
            inner_val_ds = get_dataset(X_inner_val, y_inner_val, is_training=False)
            outer_test_ds = get_dataset(X_outer_test, y_outer_test, is_training=False)
            
            if model_name == 'baseline':
                model = build_baseline_cnn(input_shape=(224, 224, 3), norm_type='layernorm')
            else:
                model = build_transfer_model(model_name, input_shape=(224, 224, 3), freeze_base=True)
                    
            # --- PHASE 1: HEAD-ONLY TRAINING ---
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                          loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            
            print(f"Training Head (or baseline) for {model_name}...")
            history_head = model.fit(inner_train_ds, validation_data=inner_val_ds, epochs=15, callbacks=[early_stop], verbose=1)
            
            # Evaluate HEAD-ONLY
            inner_val_true, inner_val_preds = get_predictions(model, inner_val_ds)
            # 3. No outer test used for threshold selection (done on inner val)
            best_thresh_head = find_best_threshold(inner_val_true, inner_val_preds)
            outer_test_true, outer_test_preds = get_predictions(model, outer_test_ds)
            
            metrics_head = evaluate_metrics(outer_test_true, outer_test_preds, best_thresh_head)
            metrics_head.update({'model': model_name, 'fold': fold + 1, 'phase': 'HEAD_ONLY'})
            all_results.append(metrics_head)
            print(f"HEAD-ONLY -> ROC-AUC: {metrics_head['roc_auc']:.4f}, Acc: {metrics_head['accuracy']:.4f}, F1: {metrics_head['f1']:.4f}")
            
            # --- PHASE 2: FULL FINE-TUNING (Only for transfer models) ---
            if model_name != 'baseline':
                print(f"Fine-tuning {model_name}...")
                model.trainable = True
                
                # 4. No outer test used to choose fine tuning params (early stopping uses inner val)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                              loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
                early_stop_ft = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
                history_ft = model.fit(inner_train_ds, validation_data=inner_val_ds, epochs=10, callbacks=[early_stop_ft], verbose=1)
                
                # Evaluate FINE-TUNED
                inner_val_true_ft, inner_val_preds_ft = get_predictions(model, inner_val_ds)
                best_thresh_ft = find_best_threshold(inner_val_true_ft, inner_val_preds_ft)
                
                outer_test_true_ft, outer_test_preds_ft = get_predictions(model, outer_test_ds)
                metrics_ft = evaluate_metrics(outer_test_true_ft, outer_test_preds_ft, best_thresh_ft)
                metrics_ft.update({'model': model_name, 'fold': fold + 1, 'phase': 'FINE_TUNED'})
                all_results.append(metrics_ft)
                print(f"FINE-TUNED -> ROC-AUC: {metrics_ft['roc_auc']:.4f}, Acc: {metrics_ft['accuracy']:.4f}, F1: {metrics_ft['f1']:.4f}")
                
                histories[model_name][fold+1] = {
                    'head': history_head.history,
                    'ft': history_ft.history
                }
            else:
                histories[model_name][fold+1] = {
                    'head': history_head.history
                }
                
            tf.keras.backend.clear_session()
            
    os.makedirs('reports', exist_ok=True)
    df_results = pd.DataFrame(all_results)
    df_results.to_csv('reports/nested_kfold_detailed_results.csv', index=False)
    
    with open('reports/training_histories.pkl', 'wb') as f:
        pickle.dump(histories, f)
        
    print("\nFINAL NESTED CV COMPARISON COMPLETED. Saved to reports/nested_kfold_detailed_results.csv")

if __name__ == "__main__":
    run_nested_kfold()
