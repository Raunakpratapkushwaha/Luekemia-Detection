import os
import argparse
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, balanced_accuracy_score, matthews_corrcoef

def verify_gpu():
    print(f"TensorFlow Version: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("CRITICAL ERROR: No GPU detected.")
        print("This script must run on Google Colab with an NVIDIA T4 GPU.")
        print("ABORTING execution.")
        exit(1)
    
    for gpu in gpus:
        print(f"Detected GPU: {gpu}")
        try:
            details = tf.config.experimental.get_device_details(gpu)
            if 'device_name' in details:
                print(f"GPU Name: {details['device_name']}")
        except:
            pass

def parse_image(filename, label, input_shape):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_image(image_string, channels=3, expand_animations=False)
    image = tf.image.resize_with_pad(image, input_shape[0], input_shape[1])
    return image, label

def augment_image(image, label):
    # Conservative morphology-preserving augmentation
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    return image, label

def build_cnmc_model(input_shape=(224, 224, 3)):
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, weights='imagenet', input_shape=input_shape
    )
    # Head-only training first
    base_model.trainable = False
    
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    predictions = tf.keras.layers.Dense(1, activation='sigmoid', dtype=tf.float32)(x)
    
    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
    return model

def calculate_metrics(y_true, y_pred_probs, threshold=0.5):
    y_pred_class = (y_pred_probs >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_pred_probs)
    pr_auc = average_precision_score(y_true, y_pred_probs)
    acc = accuracy_score(y_true, y_pred_class)
    sens = recall_score(y_true, y_pred_class)
    prec = precision_score(y_true, y_pred_class, zero_division=0)
    f1 = f1_score(y_true, y_pred_class)
    bal_acc = balanced_accuracy_score(y_true, y_pred_class)
    mcc = matthews_corrcoef(y_true, y_pred_class)
    cm = confusion_matrix(y_true, y_pred_class)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'roc_auc': float(auc),
        'pr_auc': float(pr_auc),
        'accuracy': float(acc),
        'sensitivity': float(sens),
        'specificity': float(spec),
        'precision': float(prec),
        'recall': float(sens),
        'f1': float(f1),
        'balanced_accuracy': float(bal_acc),
        'mcc': float(mcc),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str, required=True, help="Path to cnmc_subset_manifest.csv")
    parser.add_argument('--checkpoints_dir', type=str, required=True, help="Path to save checkpoints")
    parser.add_argument('--reports_dir', type=str, required=True, help="Path to save reports")
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    print("Verifying Hardware...")
    verify_gpu()
    
    # Ensure dirs exist (usually on Google Drive)
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    os.makedirs(args.reports_dir, exist_ok=True)
    
    df = pd.read_csv(args.manifest)
    input_shape = (224, 224, 3)
    
    outer_experiments = [
        {'test_fold': 'fold_0', 'train_folds': ['fold_1', 'fold_2']},
        {'test_fold': 'fold_1', 'train_folds': ['fold_0', 'fold_2']},
        {'test_fold': 'fold_2', 'train_folds': ['fold_0', 'fold_1']}
    ]
    
    overall_results = {}
    
    for exp_idx, exp in enumerate(outer_experiments):
        test_fold = exp['test_fold']
        report_path = os.path.join(args.reports_dir, f'cnmc_exp_{test_fold}_results.json')
        
        if os.path.exists(report_path):
            print(f"\n[INFO] Experiment {exp_idx+1}/3: Test Fold {test_fold} already completed. Skipping to next fold (Resume Safety).")
            with open(report_path, 'r') as f:
                overall_results[test_fold] = json.load(f)
            continue
            
        print(f"\n=======================================================")
        print(f" EXPERIMENT {exp_idx+1}/3 | Test Fold: {test_fold}")
        print(f"=======================================================")
        
        df_test = df[df['official_fold'] == test_fold].copy()
        df_train_pool = df[df['official_fold'].isin(exp['train_folds'])].copy()
        
        # Subject-level split for validation (20% of subjects in train_pool)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(gss.split(df_train_pool, groups=df_train_pool['subject_id']))
        
        df_train = df_train_pool.iloc[train_idx].copy()
        df_val = df_train_pool.iloc[val_idx].copy()
        
        # VERIFY SUBJECT LEAKAGE
        train_subjects = set(df_train['subject_id'])
        val_subjects = set(df_val['subject_id'])
        test_subjects = set(df_test['subject_id'])
        
        assert len(train_subjects.intersection(val_subjects)) == 0, "Leakage Train/Val!"
        assert len(train_subjects.intersection(test_subjects)) == 0, "Leakage Train/Test!"
        assert len(val_subjects.intersection(test_subjects)) == 0, "Leakage Val/Test!"
        
        print(f"Train Images: {len(df_train)} (Subjects: {len(train_subjects)})")
        print(f"Val Images: {len(df_val)} (Subjects: {len(val_subjects)})")
        print(f"Test Images: {len(df_test)} (Subjects: {len(test_subjects)})")
        
        # Class weights
        all_count = len(df_train[df_train['label'] == 1])
        hem_count = len(df_train[df_train['label'] == 0])
        total = all_count + hem_count
        weight_for_0 = (1 / hem_count) * (total / 2.0)
        weight_for_1 = (1 / all_count) * (total / 2.0)
        class_weight = {0: weight_for_0, 1: weight_for_1}
        print(f"Class Weights: {class_weight}")
        
        # TF Data pipelines
        def make_ds(dataframe, augment=False, shuffle=False):
            X = dataframe['filepath'].values
            y = dataframe['label'].values
            ds = tf.data.Dataset.from_tensor_slices((X, y))
            if shuffle:
                ds = ds.shuffle(buffer_size=len(X), seed=42)
            ds = ds.map(lambda x, y: parse_image(x, y, input_shape), num_parallel_calls=tf.data.AUTOTUNE)
            if augment:
                ds = ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
            return ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
            
        ds_train = make_ds(df_train, augment=True, shuffle=True)
        ds_val = make_ds(df_val, augment=False, shuffle=False)
        ds_test = make_ds(df_test, augment=False, shuffle=False)
        
        # Build model
        model = build_cnmc_model(input_shape)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        ckpt_path = os.path.join(args.checkpoints_dir, f'efficientnetb0_{test_fold}.keras')
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_auc', mode='max', save_best_only=True)
        ]
        
        print("Starting training...")
        history = model.fit(
            ds_train,
            validation_data=ds_val,
            epochs=args.epochs,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nThreshold Selection on Validation Set...")
        val_probs = model.predict(ds_val).ravel()
        val_labels = df_val['label'].values
        
        best_threshold = 0.5
        best_f1 = 0
        for th in np.arange(0.1, 0.9, 0.05):
            preds = (val_probs >= th).astype(int)
            score = f1_score(val_labels, preds)
            if score > best_f1:
                best_f1 = score
                best_threshold = th
                
        print(f"Selected Threshold: {best_threshold:.3f}")
        
        print("\nEvaluating on Held-Out Test Fold (Zero leakage)...")
        test_probs = model.predict(ds_test).ravel()
        test_labels = df_test['label'].values
        
        metrics = calculate_metrics(test_labels, test_probs, threshold=best_threshold)
        metrics['selected_threshold'] = best_threshold
        
        print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        
        with open(report_path, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        overall_results[test_fold] = metrics
        
    print("\n=======================================================")
    print(" ALL OUTER FOLDS COMPLETED ")
    print("=======================================================")
    
    # Aggregate
    agg = {}
    for metric in ['roc_auc', 'pr_auc', 'accuracy', 'sensitivity', 'specificity', 'f1', 'balanced_accuracy', 'mcc']:
        vals = [overall_results[f][metric] for f in ['fold_0', 'fold_1', 'fold_2']]
        agg[f"{metric}_mean"] = float(np.mean(vals))
        agg[f"{metric}_std"] = float(np.std(vals))
        print(f"{metric}: {agg[f'{metric}_mean']:.4f} ± {agg[f'{metric}_std']:.4f}")
        
    final_report_path = os.path.join(args.reports_dir, 'cnmc_final_aggregation.json')
    with open(final_report_path, 'w') as f:
        json.dump({'folds': overall_results, 'aggregated': agg}, f, indent=4)
    print(f"Saved aggregated report to {final_report_path}")

if __name__ == '__main__':
    main()
