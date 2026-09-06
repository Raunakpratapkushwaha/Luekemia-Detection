import os
import argparse
import pickle
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, balanced_accuracy_score

def parse_image(filename, label, input_shape):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_bmp(image_string, channels=3)
    # The locked preprocessing from ALL-IDB2 is tf.image.resize_with_pad
    image = tf.image.resize_with_pad(image, input_shape[0], input_shape[1])
    return image, label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str, required=True, help="Path to cnmc_manifest.csv")
    parser.add_argument('--model_path', type=str, required=True, help="Path to efficientnetb0_headonly_final.keras")
    parser.add_argument('--config_path', type=str, required=True, help="Path to final_config.pkl")
    parser.add_argument('--out_json', type=str, default='cnmc_frozen_results.json')
    args = parser.parse_args()

    # 1. Load config
    with open(args.config_path, 'rb') as f:
        config = pickle.load(f)
    
    input_shape = config['input_shape']
    threshold = config['threshold']
    print(f"Loaded config: Input Shape {input_shape}, Threshold {threshold}")

    # 2. Load model
    print(f"Loading frozen model from {args.model_path}...")
    model = tf.keras.models.load_model(args.model_path)

    # 3. Load dataset
    print(f"Loading external manifest from {args.manifest}...")
    df = pd.read_csv(args.manifest)
    
    # Assert subject-level separation (just informative since this is a pure test set)
    print(f"Total External Test Images: {len(df)}")
    print(f"Total Unique Subjects: {df['subject_id'].nunique()}")

    X = df['filepath'].values
    y = df['label'].values

    # tf.data pipeline
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.map(lambda x, y: parse_image(x, y, input_shape), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)

    # 4. Predict
    print("Running inference on external dataset... This may take a while on C-NMC.")
    y_pred_probs = model.predict(ds).ravel()

    # 5. Metrics
    y_pred_class = (y_pred_probs >= threshold).astype(int)

    auc = roc_auc_score(y, y_pred_probs)
    pr_auc = average_precision_score(y, y_pred_probs)
    acc = accuracy_score(y, y_pred_class)
    sens = recall_score(y, y_pred_class)
    prec = precision_score(y, y_pred_class)
    f1 = f1_score(y, y_pred_class)
    bal_acc = balanced_accuracy_score(y, y_pred_class)
    cm = confusion_matrix(y, y_pred_class)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp)

    results = {
        'roc_auc': float(auc),
        'pr_auc': float(pr_auc),
        'accuracy': float(acc),
        'sensitivity': float(sens),
        'specificity': float(spec),
        'precision': float(prec),
        'recall': float(sens),
        'f1': float(f1),
        'balanced_accuracy': float(bal_acc),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    }

    print("\n--- Frozen External Validation Results on C-NMC ---")
    for k, v in results.items():
        print(f"{k}: {v}")

    with open(args.out_json, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved results to {args.out_json}")

if __name__ == '__main__':
    main()
