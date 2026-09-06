import os
import argparse
import pickle
import json
import numpy as np
import pandas as pd
import tensorflow as tf
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
    # The locked preprocessing from ALL-IDB2 is tf.image.resize_with_pad
    image = tf.image.resize_with_pad(image, input_shape[0], input_shape[1])
    return image, label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str, required=True, help="Path to cnmc_subset_manifest.csv")
    parser.add_argument('--model_path', type=str, required=True, help="Path to locked efficientnetb0_headonly_final.keras")
    parser.add_argument('--config_path', type=str, required=True, help="Path to locked final_config.pkl")
    parser.add_argument('--out_json', type=str, default='cnmc_subset_zero_shot.json')
    args = parser.parse_args()

    print("Verifying Hardware...")
    verify_gpu()

    # 1. Load config
    with open(args.config_path, 'rb') as f:
        config = pickle.load(f)
    
    input_shape = config['input_shape']
    threshold = config['threshold']
    print(f"Loaded ALL-IDB2 config:")
    print(f" - Input Shape: {input_shape}")
    print(f" - Locked Threshold: {threshold}")
    print(f" - Locked Preprocessing: {config.get('preprocessing', 'unknown')}")

    # 2. Load model
    print(f"\nLoading frozen model from {args.model_path}...")
    model = tf.keras.models.load_model(args.model_path)

    # 3. Load dataset
    print(f"Loading external manifest from {args.manifest}...")
    df = pd.read_csv(args.manifest)
    
    print(f"Total External Test Images: {len(df)}")
    print(f"Total Unique Subjects: {df['subject_id'].nunique()}")

    X = df['filepath'].values
    y = df['label'].values

    # tf.data pipeline
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.map(lambda x, y: parse_image(x, y, input_shape), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(64).prefetch(tf.data.AUTOTUNE)

    # 4. Predict
    print("\nRunning ZERO-SHOT external inference on C-NMC... (DO NOT RETUNE)")
    y_pred_probs = model.predict(ds).ravel()

    # 5. Metrics
    y_pred_class = (y_pred_probs >= threshold).astype(int)

    auc = roc_auc_score(y, y_pred_probs)
    pr_auc = average_precision_score(y, y_pred_probs)
    acc = accuracy_score(y, y_pred_class)
    sens = recall_score(y, y_pred_class)
    prec = precision_score(y, y_pred_class, zero_division=0)
    f1 = f1_score(y, y_pred_class)
    bal_acc = balanced_accuracy_score(y, y_pred_class)
    mcc = matthews_corrcoef(y, y_pred_class)
    cm = confusion_matrix(y, y_pred_class)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Also calculate subject-level descriptive stats (not diagnostic aggregation, just stats)
    df['pred_prob'] = y_pred_probs
    df['pred_class'] = y_pred_class
    
    results = {
        'locked_threshold_used': float(threshold),
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
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'probability_distribution': {
            'min': float(np.min(y_pred_probs)),
            'max': float(np.max(y_pred_probs)),
            'mean': float(np.mean(y_pred_probs)),
            'std': float(np.std(y_pred_probs))
        }
    }

    print("\n--- ZERO-SHOT Frozen External Validation Results on C-NMC ---")
    for k, v in results.items():
        if k != 'confusion_matrix' and k != 'probability_distribution':
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved zero-shot results to {args.out_json}")

if __name__ == '__main__':
    main()
