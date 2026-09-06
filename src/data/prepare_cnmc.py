import os
import argparse
import pandas as pd
import glob
import re
import json
import tensorflow as tf

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
        # Note: tf.config.experimental.get_device_details is available in newer TF versions
        try:
            details = tf.config.experimental.get_device_details(gpu)
            if 'device_name' in details:
                print(f"GPU Name: {details['device_name']}")
        except:
            pass

    # T4 mixed precision check
    policy = tf.keras.mixed_precision.global_policy()
    print(f"Mixed Precision Policy: {policy.name}")

def check_image_readable(filepath):
    """Verifies that an image file is readable by TensorFlow."""
    try:
        image_string = tf.io.read_file(filepath)
        _ = tf.image.decode_image(image_string, channels=3, expand_animations=False)
        return True
    except Exception as e:
        return False

def parse_cnmc_directory(base_dir):
    """
    Parses the C-NMC 2019 dataset directory to extract filepaths, subject IDs, and labels.
    """
    records = []
    
    folds = ['fold_0', 'fold_1', 'fold_2']
    classes = ['all', 'hem']
    
    total_images = 0
    unreadable_count = 0
    unknown_labels = 0
    
    for fold in folds:
        fold_path = os.path.join(base_dir, fold)
        if not os.path.exists(fold_path):
            print(f"Warning: {fold_path} not found.")
            continue
            
        for class_name in classes:
            class_path = os.path.join(fold_path, class_name)
            if not os.path.exists(class_path):
                print(f"Warning: {class_path} not found.")
                continue
                
            label = 1 if class_name == 'all' else (0 if class_name == 'hem' else None)
            if label is None:
                unknown_labels += 1
                continue
                
            for img_file in os.listdir(class_path):
                if not img_file.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png')):
                    continue
                    
                total_images += 1
                filepath = os.path.join(class_path, img_file)
                
                # Check readability (sampled or full, we do full for rigorous audit)
                if not check_image_readable(filepath):
                    unreadable_count += 1
                    continue
                    
                # Extract subject ID. Example: UID_11_1_1_all.bmp -> UID_11
                # Or UID_H10_... -> UID_H10
                match = re.search(r'(UID_[A-Za-z0-9]+)_', img_file)
                if not match:
                    print(f"CRITICAL ERROR: Cannot parse subject ID from {img_file}")
                    exit(1)
                    
                subject_id = match.group(1)
                
                records.append({
                    'filepath': filepath,
                    'filename': img_file,
                    'subject_id': subject_id,
                    'label': label,
                    'official_fold': fold
                })
    
    df = pd.DataFrame(records)
    
    return df, total_images, unreadable_count, unknown_labels

def main():
    parser = argparse.ArgumentParser(description="Prepare and Audit C-NMC Subset Manifest")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to C-NMC dataset (contains fold_0, fold_1, fold_2)")
    parser.add_argument('--out_csv', type=str, default='cnmc_subset_manifest.csv', help="Output CSV path")
    parser.add_argument('--out_audit', type=str, default='cnmc_subset_audit.json', help="Output audit JSON path")
    args = parser.parse_args()
    
    print("Verifying Hardware...")
    verify_gpu()
    
    print(f"\nScanning C-NMC directory: {args.data_dir}")
    df, total_scanned, unreadable, unknown_lbl = parse_cnmc_directory(args.data_dir)
    
    if len(df) == 0:
        print("CRITICAL ERROR: No valid images found. Aborting.")
        exit(1)
        
    print(f"Total images scanned: {total_scanned}")
    print(f"Valid and readable images: {len(df)}")
    print(f"Unreadable images: {unreadable}")
    print(f"Unknown labels: {unknown_lbl}")
    
    if unreadable > 0 or unknown_lbl > 0:
        print("CRITICAL ERROR: Dataset contains unreadable images or unknown labels. Aborting.")
        exit(1)
        
    # Audit metrics
    total_all = len(df[df['label'] == 1])
    total_hem = len(df[df['label'] == 0])
    total_subjects = df['subject_id'].nunique()
    
    print(f"\n--- AUDIT RESULTS ---")
    print(f"Total ALL: {total_all}")
    print(f"Total HEM: {total_hem}")
    print(f"Total Unique Subjects: {total_subjects}")
    
    fold_audit = {}
    subject_sets = {}
    
    for fold in ['fold_0', 'fold_1', 'fold_2']:
        fold_df = df[df['official_fold'] == fold]
        subjects = set(fold_df['subject_id'].unique())
        subject_sets[fold] = subjects
        
        all_cnt = len(fold_df[fold_df['label'] == 1])
        hem_cnt = len(fold_df[fold_df['label'] == 0])
        
        fold_audit[fold] = {
            'images': len(fold_df),
            'subjects': len(subjects),
            'ALL': all_cnt,
            'HEM': hem_cnt
        }
        print(f"\n{fold}:")
        print(f"  Images: {len(fold_df)} (ALL: {all_cnt}, HEM: {hem_cnt})")
        print(f"  Subjects: {len(subjects)}")

    # Check overlaps
    overlap_0_1 = subject_sets['fold_0'].intersection(subject_sets['fold_1'])
    overlap_0_2 = subject_sets['fold_0'].intersection(subject_sets['fold_2'])
    overlap_1_2 = subject_sets['fold_1'].intersection(subject_sets['fold_2'])
    
    print("\nSubject Overlap Check:")
    print(f"fold_0 ∩ fold_1: {len(overlap_0_1)} subjects")
    print(f"fold_0 ∩ fold_2: {len(overlap_0_2)} subjects")
    print(f"fold_1 ∩ fold_2: {len(overlap_1_2)} subjects")
    
    if len(overlap_0_1) > 0 or len(overlap_0_2) > 0 or len(overlap_1_2) > 0:
        print("CRITICAL ERROR: Subject leakage detected between official folds! Aborting.")
        exit(1)
        
    print("\nAudit PASSED. Official folds are subject-disjoint.")
    
    audit_data = {
        'total_images': len(df),
        'total_all': total_all,
        'total_hem': total_hem,
        'total_subjects': total_subjects,
        'unreadable': unreadable,
        'unknown_labels': unknown_lbl,
        'folds': fold_audit
    }
    
    # Save outputs
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_audit)), exist_ok=True)
    
    df.to_csv(args.out_csv, index=False)
    with open(args.out_audit, 'w') as f:
        json.dump(audit_data, f, indent=4)
        
    print(f"\nSaved manifest to {args.out_csv}")
    print(f"Saved audit to {args.out_audit}")

if __name__ == '__main__':
    main()
