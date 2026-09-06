import os
import argparse
import pandas as pd
import glob
import re

def parse_cnmc_directory(base_dir):
    """
    Parses the C-NMC 2019 dataset directory to extract filepaths, subject IDs, and labels.
    Expected structure (ISBI / Kaggle mirror):
    C-NMC_training_data/fold_0/...
    C-NMC_training_data/fold_1/...
    C-NMC_training_data/fold_2/...
    """
    records = []
    
    # Process training folds which have UID in filename
    train_dir = os.path.join(base_dir, 'C-NMC_training_data')
    if os.path.exists(train_dir):
        for fold in ['fold_0', 'fold_1', 'fold_2']:
            fold_path = os.path.join(train_dir, fold)
            if not os.path.exists(fold_path):
                continue
                
            for class_name, label in [('all', 1), ('hem', 0)]:
                class_path = os.path.join(fold_path, class_name)
                if not os.path.exists(class_path):
                    continue
                    
                for img_file in os.listdir(class_path):
                    if not img_file.lower().endswith('.bmp'):
                        continue
                        
                    # Extract subject ID. Example: UID_11_1_1_all.bmp -> UID_11
                    match = re.search(r'(UID_[A-Za-z0-9]+)_', img_file)
                    subject_id = match.group(1) if match else "UNKNOWN"
                    
                    filepath = os.path.join(class_path, img_file)
                    records.append({
                        'filepath': filepath,
                        'subject_id': subject_id,
                        'label': label,
                        'split_source': f'train_{fold}'
                    })
    
    # We could process the prelim test phase, but it requires joining with the CSV.
    # For now, the 10,000+ training images are more than sufficient for external validation and downstream patient splitting.
    
    df = pd.DataFrame(records)
    return df

def main():
    parser = argparse.ArgumentParser(description="Prepare C-NMC 2019 Manifest")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to extracted C-NMC dataset (contains C-NMC_training_data)")
    parser.add_argument('--out_csv', type=str, default='cnmc_manifest.csv', help="Output CSV path")
    args = parser.parse_args()
    
    print(f"Scanning C-NMC directory: {args.data_dir}")
    df = parse_cnmc_directory(args.data_dir)
    
    if len(df) == 0:
        print("Error: No images found. Check data_dir path.")
        return
        
    print(f"Found {len(df)} images.")
    print("\nClass Distribution:")
    print(df['label'].value_counts())
    print(f"\nUnique Subjects: {df['subject_id'].nunique()}")
    
    df.to_csv(args.out_csv, index=False)
    print(f"Saved manifest to {args.out_csv}")

if __name__ == '__main__':
    main()
