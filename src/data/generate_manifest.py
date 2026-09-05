import os
import glob
import hashlib
import pandas as pd

def compute_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def generate_manifest(data_dir, output_csv):
    files = glob.glob(os.path.join(data_dir, '**', '*.jpg'), recursive=True)
    files.extend(glob.glob(os.path.join(data_dir, '**', '*.png'), recursive=True))
    files.extend(glob.glob(os.path.join(data_dir, '**', '*.tif'), recursive=True))
    
    records = []
    seen_hashes = set()
    
    duplicate_count = 0
    
    for f in sorted(files):
        file_hash = compute_md5(f)
        label_name = os.path.basename(os.path.dirname(f))
        label = 1 if 'Blast' in label_name and 'Non' not in label_name else 0
        
        is_duplicate = False
        if file_hash in seen_hashes:
            is_duplicate = True
            duplicate_count += 1
        else:
            seen_hashes.add(file_hash)
            
        records.append({
            'filepath': os.path.abspath(f),
            'filename': os.path.basename(f),
            'label_name': label_name,
            'label': label,
            'md5': file_hash,
            'is_duplicate': is_duplicate
        })
        
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"Manifest generated at {output_csv}")
    print(f"Total files: {len(df)}")
    print(f"Duplicates marked: {duplicate_count}")
    print(f"Unique files: {len(df) - duplicate_count}")

if __name__ == "__main__":
    generate_manifest(
        data_dir="All-IDB dataset", 
        output_csv=os.path.join(os.path.dirname(__file__), 'manifest.csv')
    )
