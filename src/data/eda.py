import os
import glob
import numpy as np
import cv2

def compute_dataset_stats(data_dir: str):
    """
    Computes class distribution, mean, and standard deviation of the dataset.
    This serves as an Exploratory Data Analysis (EDA) step and provides
    normalization constants.
    
    Args:
        data_dir (str): Path to the base dataset directory (e.g., 'All-IDB dataset').
    """
    splits = ['train', 'val']
    classes = ['nonblast', 'Blast']
    
    print("=== Dataset Distribution ===")
    total_images = 0
    
    # Calculate distributions
    for split in splits:
        split_total = 0
        for cls in classes:
            path = os.path.join(data_dir, split, cls)
            if not os.path.exists(path):
                print(f"Warning: Directory not found - {path}")
                continue
            
            # Count jpg files
            num_files = len(glob.glob(os.path.join(path, "*.jpg")))
            print(f"{split.upper()} - {cls}: {num_files} images")
            split_total += num_files
            total_images += num_files
        print(f"Total {split.upper()} images: {split_total}\n")
    
    print(f"Total Dataset size: {total_images} images")
    
    print("\n=== Computing Pixel Mean and Std ===")
    print("Loading training images...")
    
    # To avoid memory issues, we can compute running mean/var or just load all (small dataset)
    train_images = glob.glob(os.path.join(data_dir, 'train', '**', '*.jpg'), recursive=True)
    
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    total_pixels = 0
    
    for img_path in train_images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read {img_path}")
            continue
        
        # cv2 loads as BGR, convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
        
        # Reshape to (N, 3) where N is number of pixels
        pixels = img.reshape(-1, 3)
        
        pixel_sum += np.sum(pixels, axis=0)
        pixel_sq_sum += np.sum(pixels ** 2, axis=0)
        total_pixels += pixels.shape[0]
        
    if total_pixels > 0:
        mean = pixel_sum / total_pixels
        # var = E[X^2] - (E[X])^2
        variance = (pixel_sq_sum / total_pixels) - (mean ** 2)
        std = np.sqrt(variance)
        
        print(f"Train Dataset Mean (RGB): {mean}")
        print(f"Train Dataset Std (RGB): {std}")
    else:
        print("No valid images found for mean/std computation.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run EDA on dataset")
    parser.add_argument("--data_dir", type=str, default="All-IDB dataset", help="Path to the dataset directory")
    args = parser.parse_args()
    
    compute_dataset_stats(args.data_dir)
