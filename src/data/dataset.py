import os
import tensorflow as tf

def get_datasets(data_dir: str, image_size: tuple = (224, 224), batch_size: int = 32):
    """
    Creates and returns TensorFlow datasets for training and validation.
    The dataset is loaded from a directory containing 'train' and 'val' subdirectories,
    each with 'Blast' and 'nonblast' folders.
    
    Args:
        data_dir (str): Path to the base dataset directory (e.g., 'All-IDB dataset').
        image_size (tuple): Target size to resize images to.
        batch_size (int): Batch size for the datasets.
        
    Returns:
        tuple: (train_ds, val_ds) tf.data.Dataset objects.
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(f"Missing 'train' or 'val' directory inside {data_dir}")

    # Load Training Dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='binary',  # Returns float32 scalars (0.0 or 1.0)
        class_names=['nonblast', 'Blast'],  # 0: nonblast, 1: Blast
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=42
    )

    # Load Validation Dataset
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels='inferred',
        label_mode='binary',
        class_names=['nonblast', 'Blast'],
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False
    )
    
    # Optional: Prefetching for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds
