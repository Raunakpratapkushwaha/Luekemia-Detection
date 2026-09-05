import tensorflow as tf
from tensorflow.keras import layers, models

def build_baseline_cnn(input_shape=(224, 224, 3)):
    """
    Builds a simple baseline Convolutional Neural Network for binary classification.
    Since the dataset is very small (200 train images), we must heavily regularize
    to prevent extreme overfitting.
    
    Args:
        input_shape (tuple): The shape of the input images.
        
    Returns:
        tf.keras.Model: Compiled or uncompiled Keras model.
    """
    
    # Input and Data Augmentation Layer (integrated into model for deployment simplicity)
    inputs = tf.keras.Input(shape=input_shape)
    
    x = layers.Rescaling(1./255)(inputs)
    
    # Data Augmentation to combat small dataset size
    x = layers.RandomFlip("horizontal_and_vertical")(x)
    x = layers.RandomRotation(0.2)(x)
    
    # Block 1
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    # Block 2
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    # Block 3
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    # Global pooling is preferred over Flatten for better generalization
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dropout for regularization
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Binary classification output (1 neuron with sigmoid)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="Baseline_CNN")
    
    return model
