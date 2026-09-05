import tensorflow as tf
from tensorflow.keras import layers, models

def build_baseline_cnn(input_shape=(224, 224, 3), norm_type='batchnorm', bn_momentum=0.99):
    """
    Builds a simple baseline Convolutional Neural Network for binary classification.
    
    Args:
        input_shape (tuple): The shape of the input images.
        norm_type (str): 'batchnorm' or 'layernorm'
        bn_momentum (float): Momentum for BatchNormalization
        
    Returns:
        tf.keras.Model: Uncompiled Keras model.
    """
    
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Rescaling(1./255)(inputs)
    
    x = layers.RandomFlip("horizontal_and_vertical")(x)
    x = layers.RandomRotation(0.2)(x)
    
    def apply_norm(tensor):
        if norm_type == 'batchnorm':
            return layers.BatchNormalization(momentum=bn_momentum)(tensor)
        elif norm_type == 'layernorm':
            return layers.LayerNormalization()(tensor)
        return tensor
        
    # Block 1
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = apply_norm(x)
    
    # Block 2
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = apply_norm(x)
    
    # Block 3
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = apply_norm(x)
    
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
