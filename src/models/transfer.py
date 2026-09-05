import tensorflow as tf
from tensorflow.keras import layers, models

def get_conservative_augmentation():
    """
    Step 5: Conservative Augmentation for Medical Fidelity
    We avoid aggressive color jittering or extreme shearing that could destroy 
    cellular morphology or staining characteristics.
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),  # +/- 20% of 2pi (i.e. up to 72 degrees)
        layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1))
    ], name="conservative_augmentation")

def build_transfer_model(model_name="resnet50", input_shape=(224, 224, 3), freeze_base=True):
    """
    Builds a Transfer Learning model using pre-trained ImageNet weights.
    
    Args:
        model_name (str): 'resnet50', 'efficientnetb0', or 'convnexttiny'
        input_shape (tuple): Shape of the input images
        freeze_base (bool): Whether to freeze the base model for feature extraction
        
    Returns:
        tf.keras.Model: Uncompiled Keras model.
    """
    inputs = tf.keras.Input(shape=input_shape)
    
    # Models have different preprocessing requirements.
    # ConvNeXt and EfficientNet expect [0, 255] or handle rescaling internally in Keras.
    # ResNet50 expects specific preprocessing (usually centered around 0 with BGR).
    # To keep things standardized across models in Keras 3 / tf.keras.applications,
    # the application typically includes the necessary rescaling in the model itself 
    # if include_top=False, or we can use the dedicated preprocess_input functions.
    
    # Actually, the safest approach is to use the specific preprocess_input function
    # inside a Lambda layer, OR realize that EfficientNet and ConvNeXt explicitly 
    # expect [0, 255] inputs and include their own rescaling. 
    # ResNet50 expects BGR inputs zero-centered with respect to ImageNet dataset.
    
    if model_name.lower() == 'resnet50':
        base_model = tf.keras.applications.ResNet50(
            weights='imagenet', include_top=False, input_tensor=inputs
        )
        # We must preprocess inputs for ResNet50
        # tf.keras.applications.resnet50.preprocess_input expects inputs in [0, 255] BGR
        # Wait, if we use input_tensor, we can't easily prepend a preprocessing layer to a prebuilt model
        # unless we do it this way:
        inputs = tf.keras.Input(shape=input_shape)
        x = get_conservative_augmentation()(inputs)
        x = tf.keras.applications.resnet50.preprocess_input(x)
        base_model = tf.keras.applications.ResNet50(
            weights='imagenet', include_top=False, input_tensor=x
        )
        x = base_model.output
        
    elif model_name.lower() == 'efficientnetb0':
        inputs = tf.keras.Input(shape=input_shape)
        x = get_conservative_augmentation()(inputs)
        # EfficientNetB0 expects inputs in [0, 255] and includes its own rescaling
        base_model = tf.keras.applications.EfficientNetB0(
            weights='imagenet', include_top=False, input_tensor=x
        )
        x = base_model.output
        
    elif model_name.lower() == 'convnexttiny':
        inputs = tf.keras.Input(shape=input_shape)
        x = get_conservative_augmentation()(inputs)
        # ConvNeXtTiny expects inputs in [0, 255] and includes its own rescaling
        base_model = tf.keras.applications.ConvNeXtTiny(
            weights='imagenet', include_top=False, input_tensor=x
        )
        x = base_model.output
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
        
    if freeze_base:
        base_model.trainable = False
        
    # Classification Head
    x = layers.GlobalAveragePooling2D()(x)
    
    # We use LayerNormalization instead of BatchNormalization in the custom head
    # to avoid the exact small-dataset BN statistics issue we discovered.
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs, name=f"{model_name}_transfer")
    return model
