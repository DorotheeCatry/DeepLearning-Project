import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import os

def build_model(preprocessing_layers, learning_rate=0.001):
    """
    Build a Keras functional model with preprocessing layers integrated.

    Args:
        preprocessing_layers: dict, output of preprocess_data, mapping feature name to preprocessing layer(s)
        learning_rate: float, learning rate for optimizer
    
    Returns:
        model: compiled Keras model
    """

    inputs = []
    encoded_features = []

    for feature_name, layer in preprocessing_layers.items():
        if isinstance(layer, tf.keras.layers.Normalization):
            # Numeric feature
            inp = Input(shape=(1,), name=feature_name, dtype=tf.float32)
            x = layer(inp)
        else:
            # Categorical feature: layer is tuple (lookup, onehot)
            lookup, onehot = layer
            inp = Input(shape=(1,), name=feature_name, dtype=tf.string)
            x = lookup(inp)
            x = onehot(x)
        
        inputs.append(inp)
        encoded_features.append(x)

    # Concatenate all features
    x = Concatenate()(encoded_features)

    # Dense layers avec régularisation L2 et Dropout
    x = Dense(128, activation='relu',
              kernel_initializer='he_normal',
              kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(64, activation='relu',
              kernel_initializer='he_normal',
              kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(32, activation='relu',
              kernel_initializer='he_normal',
              kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)

    # Couche de sortie binaire
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    model.summary()
    return model

def get_callbacks():
    """
    Create callbacks for model training.
    
    Returns:
        List of Keras callbacks
    """
    # Create directories for logs and checkpoints if they don't exist
    os.makedirs('data/logs', exist_ok=True)
    os.makedirs('data/checkpoints', exist_ok=True)
    
    callbacks = [
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_auc',
            patience=10,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        # Model checkpoint to save the best model
        ModelCheckpoint(
            filepath='data/checkpoints/best_model.h5',
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # TensorBoard for visualization
        TensorBoard(
            log_dir='data/logs',
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
    ]
    
    return callbacks