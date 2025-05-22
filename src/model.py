import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def create_model(input_dim):
    """
    Create a neural network model for churn prediction.
    
    Args:
        input_dim: Number of input features
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model

def get_callbacks():
    """
    Create callbacks for model training.
    
    Returns:
        List of Keras callbacks
    """
    callbacks = [
        EarlyStopping(
            monitor='val_auc',
            patience=10,
            restore_best_weights=True,
            mode='max'
        ),
        ModelCheckpoint(
            filepath='models/best_model.h5',
            monitor='val_auc',
            save_best_only=True,
            mode='max'
        )
    ]
    
    return callbacks