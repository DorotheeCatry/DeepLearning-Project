import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import os

def create_model(input_dim, learning_rate=0.001):
    """
    Creates a neural network model for churn prediction
    
    Args:
        input_dim: Number of input features
        learning_rate: Learning rate for Adam optimizer
    
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
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.Precision(name='precision')
        ]
    )
    
    return model

def get_callbacks(model_dir='models'):
    """
    Creates callbacks for model training
    
    Args:
        model_dir: Directory to save model checkpoints
        
    Returns:
        List of Keras callbacks
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    checkpoint_path = os.path.join(model_dir, 'best_model.h5')
    tensorboard_path = os.path.join(model_dir, 'logs')
    
    callbacks = [
        EarlyStopping(
            monitor='val_auc',
            patience=10,
            restore_best_weights=True,
            mode='max'
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_auc',
            save_best_only=True,
            mode='max'
        ),
        TensorBoard(
            log_dir=tensorboard_path,
            histogram_freq=1
        )
    ]
    
    return callbacks