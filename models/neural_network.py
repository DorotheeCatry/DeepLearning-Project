import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
import os

def create_model(input_dim, learning_rate=0.001):
    """
    Create a neural network model for churn prediction.
    
    Args:
        input_dim: Number of input features
        learning_rate: Learning rate for the optimizer
        
    Returns:
        Compiled Keras model
    """
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    
    model = Sequential([
        # Input layer
        Dense(128, activation='relu', input_dim=input_dim, 
              kernel_initializer='he_normal', 
              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Hidden layer 1
        Dense(64, activation='relu', 
              kernel_initializer='he_normal',
              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),
        
        # Hidden layer 2
        Dense(32, activation='relu', 
              kernel_initializer='he_normal',
              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        
        # Output layer
        Dense(1, activation='sigmoid')
    ])
    
    # Compile the model with Adam optimizer and binary cross-entropy loss
    optimizer = Adam(learning_rate=learning_rate)
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
    
    # Print model summary
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