import numpy as np
from sklearn.ensemble import VotingClassifier
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
import tensorflow as tf

def create_keras_classifier(input_dim, learning_rate=0.001):
    """
    Create a KerasClassifier wrapper around our neural network.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_dim=input_dim,
                            kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(64, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(32, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

def train_ensemble(nn_model, gb_model, X_train, y_train, X_val, y_val):
    """
    Train the ensemble model.
    """
    # Create the voting classifier
    voting_clf = VotingClassifier(
        estimators=[
            ('nn', KerasClassifier(
                model=lambda: create_keras_classifier(X_train.shape[1]),
                epochs=100,
                batch_size=32,
                verbose=1,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_auc',
                        patience=10,
                        restore_best_weights=True,
                        mode='max'
                    )
                ]
            )),
            ('gb', gb_model)
        ],
        voting='soft'
    )
    
    # Train the ensemble
    print("\nTraining ensemble model...")
    voting_clf.fit(
        X_train, 
        y_train,
        nn__validation_data=(X_val, y_val)
    )
    
    # Evaluate ensemble
    y_pred = voting_clf.predict(X_val)
    y_proba = voting_clf.predict_proba(X_val)[:, 1]
    
    print("\nEnsemble Model Validation Metrics:")
    print(classification_report(y_val, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_val, y_proba):.4f}")
    
    return voting_clf