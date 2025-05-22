import numpy as np
from sklearn.ensemble import VotingClassifier
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import classification_report, roc_auc_score
import tensorflow as tf

def create_keras_classifier(build_model_fn, preprocessing_layers, learning_rate=0.001):
    """
    Create a KerasClassifier wrapper around our neural network.
    
    Args:
        build_model_fn: Function that builds and returns the Keras model
        preprocessing_layers: Dictionary of preprocessing layers
        learning_rate: Learning rate for the optimizer
    
    Returns:
        KerasClassifier instance
    """
    def create_model():
        model = build_model_fn(preprocessing_layers, learning_rate)
        return model
    
    return KerasClassifier(
        model=create_model,
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
    )

def create_voting_classifier(nn_model, gb_model):
    """
    Create a VotingClassifier that combines neural network and gradient boosting models.
    
    Args:
        nn_model: Trained KerasClassifier
        gb_model: Trained GradientBoostingClassifier
    
    Returns:
        VotingClassifier instance
    """
    return VotingClassifier(
        estimators=[
            ('nn', nn_model),
            ('gb', gb_model)
        ],
        voting='soft'
    )

def train_ensemble(nn_model, gb_model, X_train, y_train, X_val, y_val):
    """
    Train the ensemble model.
    
    Args:
        nn_model: Neural network model (KerasClassifier)
        gb_model: Gradient boosting model
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
    
    Returns:
        Trained VotingClassifier
    """
    # Create and train the voting classifier
    voting_clf = create_voting_classifier(nn_model, gb_model)
    
    print("\nTraining ensemble model...")
    voting_clf.fit(X_train, y_train)
    
    # Evaluate ensemble
    y_pred = voting_clf.predict(X_val)
    y_proba = voting_clf.predict_proba(X_val)[:, 1]
    
    print("\nEnsemble Model Validation Metrics:")
    print(classification_report(y_val, y_pred, target_names=['no', 'yes']))
    print(f"ROC AUC: {roc_auc_score(y_val, y_proba):.4f}")
    
    return voting_clf