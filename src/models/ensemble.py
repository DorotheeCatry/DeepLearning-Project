import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, average_precision_score
import joblib
import os
import tensorflow as tf

from src.models.neural_network import build_model

def create_model(input_dim):
    return build_model(input_dim)

class ManualEnsemble:
    def __init__(self, nn_model, gb_model, weights=(0.4, 0.6)):
        self.nn_model = nn_model
        self.gb_model = gb_model
        self.weights = weights
    
    def predict_proba(self, X):
        nn_prob = self.nn_model.predict_proba(X)
        gb_prob = self.gb_model.predict_proba(X)
        
        # Weighted average of probabilities
        ensemble_prob = (self.weights[0] * nn_prob + 
                        self.weights[1] * gb_prob)
        
        # Normalize probabilities
        ensemble_prob = ensemble_prob / np.sum(ensemble_prob, axis=1, keepdims=True)
        return ensemble_prob
    
    def predict(self, X, threshold=0.4):
        probas = self.predict_proba(X)
        return (probas[:, 1] >= threshold).astype(int)

def train_ensemble(X_train, X_val, y_train, y_val, gb_params, nn_epochs=150, batch_size=32, voting_weights=(0.4, 0.6)):
    """
    Train an improved ensemble model with adjusted parameters for better performance.
    """
    input_dim = X_train.shape[1]
    
    # Calculate class weights
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    weight_for_0 = (1 / n_neg) * (n_neg + n_pos) / 2.0
    weight_for_1 = (1 / n_pos) * (n_neg + n_pos) / 2.0
    
    class_weight = {0: weight_for_0, 1: weight_for_1}
    
    # Neural Network with early stopping and class weights
    model_builder = lambda: create_model(input_dim)
    nn_clf = KerasClassifier(
        model=model_builder,
        epochs=nn_epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Gradient Boosting with filtered parameters
    allowed_params = {
        'n_estimators', 'learning_rate', 'max_depth', 
        'min_samples_split', 'min_samples_leaf', 
        'subsample', 'max_features', 'random_state'
    }
    filtered_params = {k: v for k, v in gb_params.items() if k in allowed_params}
    gb_clf = GradientBoostingClassifier(**filtered_params)
    
    print("Training Neural Network...")
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    nn_clf.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        class_weight=class_weight,
        callbacks=[early_stopping]
    )
    
    print("\nTraining Gradient Boosting...")
    gb_clf.fit(X_train, y_train)
    
    # Create and evaluate ensemble
    ensemble = ManualEnsemble(nn_clf, gb_clf, voting_weights)
    
    # Evaluate on validation set
    y_pred = ensemble.predict(X_val)
    y_prob = ensemble.predict_proba(X_val)[:, 1]
    
    print("\nEnsemble Model Performance:")
    print(classification_report(y_val, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_val, y_prob):.4f}")
    print(f"PR AUC: {average_precision_score(y_val, y_prob):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
    
    # Save models
    os.makedirs('data/models', exist_ok=True)
    
    # Save individual models and ensemble configuration
    model_data = {
        'gb_model': gb_clf,
        'weights': voting_weights
    }
    joblib.dump(model_data, "data/models/ensemble_components.pkl")
    
    return ensemble