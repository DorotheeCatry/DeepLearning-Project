import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, average_precision_score
import joblib
import os
from tensorflow.keras.models import clone_model, load_model
from src.models.neural_network import build_model

class ManualEnsemble:
    def __init__(self, nn_model, gb_model, weights=(0.5, 0.5)):
        self.nn_model = nn_model
        self.gb_model = gb_model
        self.weights = weights
    
    def predict_proba(self, X):
        nn_prob = self.nn_model.predict_proba(X)
        gb_prob = self.gb_model.predict_proba(X)
        return self.weights[0] * nn_prob + self.weights[1] * gb_prob
    
    def predict(self, X, threshold=0.5):
        probas = self.predict_proba(X)
        return (probas[:, 1] >= threshold).astype(int)
    
    def save(self, path):
        """Save the ensemble model components separately"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save neural network
        self.nn_model.model_.save(f"{path}_nn.keras")
        
        # Save gradient boosting
        joblib.dump(self.gb_model, f"{path}_gb.pkl")
        
        # Save weights
        joblib.dump(self.weights, f"{path}_weights.pkl")
    
    @classmethod
    def load(cls, path):
        """Load the ensemble model from saved components"""
        # Load neural network
        nn_model = KerasClassifier(model=load_model(f"{path}_nn.keras"))
        
        # Load gradient boosting
        gb_model = joblib.load(f"{path}_gb.pkl")
        
        # Load weights
        weights = joblib.load(f"{path}_weights.pkl")
        
        return cls(nn_model, gb_model, weights)

def create_keras_model(input_dim):
    """Factory function to create the Keras model"""
    return build_model(input_dim)

def train_ensemble(X_train, X_val, y_train, y_val, gb_params, nn_epochs=100, batch_size=32, voting_weights=(0.5, 0.5)):
    """
    Train an ensemble model combining Neural Network and Gradient Boosting using manual soft voting.
    
    Args:
        X_train: Training features
        X_val: Validation features
        y_train: Training labels
        y_val: Validation labels
        gb_params: Gradient boosting parameters
        nn_epochs: Number of epochs for neural network
        batch_size: Batch size for neural network
        voting_weights: Tuple of (nn_weight, gb_weight) for soft voting
    """
    input_dim = X_train.shape[1]
    
    # Neural Network
    nn_clf = KerasClassifier(
        model=lambda: create_keras_model(input_dim),
        epochs=nn_epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Gradient Boosting
    allowed_params = {
        'n_estimators', 'learning_rate', 'max_depth', 
        'min_samples_split', 'min_samples_leaf', 
        'subsample', 'max_features', 'random_state'
    }
    filtered_params = {k: v for k, v in gb_params.items() if k in allowed_params}
    gb_clf = GradientBoostingClassifier(**filtered_params)

    # Train models separately
    print("Training Neural Network...")
    nn_clf.fit(X_train, y_train, validation_data=(X_val, y_val))
    
    print("\nTraining Gradient Boosting...")
    gb_clf.fit(X_train, y_train)

    # Create ensemble
    ensemble = ManualEnsemble(nn_clf, gb_clf, voting_weights)

    # Evaluate ensemble
    y_pred = ensemble.predict(X_val)
    y_prob = ensemble.predict_proba(X_val)[:, 1]

    print("\nEnsemble Model Performance:")
    print(classification_report(y_val, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_val, y_prob):.4f}")
    print(f"PR AUC: {average_precision_score(y_val, y_prob):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    # Save the ensemble model
    ensemble.save("data/models/ensemble_model")
    
    return ensemble
