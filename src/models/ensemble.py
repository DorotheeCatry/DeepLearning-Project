from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, average_precision_score
import numpy as np
import joblib

from src.models.neural_network import build_model

def train_ensemble(X_train_array, X_val_array, y_train, y_val, preprocessing_layers, gb_params, nn_epochs=100, batch_size=32):
    """
    Train an ensemble model combining NN and GB.
    
    Args:
        X_train_array: Training features array
        X_val_array: Validation features array
        y_train: Training labels
        y_val: Validation labels
        preprocessing_layers: Preprocessing layers dictionary
        gb_params: Gradient boosting parameters
        nn_epochs: Number of epochs for neural network
        batch_size: Batch size for neural network
    """
    # Neural Network
    input_dim = X_train_array.shape[1]
    nn_clf = KerasClassifier(
        model=lambda: build_model(input_dim),
        epochs=nn_epochs,
        batch_size=batch_size,
        verbose=0
    )

    # Gradient Boosting
    allowed_params = {
        'n_estimators', 'learning_rate', 'max_depth', 
        'min_samples_split', 'min_samples_leaf', 
        'subsample', 'max_features', 'random_state'
    }
    filtered_params = {k: v for k, v in gb_params.items() if k in allowed_params}
    gb_clf = GradientBoostingClassifier(**filtered_params)

    # Ensemble
    voting_clf = VotingClassifier(
        estimators=[
            ('nn', nn_clf),
            ('gb', gb_clf)
        ],
        voting='soft'
    )

    print("Training ensemble model...")
    voting_clf.fit(X_train_array, y_train)

    # Evaluation
    y_pred = voting_clf.predict(X_val_array)
    y_prob = voting_clf.predict_proba(X_val_array)[:, 1]

    print("\nEnsemble Model Performance:")
    print(classification_report(y_val, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_val, y_prob):.4f}")
    print(f"PR AUC: {average_precision_score(y_val, y_prob):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    # Save model
    joblib.dump(voting_clf, "data/models/ensemble_model.pkl")
    
    return voting_clf