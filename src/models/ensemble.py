from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, average_precision_score
import numpy as np
import joblib

from src.models.neural_network import build_model

def train_ensemble(X_train, X_val, y_train, y_val, preprocessing_layers, gb_params, nn_epochs=100, batch_size=32):
    """
    Train an ensemble model combining neural network and gradient boosting.
    
    Args:
        X_train: Training features dictionary for neural network
        X_val: Validation features dictionary for neural network
        y_train: Training labels (encoded)
        y_val: Validation labels (encoded)
        preprocessing_layers: Dictionary of preprocessing layers for neural network
        gb_params: Gradient boosting parameters
        nn_epochs: Number of epochs for neural network training
        batch_size: Batch size for neural network training
    """
    # 1. Build KerasClassifier
    nn_clf = KerasClassifier(
        model=lambda: build_model(preprocessing_layers),
        epochs=nn_epochs,
        batch_size=batch_size,
        verbose=0
    )

    # 2. Build Gradient Boosting Classifier with filtered parameters
    allowed_params = {
        'n_estimators', 'learning_rate', 'max_depth', 'min_samples_split',
        'min_samples_leaf', 'subsample', 'max_features', 'random_state'
    }
    filtered_params = {k: v for k, v in gb_params.items() if k in allowed_params}
    gb_clf = GradientBoostingClassifier(**filtered_params)

    # 3. Create and train voting ensemble
    voting_clf = VotingClassifier(
        estimators=[
            ('nn', nn_clf),
            ('gb', gb_clf)
        ],
        voting='soft'
    )

    print("Training ensemble model...")
    voting_clf.fit(X_train, y_train)

    # 4. Evaluation
    y_pred = voting_clf.predict(X_val)
    y_prob = voting_clf.predict_proba(X_val)[:, 1]

    print("\n📊 Ensemble Classification Report:")
    print(classification_report(y_val, y_pred))

    roc_auc = roc_auc_score(y_val, y_prob)
    pr_auc = average_precision_score(y_val, y_prob)

    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    # Save model
    joblib.dump(voting_clf, "data/models/ensemble_model.pkl")

    return voting_clf