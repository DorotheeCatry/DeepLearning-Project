from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, average_precision_score
import numpy as np
import joblib

from src.models.neural_network import build_model  # Assure-toi que cette fonction existe et retourne un modèle Keras compilé


def train_ensemble(X_train, X_val, y_train, y_val, best_gb_params, nn_epochs=100, batch_size=32):
    # 1. Build KerasClassifier (with SciKeras)
    nn_clf = KerasClassifier(
        model=build_model,
        epochs=nn_epochs,
        batch_size=batch_size,
        verbose=0
    )

    # 2. Build Gradient Boosting Classifier with best hyperparameters
    gb_clf = GradientBoostingClassifier(**best_gb_params)

    # 3. Voting Ensemble (soft voting)
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

    # Optionally save model
    joblib.dump(voting_clf, "ensemble_model.pkl")

    return voting_clf
