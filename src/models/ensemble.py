import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, average_precision_score
import joblib
import os

from src.models.neural_network import build_model

# En dehors de train_ensemble (dans ensemble.py par ex)
def create_model(input_dim):
    return build_model(input_dim)

class ManualEnsemble:
    def __init__(self, nn_model, gb_model, weights):
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


def train_ensemble(X_train, X_val, y_train, y_val, gb_params, nn_epochs=100, batch_size=32, voting_weights=(0.5, 0.5)):
    input_dim = X_train.shape[1]

    # Définition d'une fonction nommée pour KerasClassifier
    def model_fn():
        return create_model(input_dim)

    nn_clf = KerasClassifier(
        model=model_fn,
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

    # Entraînement des modèles
    print("Training Neural Network...")
    nn_clf.fit(X_train, y_train, validation_data=(X_val, y_val))

    print("\nTraining Gradient Boosting...")
    gb_clf.fit(X_train, y_train)

    ensemble = ManualEnsemble(nn_clf, gb_clf, voting_weights)

    # Évaluation
    y_pred = ensemble.predict(X_val)
    y_prob = ensemble.predict_proba(X_val)[:, 1]

    print("\nEnsemble Model Performance:")
    print(classification_report(y_val, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_val, y_prob):.4f}")
    print(f"PR AUC: {average_precision_score(y_val, y_prob):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    os.makedirs('data/models', exist_ok=True)

    # Sauvegarde des modèles et poids, sans sauvegarder l'objet ensemble (car classe locale évitée)
    joblib.dump({
        'nn_model': nn_clf,
        'gb_model': gb_clf,
        'weights': voting_weights
    }, "data/models/ensemble_model.pkl")

    return ensemble
