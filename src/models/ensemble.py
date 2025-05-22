from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, average_precision_score
import joblib

from src.models.neural_network import build_model  # build_model(preprocessing_layers) → tu vas ajuster

def train_ensemble(
    X_train_array, X_val_array, y_train, y_val,
    preprocessing_layers, gb_params,
    nn_epochs=100, batch_size=32
):
    """
    Train an ensemble model combining NN and GB on the same 2D array.
    Args:
        X_train_array, X_val_array: numpy arrays shape (n_samples, n_features)
    """
    # 1. KerasClassifier pour un modèle qui prend un array
    def _build_flat_model():
        # Reconstruis un model identique, mais directement sur array
        input_dim = X_train_array.shape[1]
        return build_model(input_dim)  # une version de build_model qui prend input_dim
    
    nn_clf = KerasClassifier(
        model=_build_flat_model,
        epochs=nn_epochs,
        batch_size=batch_size,
        verbose=0
    )

    # 2. Gradient Boosting
    allowed = {'n_estimators','learning_rate','max_depth','min_samples_split',
               'min_samples_leaf','subsample','max_features','random_state'}
    params = {k:v for k,v in gb_params.items() if k in allowed}
    gb_clf = GradientBoostingClassifier(**params)

    # 3. VotingClassifier (soft)
    voting_clf = VotingClassifier(
        estimators=[('nn', nn_clf), ('gb', gb_clf)],
        voting='soft'
    )

    print("Training ensemble model...")
    voting_clf.fit(X_train_array, y_train)

    # 4. Évaluation
    y_pred = voting_clf.predict(X_val_array)
    y_prob = voting_clf.predict_proba(X_val_array)[:, 1]

    print(classification_report(y_val, y_pred))
    print("ROC AUC:", roc_auc_score(y_val, y_prob))
    print("PR AUC:", average_precision_score(y_val, y_prob))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

    joblib.dump(voting_clf, "data/models/ensemble_model.pkl")
    return voting_clf
