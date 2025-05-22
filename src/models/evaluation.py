```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def evaluate_models(nn_model, gb_model, ensemble_model, X_test, y_test):
    """
    Evaluate neural network, gradient boosting, and ensemble models.
    
    Args:
        nn_model: Trained neural network model
        gb_model: Trained gradient boosting model
        ensemble_model: Trained ensemble model
        X_test: Preprocessed test features
        y_test: Test target (encoded)
    """
    # Get predictions from all models
    nn_pred = nn_model.predict(X_test)
    gb_pred = gb_model.predict(X_test)
    ensemble_pred = ensemble_model.predict(X_test)
    
    # Get probabilities for ROC AUC
    nn_pred_proba = nn_model.predict_proba(X_test)[:, 1]
    gb_pred_proba = gb_model.predict_proba(X_test)[:, 1]
    ensemble_pred_proba = ensemble_model.predict_proba(X_test)[:, 1]
    
    # Print classification reports
    print("\nNeural Network Classification Report:")
    print(classification_report(y_test, nn_pred, target_names=['no', 'yes']))
    
    print("\nGradient Boosting Classification Report:")
    print(classification_report(y_test, gb_pred, target_names=['no', 'yes']))
    
    print("\nEnsemble Model Classification Report:")
    print(classification_report(y_test, ensemble_pred, target_names=['no', 'yes']))
    
    # Calculate and print ROC AUC scores
    nn_auc = roc_auc_score(y_test, nn_pred_proba)
    gb_auc = roc_auc_score(y_test, gb_pred_proba)
    ensemble_auc = roc_auc_score(y_test, ensemble_pred_proba)
    
    print("\nROC AUC Scores:")
    print(f"Neural Network: {nn_auc:.4f}")
    print(f"Gradient Boosting: {gb_auc:.4f}")
    print(f"Ensemble Model: {ensemble_auc:.4f}")
    
    # Print confusion matrices
    print("\nNeural Network Confusion Matrix:")
    print(confusion_matrix(y_test, nn_pred))
    
    print("\nGradient Boosting Confusion Matrix:")
    print(confusion_matrix(y_test, gb_pred))
    
    print("\nEnsemble Model Confusion Matrix:")
    print(confusion_matrix(y_test, ensemble_pred))
```