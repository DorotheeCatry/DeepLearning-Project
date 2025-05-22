from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def evaluate_models(nn_model, gb_model, ensemble_model, X_test, y_test):
    """
    Evaluate gradient boosting and ensemble models.
    
    Args:
        nn_model: Neural network model (can be None)
        gb_model: Trained gradient boosting model
        ensemble_model: Trained ensemble model
        X_test: Preprocessed test features
        y_test: Test target (encoded)
    """
    # Get predictions from models
    if nn_model is not None:
        nn_pred = nn_model.predict(X_test)
        nn_pred_proba = nn_model.predict_proba(X_test)[:, 1]
        
        print("\nNeural Network Classification Report:")
        print(classification_report(y_test, nn_pred))
        print(f"Neural Network ROC AUC: {roc_auc_score(y_test, nn_pred_proba):.4f}")
    
    gb_pred = gb_model.predict(X_test)
    gb_pred_proba = gb_model.predict_proba(X_test)[:, 1]
    
    ensemble_pred = ensemble_model.predict(X_test)
    ensemble_pred_proba = ensemble_model.predict_proba(X_test)[:, 1]
    
    print("\nGradient Boosting Classification Report:")
    print(classification_report(y_test, gb_pred))
    print(f"Gradient Boosting ROC AUC: {roc_auc_score(y_test, gb_pred_proba):.4f}")
    
    print("\nEnsemble Model Classification Report:")
    print(classification_report(y_test, ensemble_pred))
    print(f"Ensemble Model ROC AUC: {roc_auc_score(y_test, ensemble_pred_proba):.4f}")
    
    # Print confusion matrices
    if nn_model is not None:
        print("\nNeural Network Confusion Matrix:")
        print(confusion_matrix(y_test, nn_pred))
    
    print("\nGradient Boosting Confusion Matrix:")
    print(confusion_matrix(y_test, gb_pred))
    
    print("\nEnsemble Model Confusion Matrix:")
    print(confusion_matrix(y_test, ensemble_pred))