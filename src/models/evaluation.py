from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def evaluate_models(nn_model, gb_model, X_test_dict, X_test_processed, y_test):
    """
    Evaluate both neural network and gradient boosting models.
    
    Args:
        nn_model: Trained neural network model
        gb_model: Trained gradient boosting model
        X_test_dict: Test features in dictionary format for neural network
        X_test_processed: Preprocessed test features for gradient boosting
        y_test: Test target (encoded)
    """
    # Neural Network predictions
    nn_pred_proba = nn_model.predict(X_test_dict)
    nn_pred = (nn_pred_proba > 0.5).astype(int)
    
    # Gradient Boosting predictions
    gb_pred_proba = gb_model.predict_proba(X_test_processed)[:, 1]
    gb_pred = (gb_pred_proba > 0.5).astype(int)
    
    # Print classification reports
    print("\nNeural Network Classification Report:")
    print(classification_report(y_test, nn_pred, target_names=['no', 'yes']))
    
    print("\nGradient Boosting Classification Report:")
    print(classification_report(y_test, gb_pred, target_names=['no', 'yes']))
    
    # Calculate and print ROC AUC scores
    nn_auc = roc_auc_score(y_test, nn_pred_proba)
    gb_auc = roc_auc_score(y_test, gb_pred_proba)
    
    print("\nROC AUC Scores:")
    print(f"Neural Network: {nn_auc:.4f}")
    print(f"Gradient Boosting: {gb_auc:.4f}")
    
    # Print confusion matrices
    print("\nNeural Network Confusion Matrix:")
    print(confusion_matrix(y_test, nn_pred))
    
    print("\nGradient Boosting Confusion Matrix:")
    print(confusion_matrix(y_test, gb_pred))