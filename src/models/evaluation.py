from sklearn.metrics import classification_report

def evaluate_models(nn_model, gb_model, X_test, y_test, label_encoder):
    """
    Evaluate both neural network and gradient boosting models.
    
    Args:
        nn_model: Trained neural network model
        gb_model: Trained gradient boosting model
        X_test: Test features
        y_test: Test target
        label_encoder: Label encoder for target variable
    """
    # Neural Network predictions
    nn_pred_proba = nn_model.predict(X_test)
    nn_pred = (nn_pred_proba > 0.5).astype(int)
    
    # Gradient Boosting predictions
    gb_pred_proba = gb_model.predict_proba(X_test)[:, 1]
    gb_pred = (gb_pred_proba > 0.5).astype(int)
    
    print("\nNeural Network Classification Report:")
    print(classification_report(y_test, nn_pred, target_names=label_encoder.classes_))
    
    print("\nGradient Boosting Classification Report:")
    print(classification_report(y_test, gb_pred, target_names=label_encoder.classes_))