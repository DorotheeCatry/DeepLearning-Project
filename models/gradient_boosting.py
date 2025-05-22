import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import joblib
import os

def create_gb_model():
    """
    Create a Gradient Boosting model for churn prediction.
    
    Returns:
        Gradient Boosting classifier
    """
    return GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )

def train_gb_model(X_train, y_train, X_val, y_val):
    """
    Train the Gradient Boosting model.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        
    Returns:
        Trained Gradient Boosting model
    """
    # Create and train the model
    model = create_gb_model()
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    val_pred_proba = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred_proba)
    
    # Calculate PR AUC
    precision, recall, _ = precision_recall_curve(y_val, val_pred_proba)
    pr_auc = auc(recall, precision)
    
    print(f"\nGradient Boosting Validation Metrics:")
    print(f"ROC AUC: {val_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    
    # Save the model
    os.makedirs('data/models', exist_ok=True)
    joblib.dump(model, 'data/models/gb_model.pkl')
    
    return model

def get_feature_importance(model, feature_names):
    """
    Get feature importance from the Gradient Boosting model.
    
    Args:
        model: Trained Gradient Boosting model
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importance scores
    """
    import pandas as pd
    
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    
    return importance.sort_values('importance', ascending=False)