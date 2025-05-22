import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import RandomizedSearchCV
import joblib

def create_gb_model(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    max_features=None,
    random_state=42
):
    """
    Create a Gradient Boosting classifier with specified hyperparameters.
    """
    class_weights = {0: 1, 1: 2}  # Give more weight to minority class
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        max_features=max_features,
        random_state=random_state
    )

def train_gb_model(X_train, y_train, X_val, y_val, feature_names=None):
    """
    Train a Gradient Boosting model with optimized hyperparameters.
    """
    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500],
        'learning_rate': np.linspace(0.01, 0.3, 20),
        'max_depth': [3, 4, 5, 6],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'max_features': ['sqrt', 'log2', None]
    }
    
    base_model = GradientBoostingClassifier(random_state=42)
    
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=20,
        scoring='roc_auc',
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    search.fit(X_train, y_train)
    
    model = search.best_estimator_
    
    val_pred_proba = model.predict_proba(X_val)[:, 1]
    
    val_auc = roc_auc_score(y_val, val_pred_proba)
    precision, recall, _ = precision_recall_curve(y_val, val_pred_proba)
    pr_auc = auc(recall, precision)
    
    print("\nGradient Boosting Validation Metrics:")
    print(f"ROC AUC: {val_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print("\nBest parameters:", search.best_params_)
    
    os.makedirs('data/models', exist_ok=True)
    joblib.dump(model, 'data/models/gb_model.pkl')
    
    return model

def get_feature_importance(model, feature_names):
    """
    Get feature importances from the gradient boosting model.
    """
    importances = model.feature_importances_
    
    if feature_names is None or len(feature_names) != len(importances):
        feature_names = [f'feature_{i}' for i in range(len(importances))]
    
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    return importance.sort_values(by='importance', ascending=False)