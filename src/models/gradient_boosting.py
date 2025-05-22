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
    
    Args:
        n_estimators (int): Number of boosting stages.
        learning_rate (float): Shrinkage factor.
        max_depth (int): Maximum tree depth.
        min_samples_split (int): Minimum samples to split a node.
        min_samples_leaf (int): Minimum samples in a leaf.
        subsample (float): Fraction of samples for fitting individual trees.
        max_features (int, str or None): Number of features to consider at each split.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        GradientBoostingClassifier instance
    """
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
    
    Args:
        X_train (np.ndarray): Training features (preprocessed).
        y_train (np.ndarray): Training target (encoded).
        X_val (np.ndarray): Validation features (preprocessed).
        y_val (np.ndarray): Validation target (encoded).
        feature_names (list, optional): List of feature names.
        
    Returns:
        Trained Gradient Boosting model.
    """
    # Define parameter space for optimization
    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500],
        'learning_rate': np.linspace(0.01, 0.3, 20),
        'max_depth': [3, 4, 5, 6],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Create base model
    base_model = GradientBoostingClassifier(random_state=42)
    
    # Initialize RandomizedSearchCV
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=20,  # Number of parameter settings sampled
        scoring='roc_auc',
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    # Fit RandomizedSearchCV
    search.fit(X_train, y_train)
    
    # Get best model
    model = search.best_estimator_
    
    # Predict probabilities on validation set
    val_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Compute metrics
    val_auc = roc_auc_score(y_val, val_pred_proba)
    precision, recall, _ = precision_recall_curve(y_val, val_pred_proba)
    pr_auc = auc(recall, precision)
    
    print("\nGradient Boosting Validation Metrics:")
    print(f"ROC AUC: {val_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print("\nBest parameters:", search.best_params_)
    
    # Save model
    os.makedirs('data/models', exist_ok=True)
    joblib.dump(model, 'data/models/gb_model.pkl')
    
    return model

def get_feature_importance(model, feature_names):
    """
    Get feature importances from the gradient boosting model.
    
    Args:
        model (GradientBoostingClassifier): Trained model.
        feature_names (list): List of feature names.
        
    Returns:
        pd.DataFrame: Features and their importance sorted descendingly.
    """
    importances = model.feature_importances_
    
    # If feature_names is not provided or lengths don't match, use generic names
    if feature_names is None or len(feature_names) != len(importances):
        feature_names = [f'feature_{i}' for i in range(len(importances))]
    
    # Create feature importance DataFrame
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    return importance.sort_values(by='importance', ascending=False)