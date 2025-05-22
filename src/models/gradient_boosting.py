# gradient_boosting.py

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

def train_gb_model(X_train, y_train, X_val, y_val):
    """
    Train a Gradient Boosting model with default hyperparameters, evaluate and save it.
    
    Args:
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (pd.Series or np.ndarray): Training target.
        X_val (pd.DataFrame or np.ndarray): Validation features.
        y_val (pd.Series or np.ndarray): Validation target.
        
    Returns:
        Trained Gradient Boosting model.
    """
    model = create_gb_model()
    model.fit(X_train, y_train)
    
    # Predict probabilities on validation set
    val_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Compute metrics
    val_auc = roc_auc_score(y_val, val_pred_proba)
    precision, recall, _ = precision_recall_curve(y_val, val_pred_proba)
    pr_auc = auc(recall, precision)
    
    print("\nGradient Boosting Validation Metrics:")
    print(f"ROC AUC: {val_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    
    # Save model
    os.makedirs('data/models', exist_ok=True)
    joblib.dump(model, 'data/models/gb_model.pkl')
    
    return model

def optimize_gb_model(X_train, y_train, X_val, y_val, n_iter=50, random_state=42):
    """
    Optimize Gradient Boosting hyperparameters using RandomizedSearchCV.
    
    Args:
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (pd.Series or np.ndarray): Training target.
        X_val (pd.DataFrame or np.ndarray): Validation features.
        y_val (pd.Series or np.ndarray): Validation target.
        n_iter (int): Number of parameter settings sampled.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        Best trained Gradient Boosting model.
    """
    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500],
        'learning_rate': np.linspace(0.01, 0.3, 30),
        'max_depth': [3, 4, 5, 6, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.6, 0.8, 1.0],
        'max_features': [None, 'auto', 'sqrt', 'log2']
    }
    
    base_model = GradientBoostingClassifier(random_state=random_state)
    
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='roc_auc',
        cv=3,
        verbose=2,
        random_state=random_state,
        n_jobs=-1
    )
    
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    
    # Evaluate on validation set
    val_pred_proba = best_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred_proba)
    precision, recall, _ = precision_recall_curve(y_val, val_pred_proba)
    pr_auc = auc(recall, precision)
    
    print("\nBest Gradient Boosting Model Validation Metrics:")
    print(f"ROC AUC: {val_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print(f"Best Hyperparameters: {search.best_params_}")
    
    # Save best model
    os.makedirs('data/models', exist_ok=True)
    joblib.dump(best_model, 'data/models/gb_best_model.pkl')
    
    return best_model

def get_feature_importance(model, feature_names):
    """
    Retrieve and sort feature importances from a trained Gradient Boosting model.
    
    Args:
        model (GradientBoostingClassifier): Trained model.
        feature_names (list of str): Feature names.
        
    Returns:
        pd.DataFrame: Features and their importance sorted descendingly.
    """
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    
    return importance.sort_values(by='importance', ascending=False)
