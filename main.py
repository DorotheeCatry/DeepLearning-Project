import os
import pandas as pd

# Import preprocessing, split, models modules
from src.preprocessing.preprocessing import preprocess_data
from src.utils.split import split_data
from src.models.gradient_boosting import train_gb_model, get_feature_importance
from src.models.ensemble import train_ensemble

# Import visualization functions
from src.vizualization.vizualization import (
    plot_churn_distribution, plot_correlation_heatmap, plot_tenure_churn_boxplot,
    plot_monthly_charges_churn_boxplot, plot_contract_churn_barplot,
    plot_feature_importance, plot_roc_curves
)

# Import functions externalisées
from src.data.explore import explore_data
from src.models.evaluation import evaluate_models

def main():
    print("Starting Customer Churn Prediction Pipeline...")
    
    os.makedirs('data', exist_ok=True)
    os.makedirs('visualization', exist_ok=True)
    
    print("Loading data...")
    df = pd.read_csv('data/Telco-Customer-Chrun_cleaned.csv')
    
    # Data exploration with visualizations
    print("Exploring data...")
    explore_data(df)
    
    # Visualize distributions and correlations
    plot_churn_distribution(df)
    
    cat_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
    if 'customerID' in cat_cols:
        cat_cols.remove('customerID')
    if 'Churn' in cat_cols:
        cat_cols.remove('Churn')
    
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    plot_correlation_heatmap(df, num_cols)
    plot_tenure_churn_boxplot(df)
    plot_monthly_charges_churn_boxplot(df)
    plot_contract_churn_barplot(df)
    
    print("Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target='Churn')
    
    print("Preprocessing data...")
    (X_train_processed, X_val_processed, X_test_processed,
     y_train_enc, y_val_enc, y_test_enc) = preprocess_data(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    print("\nTraining gradient boosting model...")
    gb_model = train_gb_model(X_train_processed, y_train_enc, X_val_processed, y_val_enc)
    
    print("\nTraining ensemble model...")
    ensemble_model = train_ensemble(
        None,  # Neural network will be created inside train_ensemble
        gb_model,
        X_train_processed,
        y_train_enc,
        X_val_processed,
        y_val_enc
    )
    
    importance_df = get_feature_importance(gb_model, X_train.columns)
    plot_feature_importance(importance_df)
    
    print("\nEvaluating models...")
    evaluate_models(None, gb_model, ensemble_model, X_test_processed, y_test_enc)
    
    # Get predictions for ROC curves
    gb_pred_proba = gb_model.predict_proba(X_test_processed)[:, 1]
    ensemble_pred_proba = ensemble_model.predict_proba(X_test_processed)[:, 1]
    
    # Plot ROC curves
    plot_roc_curves(y_test_enc, None, gb_pred_proba, ensemble_pred_proba)
    
    # Save models
    os.makedirs('data/models', exist_ok=True)
    print("Saving models...")
    
    print("Pipeline completed successfully!")

if __name__ == '__main__':
    main()