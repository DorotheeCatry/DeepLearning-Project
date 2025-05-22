import os
import pandas as pd

# Import preprocessing, split, models modules
from src.preprocessing.preprocessing import preprocess_data
from src.utils.split import split_data
from src.models.neural_network import build_model, get_callbacks
from src.models.gradient_boosting import train_gb_model, get_feature_importance

# Import visualization functions
from src.vizualization.vizualization import (
    plot_churn_distribution, plot_correlation_heatmap, plot_tenure_churn_boxplot,
    plot_monthly_charges_churn_boxplot, plot_contract_churn_barplot,
    plot_feature_importance, plot_roc_curves, plot_learning_curves
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
    X_train_processed, X_test_processed, X_val_processed, y_train_encoded, y_val_encoded, y_test_encoded, preprocessor, le = preprocess_data(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    print("\nTraining neural network model...")
    input_dim = X_train_processed.shape[1]
    nn_model = build_model(input_dim)
    callbacks = get_callbacks()
    
    class_weight = {
        0: 1.0,
        1: (y_train == 'no').sum() / (y_train == 'yes').sum()
    }
    
    history = nn_model.fit(
        X_train_processed, y_train_encoded,
        validation_data=(X_val_processed, y_val_encoded),
        epochs=100,
        batch_size=32,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\nTraining gradient boosting model...")
    gb_model = train_gb_model(X_train_processed, y_train_encoded, 
                             X_val_processed, y_val_encoded)
    
    feature_names = []
    for name, transformer, features in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(features)
        elif name == 'cat':
            encoder = transformer.named_steps['onehotencoder']
            if hasattr(encoder, 'get_feature_names_out'):
                feature_names.extend(encoder.get_feature_names_out(features))
    
    importance_df = get_feature_importance(gb_model, feature_names)
    importance_df.to_csv('data/feature_importance_gb.csv', index=False)
    
    plot_feature_importance(importance_df)
    
    print("\nEvaluating models...")
    evaluate_models(nn_model, gb_model, X_test_processed, y_test_encoded, le)
    
    # ROC plots require predicted probabilities
    nn_pred_proba = nn_model.predict(X_test_processed)
    gb_pred_proba = gb_model.predict_proba(X_test_processed)[:, 1]
    plot_roc_curves(y_test_encoded, nn_pred_proba, gb_pred_proba)
    
    plot_learning_curves(history)
    
    nn_model.save("data/churn_model.keras")
    print("Neural Network model saved to data/churn_model.keras")
    
    import joblib
    joblib.dump(preprocessor, "data/preprocessor.pkl")
    print("Preprocessor saved to data/preprocessor.pkl")
    
    print("Pipeline completed successfully!")

if __name__ == '__main__':
    main()
