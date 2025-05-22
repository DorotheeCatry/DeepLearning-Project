import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

# Import custom modules
from src.utils.load import load_data
from src.preprocessing.preprocessing import preprocess_data
from src.utils.split import split_data
from src.models.neural_network import create_model, get_callbacks
from src.models.gradient_boosting import train_gb_model, get_feature_importance

def main():
    """
    Main function to run the customer churn prediction pipeline.
    """
    print("Starting Customer Churn Prediction Pipeline...")
    
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('visualization', exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = load_data()
    
    # Data exploration
    print("Exploring data...")
    explore_data(df)
    
    # Split data
    print("Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target='Churn')
    
    # Preprocess data
    print("Preprocessing data...")
    X_train_processed, X_test_processed, X_val_processed, y_train_encoded, y_val_encoded, y_test_encoded, preprocessor, le = preprocess_data(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    # Train Neural Network
    print("\nTraining neural network model...")
    input_dim = X_train_processed.shape[1]
    nn_model = create_model(input_dim)
    callbacks = get_callbacks()
    
    # Calculate class weights
    class_weight = {
        0: 1.0,
        1: (y_train == 'no').sum() / (y_train == 'yes').sum()
    }
    
    # Train the neural network
    history = nn_model.fit(
        X_train_processed, y_train_encoded,
        validation_data=(X_val_processed, y_val_encoded),
        epochs=100,
        batch_size=32,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )
    
    # Train Gradient Boosting
    print("\nTraining gradient boosting model...")
    gb_model = train_gb_model(X_train_processed, y_train_encoded, 
                             X_val_processed, y_val_encoded)
    
    # Get feature importance from GB model
    feature_names = []
    for name, transformer, features in preprocessor.transformers_:
        if hasattr(transformer, 'get_feature_names_out'):
            feature_names.extend(transformer.named_steps['onehot'].get_feature_names_out(features))
        else:
            feature_names.extend(features)
    
    importance_df = get_feature_importance(gb_model, feature_names)
    importance_df.to_csv('data/feature_importance_gb.csv', index=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance_df.head(15), x='importance', y='feature')
    plt.title('Top 15 Features (Gradient Boosting)', fontsize=12)
    plt.tight_layout()
    plt.savefig('visualization/feature_importance_gb.png')
    
    # Evaluate models
    print("\nEvaluating models...")
    evaluate_models(nn_model, gb_model, X_test_processed, y_test_encoded, le)
    
    # Plot learning curves for neural network
    plot_learning_curves(history)
    
    # Save the models
    nn_model.save("data/churn_model_tf")
    print("Neural Network model saved to data/churn_model_tf")
    
    # Save the preprocessor
    import joblib
    joblib.dump(preprocessor, "data/preprocessor.pkl")
    print("Preprocessor saved to data/preprocessor.pkl")
    
    print("Pipeline completed successfully!")

def evaluate_models(nn_model, gb_model, X_test, y_test, label_encoder):
    """
    Evaluate both neural network and gradient boosting models.
    """
    # Neural Network predictions
    nn_pred_proba = nn_model.predict(X_test)
    nn_pred = (nn_pred_proba > 0.5).astype(int)
    
    # Gradient Boosting predictions
    gb_pred_proba = gb_model.predict_proba(X_test)[:, 1]
    gb_pred = (gb_pred_proba > 0.5).astype(int)
    
    # Print classification reports
    print("\nNeural Network Classification Report:")
    print(classification_report(y_test, nn_pred, target_names=label_encoder.classes_))
    
    print("\nGradient Boosting Classification Report:")
    print(classification_report(y_test, gb_pred, target_names=label_encoder.classes_))
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    
    # Neural Network ROC
    fpr_nn, tpr_nn, _ = roc_curve(y_test, nn_pred_proba)
    roc_auc_nn = auc(fpr_nn, tpr_nn)
    plt.plot(fpr_nn, tpr_nn, color='blue', lw=2, 
             label=f'Neural Network (AUC = {roc_auc_nn:.2f})')
    
    # Gradient Boosting ROC
    fpr_gb, tpr_gb, _ = roc_curve(y_test, gb_pred_proba)
    roc_auc_gb = auc(fpr_gb, tpr_gb)
    plt.plot(fpr_gb, tpr_gb, color='red', lw=2, 
             label=f'Gradient Boosting (AUC = {roc_auc_gb:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc="lower right")
    plt.savefig('visualization/roc_curves_comparison.png')

def explore_data(df):
    """
    Perform exploratory data analysis on the dataset.
    
    Args:
        df: pandas DataFrame with the data
    """
    # Basic info
    print(f"Dataset shape: {df.shape}")
    print(f"Number of customers: {df.shape[0]}")
    print(f"Number of features: {df.shape[1]}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("Missing values per column:")
        print(missing_values[missing_values > 0])
    else:
        print("No missing values found in the dataset.")
    
    # Check class distribution
    churn_distribution = df['Churn'].value_counts(normalize=True) * 100
    print("\nChurn Distribution:")
    print(f"No: {churn_distribution['no']:.2f}%")
    print(f"Yes: {churn_distribution['yes']:.2f}%")
    
    # Save distribution plot
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='Churn', data=df, palette='viridis')
    plt.title('Churn Distribution', fontsize=15)
    plt.xlabel('Churn', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Add count labels
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'bottom', 
                    fontsize=12)
    
    plt.savefig('visualization/churn_distribution.png')
    
    # Analyze categorical features
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    cat_cols.remove('customerID')  # Remove ID column
    if 'Churn' in cat_cols:
        cat_cols.remove('Churn')  # Remove target column
    
    # Analyze numerical features
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"\nCategorical features: {len(cat_cols)}")
    print(f"Numerical features: {len(num_cols)}")
    
    # Save correlation heatmap for numerical features
    plt.figure(figsize=(14, 12))
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', mask=mask)
    plt.title('Correlation Matrix of Numerical Features', fontsize=16)
    plt.tight_layout()
    plt.savefig('visualization/correlation_heatmap.png')
    
    # Analyze relationship between tenure and churn
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Churn', y='tenure', data=df, palette='viridis')
    plt.title('Tenure by Churn Status', fontsize=15)
    plt.savefig('visualization/tenure_churn_boxplot.png')
    
    # Analyze relationship between monthly charges and churn
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Churn', y='MonthlyCharges', data=df, palette='viridis')
    plt.title('Monthly Charges by Churn Status', fontsize=15)
    plt.savefig('visualization/monthly_charges_churn_boxplot.png')
    
    # Analyze relationship between contract type and churn
    plt.figure(figsize=(12, 6))
    contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
    contract_churn.plot(kind='bar', stacked=True, figsize=(10, 6), color=['green', 'red'])
    plt.title('Churn Rate by Contract Type', fontsize=15)
    plt.xlabel('Contract Type', fontsize=12)
    plt.ylabel('Percentage', fontsize=12)
    plt.savefig('visualization/contract_churn_barplot.png')

def plot_learning_curves(history):
    """
    Plot learning curves from model training history.
    
    Args:
        history: Keras history object from model training
    """
    # Plot accuracy
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy', fontsize=14)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss', fontsize=14)
    plt.ylabel('Loss', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('visualization/learning_curves.png')

if __name__ == "__main__":
    main()