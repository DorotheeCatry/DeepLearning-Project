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
from models.rna_model import create_model, get_callbacks

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
    
    # Create and train model
    print("Training neural network model...")
    input_dim = X_train_processed.shape[1]
    model = create_model(input_dim)
    callbacks = get_callbacks()
    
    # Calculate class weights to handle imbalance
    class_weight = {
        0: 1.0,
        1: (y_train == 'no').sum() / (y_train == 'yes').sum()
    }
    
    # Train the model
    history = model.fit(
        X_train_processed, y_train_encoded,
        validation_data=(X_val_processed, y_val_encoded),
        epochs=100,
        batch_size=32,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("Evaluating model...")
    evaluate_model(model, X_test_processed, y_test_encoded, le)
    
    # Plot learning curves
    plot_learning_curves(history)
    
    # Feature importance analysis
    print("Analyzing feature importance...")
    feature_importance_analysis(model, preprocessor, X_test_processed)
    
    # Save the model
    model.save("data/churn_model_tf")
    print("Model saved to data/churn_model_tf")
    
    # Save the preprocessor
    import joblib
    joblib.dump(preprocessor, "data/preprocessor.pkl")
    print("Preprocessor saved to data/preprocessor.pkl")
    
    print("Pipeline completed successfully!")

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

def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluate the model performance on test data.
    
    Args:
        model: Trained Keras model
        X_test: Processed test features
        y_test: Encoded test target
        label_encoder: Label encoder for the target variable
    """
    # Predict probabilities
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('visualization/confusion_matrix.png')
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('visualization/roc_curve.png')
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig('visualization/precision_recall_curve.png')

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

def feature_importance_analysis(model, preprocessor, X_test):
    """
    Analyze feature importance using a permutation-based approach.
    
    Args:
        model: Trained model
        preprocessor: Fitted preprocessor
        X_test: Processed test features
    """
    try:
        from sklearn.inspection import permutation_importance
        import tensorflow as tf
        
        # Create a wrapper function for the model prediction
        def model_predict(X):
            return model.predict(X)
        
        # Calculate permutation importance
        result = permutation_importance(
            model_predict, X_test, np.argmax(model.predict(X_test), axis=1),
            n_repeats=10, random_state=42, n_jobs=-1
        )
        
        # Get feature names
        feature_names = []
        for name, transformer, features in preprocessor.transformers_:
            if hasattr(transformer, 'get_feature_names_out'):
                feature_names.extend(transformer.get_feature_names_out(features))
            else:
                feature_names.extend(features)
        
        # Create a DataFrame with feature importances
        importance_df = pd.DataFrame({
            'Feature': feature_names[:len(result.importances_mean)],
            'Importance': result.importances_mean
        }).sort_values('Importance', ascending=False)
        
        # Plot top 15 features
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), palette='viridis')
        plt.title('Feature Importance (Permutation-based)', fontsize=15)
        plt.tight_layout()
        plt.savefig('visualization/feature_importance.png')
        
        # Save feature importance to CSV
        importance_df.to_csv('data/feature_importance.csv', index=False)
        print("Feature importance analysis completed and saved.")
        
    except Exception as e:
        print(f"Could not perform feature importance analysis: {e}")
        print("Consider using SHAP or other methods for neural network interpretability.")

if __name__ == "__main__":
    main()