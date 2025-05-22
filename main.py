import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

# Import custom modules
from utils.load import load_data
from utils.preprocessing import preprocess_data
from utils.split import split_data
from models.neural_network import create_model, get_callbacks

def main():
    """
    Main function to run the customer churn prediction pipeline.
    """
    print("Starting Customer Churn Prediction Pipeline...")
    
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
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Churn', data=df)
    plt.title('Churn Distribution')
    plt.savefig('data/churn_distribution.png')
    
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
    plt.figure(figsize=(12, 10))
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.savefig('data/correlation_heatmap.png')

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
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('data/confusion_matrix.png')
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('data/roc_curve.png')
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig('data/precision_recall_curve.png')

def plot_learning_curves(history):
    """
    Plot learning curves from model training history.
    
    Args:
        history: Keras history object from model training
    """
    # Plot accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('data/learning_curves.png')

if __name__ == "__main__":
    main()