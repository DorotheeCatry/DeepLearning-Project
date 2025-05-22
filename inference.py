import os
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf

def load_model_and_preprocessor():
    """
    Load the trained model and preprocessor.
    
    Returns:
        model: Trained TensorFlow model
        preprocessor: Fitted preprocessor
        le: Label encoder for the target variable
    """
    # Check if model and preprocessor exist
    if not os.path.exists('data/churn_model_tf'):
        raise FileNotFoundError("Model not found. Please train the model first.")
    
    if not os.path.exists('data/preprocessor.pkl'):
        raise FileNotFoundError("Preprocessor not found. Please train the model first.")
    
    # Load model
    model = tf.keras.models.load_model('data/churn_model_tf')
    
    # Load preprocessor
    preprocessor = joblib.load('data/preprocessor.pkl')
    
    # Create a simple label encoder for the target
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.classes_ = np.array(['no', 'yes'])
    
    return model, preprocessor, le

def predict_single_customer(customer_data, model=None, preprocessor=None, le=None):
    """
    Make a prediction for a single customer.
    
    Args:
        customer_data: DataFrame with a single customer's data
        model: Trained model (optional, will be loaded if not provided)
        preprocessor: Fitted preprocessor (optional, will be loaded if not provided)
        le: Label encoder for the target variable (optional, will be loaded if not provided)
        
    Returns:
        prediction: Predicted class
        probability: Probability of churn
    """
    # Load model and preprocessor if not provided
    if model is None or preprocessor is None or le is None:
        model, preprocessor, le = load_model_and_preprocessor()
    
    # Preprocess the data
    X_processed = preprocessor.transform(customer_data)
    
    # Make prediction
    probability = model.predict(X_processed)[0][0]
    prediction = le.classes_[int(probability > 0.5)]
    
    return prediction, probability

def batch_predict(file_path, model=None, preprocessor=None, le=None):
    """
    Make predictions for multiple customers from a CSV file.
    
    Args:
        file_path: Path to CSV file with customer data
        model: Trained model (optional, will be loaded if not provided)
        preprocessor: Fitted preprocessor (optional, will be loaded if not provided)
        le: Label encoder for the target variable (optional, will be loaded if not provided)
        
    Returns:
        DataFrame with customer IDs, predictions, and probabilities
    """
    # Load model and preprocessor if not provided
    if model is None or preprocessor is None or le is None:
        model, preprocessor, le = load_model_and_preprocessor()
    
    # Load customer data
    customers = pd.read_csv(file_path)
    
    # Store customer IDs
    customer_ids = customers['customerID'] if 'customerID' in customers.columns else None
    
    # Remove target column if present
    if 'Churn' in customers.columns:
        customers = customers.drop(columns=['Churn'])
    
    # Preprocess the data
    X_processed = preprocessor.transform(customers)
    
    # Make predictions
    probabilities = model.predict(X_processed).flatten()
    predictions = [le.classes_[int(p > 0.5)] for p in probabilities]
    
    # Create results DataFrame
    results = pd.DataFrame({
        'customerID': customer_ids if customer_ids is not None else range(len(predictions)),
        'prediction': predictions,
        'churn_probability': probabilities
    })
    
    return results

if __name__ == "__main__":
    # Example usage
    try:
        model, preprocessor, le = load_model_and_preprocessor()
        print("Model and preprocessor loaded successfully.")
        
        # Check if test data exists
        if os.path.exists('data/WA_Fn-UseC_-Telco-Customer-Churn.csv'):
            # Load a few samples for demonstration
            df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
            sample = df.drop(columns=['Churn']).head(5)
            
            # Make predictions
            for i, row in sample.iterrows():
                customer_data = pd.DataFrame([row])
                prediction, probability = predict_single_customer(customer_data, model, preprocessor, le)
                print(f"Customer {row['customerID']}:")
                print(f"  Prediction: {prediction}")
                print(f"  Churn Probability: {probability:.4f}")
                print()
            
            print("For batch predictions, use the batch_predict() function.")
        else:
            print("Test data not found. Please provide a CSV file with customer data.")
    except Exception as e:
        print(f"Error: {e}")
        print("Please make sure the model has been trained and saved correctly.")