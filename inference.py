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
        label_encoder: Fitted label encoder
    """
    model_path = os.path.join('data', 'churn_model_tf')
    preprocessor_path = os.path.join('data', 'preprocessor.pkl')
    
    model = tf.keras.models.load_model(model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    # Create a simple label encoder for the target
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.classes_ = np.array(['no', 'yes'])
    
    return model, preprocessor, le

def predict_churn(customer_data, model, preprocessor, label_encoder):
    """
    Predict churn for a new customer.
    
    Args:
        customer_data: pandas DataFrame with customer data
        model: Trained TensorFlow model
        preprocessor: Fitted preprocessor
        label_encoder: Fitted label encoder
        
    Returns:
        prediction: Churn prediction (yes/no)
        probability: Probability of churn
    """
    # Preprocess the data
    X_processed = preprocessor.transform(customer_data)
    
    # Make prediction
    churn_probability = model.predict(X_processed)[0][0]
    churn_prediction = (churn_probability > 0.5).astype(int)
    
    # Convert prediction to label
    prediction_label = label_encoder.inverse_transform([churn_prediction])[0]
    
    return prediction_label, churn_probability

def main():
    """
    Example of using the model for inference.
    """
    # Load model and preprocessor
    model, preprocessor, label_encoder = load_model_and_preprocessor()
    
    # Example customer data (modify as needed)
    customer_data = pd.DataFrame({
        'gender': ['female'],
        'SeniorCitizen': [0],
        'Partner': ['yes'],
        'Dependents': ['no'],
        'tenure': [24],
        'PhoneService': ['yes'],
        'MultipleLines': ['no'],
        'InternetService': ['fiber optic'],
        'OnlineSecurity': ['no'],
        'OnlineBackup': ['yes'],
        'DeviceProtection': ['no'],
        'TechSupport': ['no'],
        'StreamingTV': ['yes'],
        'StreamingMovies': ['yes'],
        'Contract': ['month-to-month'],
        'PaperlessBilling': ['yes'],
        'PaymentMethod': ['electronic check'],
        'MonthlyCharges': [95.7],
        'TotalCharges': [2283.3]
    })
    
    # Make prediction
    prediction, probability = predict_churn(customer_data, model, preprocessor, label_encoder)
    
    # Print results
    print(f"Churn Prediction: {prediction}")
    print(f"Churn Probability: {probability:.4f}")
    
    # Feature importance analysis (simplified)
    print("\nNote: For a proper feature importance analysis, consider using:")
    print("1. SHAP (SHapley Additive exPlanations) values")
    print("2. Permutation importance")
    print("3. Integrated Gradients")
    print("These methods provide better insights for neural networks than traditional feature importance.")

if __name__ == "__main__":
    main()