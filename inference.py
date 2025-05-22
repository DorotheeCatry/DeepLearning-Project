import os
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

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
    
    # Check if model and preprocessor exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}. Please train the model first.")
    
    model = tf.keras.models.load_model(model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    # Create a simple label encoder for the target
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.classes_ = np.array(['no', 'yes'])
    
    return model, preprocessor, le

def predict_churn(customer_data, model=None, preprocessor=None, label_encoder=None):
    """
    Predict churn for a new customer.
    
    Args:
        customer_data: pandas DataFrame with customer data
        model: Trained TensorFlow model (optional)
        preprocessor: Fitted preprocessor (optional)
        label_encoder: Fitted label encoder (optional)
        
    Returns:
        prediction: Churn prediction (yes/no)
        probability: Probability of churn
        explanation: Dictionary with feature contributions
    """
    # Load model and preprocessor if not provided
    if model is None or preprocessor is None or label_encoder is None:
        model, preprocessor, label_encoder = load_model_and_preprocessor()
    
    # Preprocess the data
    X_processed = preprocessor.transform(customer_data)
    
    # Make prediction
    churn_probability = model.predict(X_processed)[0][0]
    churn_prediction = (churn_probability > 0.5).astype(int)
    
    # Convert prediction to label
    prediction_label = label_encoder.inverse_transform([churn_prediction])[0]
    
    # Create a simple explanation
    explanation = {
        'prediction': prediction_label,
        'probability': float(churn_probability),
        'threshold': 0.5,
        'confidence': float(abs(churn_probability - 0.5) * 2)  # Scale to 0-1
    }
    
    return prediction_label, churn_probability, explanation

def explain_prediction(customer_data, prediction, probability, model=None, preprocessor=None):
    """
    Provide a simple explanation for the prediction.
    
    Args:
        customer_data: pandas DataFrame with customer data
        prediction: Churn prediction (yes/no)
        probability: Probability of churn
        model: Trained TensorFlow model (optional)
        preprocessor: Fitted preprocessor (optional)
        
    Returns:
        explanation_text: Text explanation of the prediction
    """
    # Load model and preprocessor if not provided
    if model is None or preprocessor is None:
        model, preprocessor, _ = load_model_and_preprocessor()
    
    # Get key features that typically influence churn
    key_features = {
        'Contract': customer_data['Contract'].values[0],
        'tenure': customer_data['tenure'].values[0],
        'MonthlyCharges': customer_data['MonthlyCharges'].values[0],
        'InternetService': customer_data['InternetService'].values[0] if 'InternetService' in customer_data.columns else 'N/A',
        'OnlineBackup': customer_data['OnlineBackup'].values[0] if 'OnlineBackup' in customer_data.columns else 'N/A',
        'TechSupport': customer_data['TechSupport'].values[0] if 'TechSupport' in customer_data.columns else 'N/A',
        'PaymentMethod': customer_data['PaymentMethod'].values[0] if 'PaymentMethod' in customer_data.columns else 'N/A'
    }
    
    # Create explanation text
    explanation_text = f"Churn Prediction: {prediction.upper()} (Probability: {probability:.2f})\n\n"
    explanation_text += "Key factors influencing this prediction:\n"
    
    # Contract type is a strong predictor
    if key_features['Contract'] == 'month-to-month':
        explanation_text += "- Month-to-month contract (higher churn risk)\n"
    elif key_features['Contract'] == 'one year':
        explanation_text += "- One-year contract (moderate churn risk)\n"
    elif key_features['Contract'] == 'two year':
        explanation_text += "- Two-year contract (lower churn risk)\n"
    
    # Tenure is a strong predictor
    if key_features['tenure'] < 12:
        explanation_text += "- Short tenure of less than 1 year (higher churn risk)\n"
    elif key_features['tenure'] >= 12 and key_features['tenure'] < 24:
        explanation_text += "- Moderate tenure of 1-2 years (moderate churn risk)\n"
    else:
        explanation_text += "- Long tenure of 2+ years (lower churn risk)\n"
    
    # Monthly charges
    if key_features['MonthlyCharges'] > 80:
        explanation_text += "- High monthly charges (higher churn risk)\n"
    
    # Internet service
    if key_features['InternetService'] == 'fiber optic':
        explanation_text += "- Fiber optic internet service (higher churn risk)\n"
    elif key_features['InternetService'] == 'dsl':
        explanation_text += "- DSL internet service (moderate churn risk)\n"
    
    # Support services
    if key_features['OnlineBackup'] == 'no' or key_features['TechSupport'] == 'no':
        explanation_text += "- Lack of support services (higher churn risk)\n"
    
    # Payment method
    if key_features['PaymentMethod'] == 'electronic check':
        explanation_text += "- Electronic check payment method (higher churn risk)\n"
    
    return explanation_text

def batch_predict(customer_data_path):
    """
    Make predictions for multiple customers from a CSV file.
    
    Args:
        customer_data_path: Path to CSV file with customer data
        
    Returns:
        predictions_df: DataFrame with predictions
    """
    # Load customer data
    customers_df = pd.read_csv(customer_data_path)
    
    # Load model and preprocessor
    model, preprocessor, label_encoder = load_model_and_preprocessor()
    
    # Preprocess the data
    X_processed = preprocessor.transform(customers_df)
    
    # Make predictions
    churn_probabilities = model.predict(X_processed).flatten()
    churn_predictions = (churn_probabilities > 0.5).astype(int)
    
    # Convert predictions to labels
    prediction_labels = label_encoder.inverse_transform(churn_predictions)
    
    # Add predictions to the DataFrame
    customers_df['ChurnPrediction'] = prediction_labels
    customers_df['ChurnProbability'] = churn_probabilities
    
    # Save predictions
    output_path = os.path.join('data', 'churn_predictions.csv')
    customers_df.to_csv(output_path, index=False)
    print(f"Batch predictions saved to {output_path}")
    
    # Create a summary
    summary = {
        'total_customers': len(customers_df),
        'predicted_churn': sum(churn_predictions),
        'churn_rate': sum(churn_predictions) / len(customers_df) * 100
    }
    
    print(f"\nPrediction Summary:")
    print(f"Total customers: {summary['total_customers']}")
    print(f"Predicted to churn: {summary['predicted_churn']} ({summary['churn_rate']:.2f}%)")
    
    # Plot churn probability distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(churn_probabilities, bins=20, kde=True)
    plt.title('Distribution of Churn Probabilities')
    plt.xlabel('Churn Probability')
    plt.ylabel('Count')
    plt.savefig('visualization/churn_probability_distribution.png')
    
    return customers_df

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
    prediction, probability, explanation = predict_churn(customer_data, model, preprocessor, label_encoder)
    
    # Print results
    print(f"Churn Prediction: {prediction}")
    print(f"Churn Probability: {probability:.4f}")
    
    # Get explanation
    explanation_text = explain_prediction(customer_data, prediction, probability, model, preprocessor)
    print("\nExplanation:")
    print(explanation_text)
    
    # Feature importance analysis (simplified)
    print("\nNote: For a proper feature importance analysis, consider using:")
    print("1. SHAP (SHapley Additive exPlanations) values")
    print("2. Permutation importance")
    print("3. Integrated Gradients")
    print("These methods provide better insights for neural networks than traditional feature importance.")

if __name__ == "__main__":
    main()