import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing import preprocess_data
from src.model import create_model, get_callbacks
import tensorflow as tf

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

def main():
    # Load data
    df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Preprocess data
    X, y, preprocessor, le = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Calculate class weights
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    class_weights = {0: 1.0, 1: n_neg/n_pos}
    
    # Create and train model
    model = create_model(input_dim=X_train.shape[1])
    callbacks = get_callbacks()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    test_results = model.evaluate(X_test, y_test, verbose=0)
    print("\nTest Results:")
    for metric, value in zip(model.metrics_names, test_results):
        print(f"{metric}: {value:.4f}")
    
    # Save the model and preprocessor
    model.save('models/final_model')
    
if __name__ == "__main__":
    main()