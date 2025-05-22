import numpy as np
import pandas as pd
from utils.load import load_data
from utils.split import split_data
from utils.preprocessing import preprocess
from models.neural_network import create_model, get_callbacks
import tensorflow as tf

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load and split data
df = load_data()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

# Preprocess data
X_train_processed, X_test_processed, X_val_processed, y_test_encoded, y_train_encoded, y_val_encoded, pipeline, le = preprocess(
    X_train, X_val, X_test, y_train, y_val, y_test
)

# Convert sparse matrices to dense if needed
if isinstance(X_train_processed, np.ndarray):
    X_train_dense = X_train_processed
    X_val_dense = X_val_processed
    X_test_dense = X_test_processed
else:
    X_train_dense = X_train_processed.toarray()
    X_val_dense = X_val_processed.toarray()
    X_test_dense = X_test_processed.toarray()

# Calculate class weights to handle imbalance
n_neg = np.sum(y_train_encoded == 0)
n_pos = np.sum(y_train_encoded == 1)
class_weights = {0: 1.0, 1: n_neg/n_pos}

# Create and train model
model = create_model(input_dim=X_train_dense.shape[1])
callbacks = get_callbacks()

history = model.fit(
    X_train_dense,
    y_train_encoded,
    validation_data=(X_val_dense, y_val_encoded),
    epochs=100,
    batch_size=32,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# Evaluate model
test_results = model.evaluate(X_test_dense, y_test_encoded, verbose=0)
print("\nTest Results:")
for metric, value in zip(model.metrics_names, test_results):
    print(f"{metric}: {value:.4f}")

# Save the model and preprocessing pipeline
model.save('models/final_model')