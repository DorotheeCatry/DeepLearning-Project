import mlflow
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.load import load_data
from utils.preprocessing import preprocess
from utils.split import split_data

import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score, f1_score, recall_score, precision_score


checkpoint_path = "checkpoints/churn_models.keras"

# Load data
df = load_data()

# Split and preprocess datas set
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
X_train_processed, X_test_processed, X_val_processed, y_test_encoding, y_train_encoded, y_val_encoding, pipeline, le = preprocess(X_train, X_val, X_test, y_train, y_val, y_test)
X_train_dense = X_train_processed.toarray() if hasattr(X_train_processed, "toarray") else X_train_processed
X_val_dense = X_val_processed.toarray() if hasattr(X_val_processed, "toarray") else X_val_processed
X_test_dense = X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed

y_train_vector = y_train_encoded.reshape(-1)
y_val_vector = y_val_encoding.reshape(-1)
y_test_vector = y_test_encoding.reshape(-1)
input_dim = X_train_dense.shape[1]

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy', 'recall', 'auc']
)



os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
model_ckpt = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_accuracy',
    mode='max',  
    save_best_only=True,
    verbose=1
)

# Should stay before the with mlflow.....
mlflow.set_experiment("experiment_name")
mlflow.set_tracking_uri("http://127.0.0.1:5000")



with mlflow.start_run(run_name='model_tf_keras'):
    # Training
    history = model.fit(
        X_train_dense, y_train_vector,
        validation_data=(X_val_dense, y_val_vector),
        epochs=20,
        batch_size=32,
        verbose=1,
        callbacks=[model_ckpt]
    )

    # Rating on the test set
    y_pred_proba = model.predict(X_test_dense)
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Metrics
    roc_auc = roc_auc_score(y_test_vector, y_pred_proba)
    f1 = f1_score(y_test_vector, y_pred)
    recall = recall_score(y_test_vector, y_pred)
    precision = precision_score(y_test_vector, y_pred)

    # Log on the metrics
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("precision", precision)

    # Log on hyperparameters
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("epochs", 20)
    mlflow.log_param("batch_size", 32)
    
    # Log on model
    mlflow.tensorflow.log_model(
        model=model,
        artifact_path="model",
        input_example=X_test_dense[:5]
    )

    # Optional : export the model under a pickle file
    import joblib
    joblib.dump(pipeline, "pipeline.pkl")
    mlflow.log_artifact("pipeline.pkl", artifact_path="preprocessing")



# for param_name, param_value in mlflow_dict.params.items():
#             mlflow.log_param(param_name, param_value)

# for artifact_name, artifact_path in mlflow_dict.artifacts.items():
#             mlflow.log_artifact(artifact_path, artifact_name)for artifact_name, artifact_path in mlflow_dict.artifacts.items():
#             mlflow.log_artifact(artifact_path, artifact_name)

# mlflow.tensorflow.log_model(
#     model=model,
#     artifact_path="model",
#     input_example=input_example,
#     registered_model_name=None
# )