import mlflow
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.load import load_data
from utils.preprocessing import preprocess
from utils.split import split_data
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, f1_score, recall_score, precision_score


checkpoint_path = "checkpoints/churn_models.keras"

# Load data
df = load_data()

# Split and preprocess datas set
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
X_train_processed, X_test_processed, X_val_processed, y_test_encoding, y_train_encoded, y_val_encoding, pipeline, le = preprocess(X_train, X_val, X_test, y_train, y_val, y_test)


# Recombine train + val for final training
from scipy.sparse import vstack
X_combined = vstack([X_train_processed, X_val_processed])
y_combined = np.concatenate([y_train_encoded, y_val_encoding])


# Should stay before the with mlflow.....
mlflow.set_experiment("logistic regression")
mlflow.set_tracking_uri("http://127.0.0.1:5000")



with mlflow.start_run(run_name='model_logistic_regression'):
    
    # Logistic regression model
    model = LogisticRegression(max_iter=1000, C=1.0, penalty='l2', solver='liblinear')
    model.fit(X_combined, y_combined)

    # Predict
    y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
    y_pred = model.predict(X_test_processed)
    

    # Metrics
    roc_auc = roc_auc_score(y_test_encoding, y_pred_proba)
    f1 = f1_score(y_test_encoding, y_pred)
    recall = recall_score(y_test_encoding, y_pred)
    precision = precision_score(y_test_encoding, y_pred)

    # Log on the metrics
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("precision", precision)

    # Log on hyperparameters
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("penalty", "l2")
    mlflow.log_param("C", 1.0)
    mlflow.log_param("solver", "liblinear")
    mlflow.log_param("max_iter", 1000)
    
    

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