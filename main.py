import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.load import load_data
from utils.preprocessing import preprocess
from utils.split import split_data
from utils.objective_function import create_objective

import optuna
import matplotlib.pyplot as plt
from optuna.visualization import plot_optimization_history, plot_param_importances
import joblib
import tensorflow
from datetime import datetime
from sklearn.utils import class_weight
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam, RMSprop



def main():
    # Create dynamic name for experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"experiment_{timestamp}"

    # Load the data
    df = load_data()

    # Split and preprocess datas set
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    X_train_processed, X_test_processed, X_val_processed, y_test_encoding, y_train_encoded, y_val_encoding, pipeline, le = preprocess(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Transformation of dataframes in array
    X_train_dense = X_train_processed.toarray() if hasattr(X_train_processed, "toarray") else X_train_processed
    X_val_dense = X_val_processed.toarray() if hasattr(X_val_processed, "toarray") else X_val_processed
    X_test_dense = X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed
    y_train_vector = y_train_encoded.reshape(-1)
    y_val_vector = y_val_encoding.reshape(-1)
    y_test_vector = y_test_encoding.reshape(-1)
    
    # Create the objective function
    objective = create_objective(X_train_dense, y_train_vector, X_val_dense, y_val_vector)

    # Set seed to have reproductibility
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', storage="sqlite:///notebooks/db.sqlite3",sampler=sampler, study_name=study_name)
    study.optimize(objective, n_trials=20)

    # Save the best model
    best_params = study.best_params
    print("Best trial:")
    print("  Recall:", study.best_value)
    print("  Params:", best_params)

    # Training of the best model
    optimizer = {
        'adam': Adam(learning_rate=best_params['learning_rate']),
        'rmsprop': RMSprop(learning_rate=best_params['learning_rate'])
    }[best_params['optimizer']]

    final_model = Sequential()
    final_model.add(Input(shape=(X_train_dense.shape[1],)))
    for _ in range(best_params['n_layers']):
        final_model.add(Dense(best_params['n_units'], activation=best_params['activation']))
        final_model.add(Dropout(best_params['dropout_rate']))
    final_model.add(Dense(1, activation='sigmoid'))

    final_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'recall'])

    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_vector),
        y=y_train_vector
    )
    class_weights = dict(enumerate(class_weights))

    final_model.fit(
        X_train_dense, y_train_vector,
        validation_data=(X_val_dense, y_val_vector),
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        class_weight=class_weights,
        verbose=1
    )

    # Save the model
    final_model.save("notebooks/checkpoints/best_churn_model.keras")

    # Rating on the test dataset
    y_test_pred = (final_model.predict(X_test_dense) > 0.5).astype(int)
    from sklearn.metrics import classification_report
    print("Classification Report on Test Set:")
    print(classification_report(y_test_vector, y_test_pred))

    # Visualization matplotlib
    recalls = [trial.value for trial in study.trials if trial.value is not None]
    plt.plot(recalls)
    plt.title("Recall per Trial")
    plt.xlabel("Trial")
    plt.ylabel("Recall")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/recall_per_trial.png")
    plt.show()

    # Visualizations with Optuna
    try:
        import plotly.io as pio
        pio.renderers.default = "svg"  
        os.makedirs("plots", exist_ok=True)

        from optuna.visualization import plot_optimization_history, plot_param_importances

        fig1 = plot_optimization_history(study)
        fig1.write_html("plots/optuna_history.html")

        fig2 = plot_param_importances(study)
        fig2.write_html("plots/optuna_importances.html")

        print(" Figures saved in plots/.")

    except Exception as e:
        print("Impossible to get the visualizations")
        print(f"Error : {e}")


if __name__ == "__main__":
    main()