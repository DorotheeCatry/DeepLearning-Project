import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight
from sklearn.metrics import recall_score
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential

def create_objective(X_train_dense, y_train_vector, X_val_dense, y_val_vector):
    def objective(trial):
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        epochs = trial.suggest_int('epochs', 10, 50)
        n_layers = trial.suggest_int('n_layers', 1, 3)
        n_units = trial.suggest_int('n_units', 32, 128, step=32)
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
        activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
        optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])

        optimizer = {
            'adam': Adam(learning_rate=learning_rate),
            'rmsprop': RMSprop(learning_rate=learning_rate)
        }[optimizer_name]

        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train_vector),
            y=y_train_vector
        )
        class_weights = dict(enumerate(class_weights))

        model = Sequential()
        model.add(Input(shape=(X_train_dense.shape[1],)))
        for _ in range(n_layers):
            model.add(Dense(n_units, activation=activation))
            model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['recall'])

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='recall', mode='max', patience=5,
            restore_best_weights=True, verbose=0
        )

        model.fit(
            X_train_dense, y_train_vector,
            validation_data=(X_val_dense, y_val_vector),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[early_stop],
            class_weight=class_weights
        )

        y_pred = (model.predict(X_val_dense) > 0.5).astype(int)
        recall = recall_score(y_val_vector, y_pred)
        return recall

    return objective
