# %%
import pandas as pd
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import StringLookup, CategoryEncoding, Normalization, Lambda, Concatenate
from tensorflow.keras.models import Model

# %%
data = pd.read_csv('data/cleaned_datas.csv')
df = data.copy()
df

# %%
df.columns

# %%
unique_values = pd.DataFrame({
    'Variable': df.columns,
    'Unique Values': [list(df[col].unique()) for col in df.columns]
})


# %%
unique_values

# %%
# Variables
id_col = "customerID"
target_col = "Churn"

# Colonnes numériques (à normaliser)
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
for col in df[numeric_features]:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df.fillna({col: 0}, inplace=True)

# Colonnes binaires (yes/no à encoder comme int)
binary_features = [
    "gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"
]
binary_numeric_features = ['SeniorCitizen']

# Colonnes catégorielles avec 3+ modalités (one-hot encoding)
multiclass_features = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod"
]

# %%
def get_preprocessing_layers(df):
    inputs = {}
    encoded_features = []
    encoded_column_names = []

    # Variables numériques
    for feature in numeric_features:
        inputs[feature] = tf.keras.Input(shape=(1,), name=feature)
        normalizer = Normalization()
        normalizer.adapt(np.array(df[feature]).reshape(-1, 1))
        encoded = normalizer(inputs[feature])
        encoded_features.append(encoded)
        encoded_column_names.append(feature)
        # print(f"Numeric feature '{feature}' -> 1 column")

    binary_vocab = {
    "gender": ["female", "male"],
    "Partner": ["yes", "no"],
    "Dependents": ["yes", "no"],
    "PhoneService": ["yes", "no"],
    "PaperlessBilling": ["yes", "no"],
    }
    # Variables binaires texte (ex: yes/no)
    for feature in binary_features:
        inputs[feature] = tf.keras.Input(shape=(1,), dtype=tf.string, name=feature)
        vocab = binary_vocab.get(feature, ["no", "yes"])  # fallback si pas dans dict
        lookup = StringLookup(vocabulary=vocab, output_mode="int", oov_token=None)
        encoded_int = lookup(inputs[feature])
        encoded = Lambda(lambda x: tf.cast(x - 1, tf.float32))(encoded_int)
        encoded_features.append(encoded)
        encoded_column_names.append(feature)
        # print(f"Binary feature '{feature}' -> 1 column")

    # Variables binaires numériques (0/1)
    for feature in binary_numeric_features:
        inputs[feature] = tf.keras.Input(shape=(1,), dtype=tf.int32, name=feature)
        encoded = Lambda(lambda x: tf.cast(x, tf.float32))(inputs[feature])
        encoded_features.append(encoded)
        encoded_column_names.append(feature)
        # print(f"Binary numeric feature '{feature}' -> 1 column")

    # Variables multiclasses (one-hot)
    for feature in multiclass_features:
        inputs[feature] = tf.keras.Input(shape=(1,), dtype=tf.string, name=feature)
        lookup = StringLookup(output_mode="int", oov_token="[UNK]")
        lookup.adapt(df[feature])
        encoded_int = lookup(inputs[feature])
        one_hot = CategoryEncoding(num_tokens=lookup.vocabulary_size(), output_mode="one_hot")
        encoded = one_hot(encoded_int)
        encoded_features.append(encoded)

        vocab = lookup.get_vocabulary()
        # Exclusion possible du token OOV si souhaité ici
        # Exemple : col_names = [f"{feature}_{cat}" for cat in vocab if cat != "[UNK]"]
        col_names = [f"{feature}_{cat}" for cat in vocab]
        encoded_column_names.extend(col_names)
        # print(f"Multiclass feature '{feature}' -> {len(col_names)} columns")

    concatenated = Concatenate()(encoded_features)
    preprocessing_model = Model(inputs=inputs, outputs=concatenated, name="preprocessing")

    return preprocessing_model, inputs, encoded_features, encoded_column_names


# %%
# Traitement à part pour la cible
target_lookup = tf.keras.layers.StringLookup(vocabulary=["no", "yes"], output_mode="int")
y = target_lookup(df[target_col].values) - 1  # tenseur ou numpy array

# %%
preprocessing_model, inputs, encoded_features, encoded_column_names = get_preprocessing_layers(df)

# %%
preprocessing_model

# %%
# Utilise le preprocessing_model déjà construit et les variables existantes
df_inputs = {name: df[name].values for name in inputs}
encoded_array = preprocessing_model.predict(df_inputs, verbose=0)

# Si encoded_array est une liste de tableaux, concatène-les
if isinstance(encoded_array, list):
    encoded_concat = np.concatenate(encoded_array, axis=1)
else:
    encoded_concat = encoded_array

# Crée le DataFrame avec les bons noms de colonnes
encoded_df = pd.DataFrame(encoded_concat, columns=encoded_column_names)


# %%
encoded_df

# %%
encoded_df.to_csv('data/encoded_df.csv', index=False)
# Sauvegarde du modèle de prétraitement Keras
preprocessing_model.save('data/preprocessing_model.keras')
# %% [markdown]
# ### **Keras Normalization Layer**
# La couche tf.keras.layers.Normalization est une couche de prétraitement qui permet de normaliser les données numériques en apprenant les statistiques (moyenne, écart-type) du jeu de données.
# 
# ***Fonctionnement***
# - Adaptation : la méthode .adapt(data) calcule la moyenne et l’écart-type à partir des données fournies.
# - Transformation : lors de l’appel de la couche, elle normalise les données en appliquant la formule classique de normalisation z-score.
# 
# ***Pourquoi utiliser cette couche ?***
# - Robustesse : la normalisation est un prérequis important pour de nombreux algorithmes d’apprentissage, notamment les réseaux neuronaux, car elle facilite la convergence et stabilise l’entraînement.
# - Automatisation : Keras propose une couche intégrée qui peut être insérée directement dans un pipeline, ce qui rend la normalisation réutilisable et intégrée dans un modèle TensorFlow/Keras.
# 
# ***Limites / Points d’attention***
# La normalisation dépend des données d’adaptation : si les statistiques changent (par exemple en production), il faudra re-adapter ou gérer dynamiquement.
# 
# Adaptée pour des données numériques continues, pas pour les données catégorielles.
# 
# 
# ### **Keras StringLookup + CategoryEncoding**
# **StringLookup**
# Sert à transformer des chaînes de caractères en indices entiers (encodage numérique).
# Permet de gérer un vocabulaire dynamique (adapté au jeu de données).
# 
# Peut gérer un token hors-vocabulaire (OOV) pour les valeurs inconnues.
# 
# **CategoryEncoding**
# Transforme les indices entiers en représentations one-hot ou multi-hot.
# Utile pour représenter des variables catégorielles dans un format exploitable par les modèles de machine learning.
# 
# ***Pourquoi utiliser cette combinaison ?***
# - Modularité : chaque étape (lookup puis encodage) est claire, paramétrable et adaptée à la manipulation de données catégorielles dans TensorFlow.
# - Performance : intégré à TensorFlow, ce processus est optimisé pour le calcul sur GPU et la compatibilité avec les modèles Keras.
# - Sécurité : gestion explicite des tokens inconnus évite les erreurs liées aux catégories non vues en entraînement.
# 
# Alternatives classiques
# pandas.get_dummies() ou sklearn.OneHotEncoder : hors TensorFlow, plus simple pour un usage hors pipeline TensorFlow.
# tf.keras.layers.experimental.preprocessing.StringLookup et CategoryEncoding sont privilégiés dans un workflow Keras/TF pour garder la cohérence dans la pipeline et le modèle.


