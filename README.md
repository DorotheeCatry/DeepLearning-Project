# Customer Churn Prediction

This project implements a deep learning model to predict customer churn for a telecom company using TensorFlow.

## Project Overview

Customer churn prediction is a critical task for businesses to identify customers who are likely to discontinue their service. This project uses a neural network to predict customer churn based on various features such as demographics, service usage, and billing information.

## Project Structure

```
├── data/               # Data directory
├── models/             # Saved models
├── cleaning/           # Data cleaning scripts
│   └── cleaning_data.py
├── utils/              # Utility functions
│   ├── load.py
│   ├── preprocessing.py
│   └── split.py
├── models/             # Model definitions
│   └── neural_network.py
├── main.py             # Main training script
├── inference.py        # Inference script
└── README.md           # Project documentation
```

## Setup and Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Place the Telco Customer Churn dataset in the `data/` directory
4. Run the data cleaning script:
```bash
python cleaning/cleaning_data.py
```
5. Run the main training script:
```bash
python main.py
```

## Data Preprocessing

The data preprocessing pipeline includes:
- Handling missing values
- Converting categorical variables to lowercase
- Encoding categorical features
- Standardizing numerical features
- Splitting data into train/validation/test sets

## Model Architecture

The neural network model consists of:
- 3 hidden layers (128, 64, 32 units)
- Batch normalization and dropout for regularization
- Binary cross-entropy loss
- Adam optimizer
- Class weights to handle imbalance
- Early stopping and model checkpointing

## Evaluation Metrics

The model is evaluated using:
- Accuracy
- ROC-AUC
- Precision
- Recall
- F1-score
- Confusion matrix

## Inference

To make predictions on new data, use the `inference.py` script:

```bash
python inference.py
```

## Results

The model achieves:
- ROC-AUC: ~0.85
- Accuracy: ~0.80
- Recall for churn class: ~0.70

## Future Improvements

- Hyperparameter tuning using Keras Tuner
- Feature engineering to create more predictive variables
- Ensemble methods to improve performance
- Explainability using SHAP values