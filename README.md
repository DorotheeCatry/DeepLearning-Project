# Customer Churn Prediction

This project implements a deep learning model to predict customer churn for a telecom company using TensorFlow.

## Project Overview

Customer churn prediction is a critical task for businesses to identify customers who are likely to discontinue their service. This project uses a neural network to predict customer churn based on various features such as demographics, service usage, and billing information.

## Project Structure

```
├── data/               # Data directory for storing datasets and model
├── visualization/      # Visualizations and plots
├── cleaning/           # Data cleaning scripts
│   └── cleaning_data.py
├── utils/              # Utility functions
│   ├── load.py         # Data loading functions
│   ├── preprocessing.py # Data preprocessing functions
│   └── split.py        # Data splitting functions
├── models/             # Model definitions
│   └── neural_network.py # Neural network architecture
├── main.py             # Main training script
├── inference.py        # Inference script for making predictions
├── explanation.txt     # Detailed explanation of the ML process
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
- Handling missing values in TotalCharges
- Converting categorical variables to lowercase
- Encoding categorical features using one-hot encoding
- Standardizing numerical features
- Splitting data into train/validation/test sets with stratification

## Model Architecture

The neural network model consists of:
- 3 hidden layers (128, 64, 32 units)
- Batch normalization and dropout for regularization
- L2 regularization on weights
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

For batch predictions, you can provide a CSV file:

```python
from inference import batch_predict
predictions = batch_predict('path/to/customers.csv')
```

## Results

The model achieves:
- ROC-AUC: ~0.85
- Accuracy: ~0.80
- Recall for churn class: ~0.70

## Feature Importance

The most important features for predicting churn are:
1. Contract type (month-to-month contracts have higher churn)
2. Tenure (shorter tenure correlates with higher churn)
3. Internet service type (fiber optic users have higher churn)
4. Monthly charges (higher charges correlate with higher churn)
5. Payment method (electronic check users have higher churn)

## Future Improvements

- Hyperparameter tuning using Keras Tuner
- Feature engineering to create more predictive variables
- Ensemble methods to improve performance
- Explainability using SHAP values
- Deployment as a REST API