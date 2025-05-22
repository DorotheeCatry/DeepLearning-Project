# Customer Churn Prediction

This project implements an ensemble learning approach combining deep learning and gradient boosting models to predict customer churn for a telecom company.

## Project Overview

Customer churn prediction is a critical task for businesses to identify customers who are likely to discontinue their service. This project uses an ensemble of neural networks and gradient boosting to predict customer churn based on various features such as demographics, service usage, and billing information.

## Project Structure

```
├── data/               # Data directory
│   ├── models/        # Saved models
│   └── logs/         # Training logs
├── visualization/      # Visualizations and plots
├── cleaning/           # Data cleaning scripts
│   └── cleaning_data.py
├── src/               
│   ├── preprocessing/  # Preprocessing modules
│   │   └── preprocessing.py
│   ├── models/        # Model definitions
│   │   ├── neural_network.py
│   │   ├── gradient_boosting.py
│   │   └── ensemble.py
│   ├── utils/         # Utility functions
│   │   ├── load.py
│   │   └── split.py
│   └── visualization/ # Visualization functions
│       └── visualization.py
├── main.py            # Main training script
├── inference.py       # Inference script
├── explanation.txt    # Detailed ML process explanation
└── README.md          # Project documentation
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

## Model Architecture

### Neural Network
- 3 hidden layers (128, 64, 32 units)
- Batch normalization and dropout
- L2 regularization
- Binary cross-entropy loss
- Adam optimizer

### Gradient Boosting
- Optimized hyperparameters via RandomizedSearchCV
- Feature importance analysis
- Early stopping to prevent overfitting

### Ensemble Model
- Soft voting classifier combining neural network and gradient boosting
- Leverages strengths of both models
- Improved robustness and performance
- Automatic hyperparameter optimization

## Data Preprocessing

- Missing value handling in TotalCharges
- Categorical variable encoding
- Numerical feature standardization
- Stratified data splitting (train/validation/test)

## Evaluation Metrics

Models are evaluated using:
- Accuracy
- ROC-AUC
- Precision
- Recall
- F1-score
- Confusion matrix

## Results

The ensemble model achieves:
- Higher ROC-AUC compared to individual models
- Improved robustness
- Better generalization
- More reliable predictions

## Feature Importance

Key predictors of churn:
1. Contract type
2. Tenure
3. Internet service type
4. Monthly charges
5. Payment method

## Inference

To make predictions:

```python
from inference import predict_churn

# Single prediction
prediction = predict_churn(customer_data)

# Batch predictions
predictions = predict_churn(customers_df)
```

## Future Improvements

- Hyperparameter tuning optimization
- Additional feature engineering
- Model interpretability (SHAP values)
- API deployment
- Real-time prediction capabilities