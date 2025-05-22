# Customer Churn Prediction

This project implements a deep learning model to predict customer churn for TelcoNova using TensorFlow.

## Project Structure
```
├── data/               # Data directory (not included in repo)
├── models/            # Saved models
├── src/               # Source code
│   ├── preprocessing.py
│   └── model.py
├── train.py           # Training script
└── requirements.txt   # Project dependencies
```

## Setup and Training

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place the Telco Customer Churn dataset in the `data/` directory

3. Run training:
```bash
python train.py
```

## Model Architecture

- 3 hidden layers (128, 64, 32 units)
- Batch normalization and dropout for regularization
- Binary cross-entropy loss
- Class weights to handle imbalance
- Early stopping and model checkpointing