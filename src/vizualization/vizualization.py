import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

def plot_churn_distribution(df):
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='Churn', data=df, palette='viridis')
    plt.title('Churn Distribution', fontsize=15)
    plt.xlabel('Churn', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=12)
    
    plt.savefig('visualization/churn_distribution.png')
    plt.close()

def plot_correlation_heatmap(df, num_cols):
    plt.figure(figsize=(14, 12))
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', mask=mask)
    plt.title('Correlation Matrix of Numerical Features', fontsize=16)
    plt.tight_layout()
    plt.savefig('visualization/correlation_heatmap.png')
    plt.close()

def plot_tenure_churn_boxplot(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Churn', y='tenure', data=df, palette='viridis')
    plt.title('Tenure by Churn Status', fontsize=15)
    plt.savefig('visualization/tenure_churn_boxplot.png')
    plt.close()

def plot_monthly_charges_churn_boxplot(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Churn', y='MonthlyCharges', data=df, palette='viridis')
    plt.title('Monthly Charges by Churn Status', fontsize=15)
    plt.savefig('visualization/monthly_charges_churn_boxplot.png')
    plt.close()

def plot_contract_churn_barplot(df):
    plt.figure(figsize=(12, 6))
    contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
    contract_churn.plot(kind='bar', stacked=True, figsize=(10, 6), color=['green', 'red'])
    plt.title('Churn Rate by Contract Type', fontsize=15)
    plt.xlabel('Contract Type', fontsize=12)
    plt.ylabel('Percentage', fontsize=12)
    plt.savefig('visualization/contract_churn_barplot.png')
    plt.close()

def plot_feature_importance(importance_df, top_n=15):
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance_df.head(top_n), x='importance', y='feature')
    plt.title(f'Top {top_n} Features (Gradient Boosting)', fontsize=12)
    plt.tight_layout()
    plt.savefig('visualization/feature_importance_gb.png')
    plt.close()

def plot_roc_curves(y_test, nn_pred_proba, gb_pred_proba, ensemble_pred_proba):
    plt.figure(figsize=(10, 8))
    
    # Only plot Neural Network ROC if predictions are available
    if nn_pred_proba is not None:
        fpr_nn, tpr_nn, _ = roc_curve(y_test, nn_pred_proba)
        roc_auc_nn = auc(fpr_nn, tpr_nn)
        plt.plot(fpr_nn, tpr_nn, color='blue', lw=2, 
                label=f'Neural Network (AUC = {roc_auc_nn:.2f})')
    
    # Gradient Boosting ROC
    fpr_gb, tpr_gb, _ = roc_curve(y_test, gb_pred_proba)
    roc_auc_gb = auc(fpr_gb, tpr_gb)
    plt.plot(fpr_gb, tpr_gb, color='red', lw=2,
             label=f'Gradient Boosting (AUC = {roc_auc_gb:.2f})')
    
    # Ensemble ROC
    fpr_ens, tpr_ens, _ = roc_curve(y_test, ensemble_pred_proba)
    roc_auc_ens = auc(fpr_ens, tpr_ens)
    plt.plot(fpr_ens, tpr_ens, color='green', lw=2,
             label=f'Ensemble (AUC = {roc_auc_ens:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc="lower right")
    plt.savefig('visualization/roc_curves_comparison.png')
    plt.close()

def plot_learning_curves(history):
    plt.figure(figsize=(16, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy', fontsize=14)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss', fontsize=14)
    plt.ylabel('Loss', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('visualization/learning_curves.png')
    plt.close()
