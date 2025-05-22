def explore_data(df):
    # Garder cette fonction pour les impressions console et info rapide
    print(f"Dataset shape: {df.shape}")
    print(f"Number of customers: {df.shape[0]}")
    print(f"Number of features: {df.shape[1]}")
    
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("Missing values per column:")
        print(missing_values[missing_values > 0])
    else:
        print("No missing values found in the dataset.")
    
    churn_distribution = df['Churn'].value_counts(normalize=True) * 100
    print("\nChurn Distribution:")
    print(f"No: {churn_distribution.get('no', 0):.2f}%")
    print(f"Yes: {churn_distribution.get('yes', 0):.2f}%")