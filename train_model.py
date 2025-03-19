import pandas as pd
import numpy as np
import pickle
import json
from scipy.stats import linregress
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import shap
import os

def rolling_slope(series):
    """Calculate the slope of a linear regression over a rolling window."""
    if len(series) < 2:
        return None
    x = range(len(series))
    slope, _, _, _, _ = linregress(x, series)
    return slope

def preprocess_churn_data(churn_filepath, cpi_filepath, output_filepath):
    """
    Preprocess the churn dataset by encoding categorical variables, computing lag features,
    rolling means, and trend slopes.
    
    Parameters:
        churn_filepath (str): Path to the churn dataset CSV.
        cpi_filepath (str): Path to the CPI dataset CSV.
        output_filepath (str): Path to save the processed dataset.
    
    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    # Load churn data
    df = pd.read_csv(churn_filepath)
    
    # Convert date columns to datetime
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['issuing_date'] = pd.to_datetime(df['issuing_date'], format='%Y-%m-%d')
    
    # Extract month and calculate months since issuing
    df['month'] = df['date'].dt.month
    df['months_since_issuing'] = (df['date'].dt.year - df['issuing_date'].dt.year) * 12 + \
                                  (df['date'].dt.month - df['issuing_date'].dt.month)
    
    # Encode plan_type
    plan_mapping = {'Basic': 1, 'Standard': 2, 'Premium': 3}
    df['plan_type_encoded'] = df['plan_type'].map(plan_mapping)
    
    # Load CPI data
    df_cpi = pd.read_csv(cpi_filepath)
    df_cpi['month'] = df_cpi['Period'].apply(lambda x: int(x.replace('M', '')))

    # Merge CPI data with churn data
    df = pd.merge(df, df_cpi[['month', 'Value']], how='left', on=['month'])
    df.rename(columns={'Value': 'cpi'}, inplace=True)
    
    # List of features for lag, rolling mean, and slope computations
    features = ['transaction_amount', 'plan_type_encoded', 'cpi']
    
    for feature in features:
        # Compute lag of 1 month
        df[f'{feature}_lag_1m'] = df.groupby('customer_id')[feature].shift(1)
        
        # Compute rolling mean over previous 3 months (excluding current month)
        df[f'{feature}_mean_prev_3m'] = (
            df.groupby('customer_id')[feature]
            .shift(1)  # Exclude current month
            .rolling(window=3, min_periods=1)
            .mean()
        )
        
        # Compute rolling slope (trend) over previous 3 months (excluding current month)
        df[f'{feature}_slope_3m'] = (
            df.groupby('customer_id')[feature]
            .shift(1)  # Exclude current month
            .rolling(window=3, min_periods=2)
            .apply(rolling_slope, raw=True)
        )
    
    # Save processed data
    df.to_csv(output_filepath, index=False)
    
    return df

def train_churn_model(df, model_filepath, scaler_filepath, metrics_filepath, feature_list_filepath, shap_filepath):
    """
    Train a logistic regression model for churn prediction, explain the model using SHAP,
    and save the model, scaler, evaluation metrics, feature list, and SHAP values.
    
    Parameters:
        df (pd.DataFrame): Processed DataFrame.
        model_filepath (str): Path to save the trained model.
        scaler_filepath (str): Path to save the scaler.
        metrics_filepath (str): Path to save the evaluation metrics.
        feature_list_filepath (str): Path to save the feature list.
        shap_filepath (str): Path to save the SHAP values.
    """
    # Ensure data is sorted by customer & date
    df = df.sort_values(by=["customer_id", "date"])
    
    # Keep only first churn event per customer
    df['first_churn_idx'] = df.groupby('customer_id')['churn'].transform(lambda x: x.idxmax() if x.max() == 1 else np.nan)
    df_filtered = df.loc[(df['churn'] == 0) | (df.index == df['first_churn_idx'])].drop(columns=['first_churn_idx'])
    
    # Select features & target
    features = [
        "month", "months_since_issuing", "plan_type_encoded", "cpi",
        "transaction_amount_lag_1m", "transaction_amount_mean_prev_3m", "transaction_amount_slope_3m",
        "plan_type_encoded_lag_1m", "plan_type_encoded_mean_prev_3m", "plan_type_encoded_slope_3m",
        "cpi_lag_1m", "cpi_mean_prev_3m", "cpi_slope_3m"
    ]
    target = "churn"
    
    X = df_filtered[features]
    y = df_filtered[target]
    
    # Handle NaNs
    X.fillna(X.median(), inplace=True)
    
    # Split data (time-based)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Apply SMOTE to balance classes
    smote = SMOTE(sampling_strategy=0.7, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Logistic Regression Model
    log_reg = LogisticRegression(class_weight="balanced", random_state=42)
    log_reg.fit(X_train_scaled, y_train_resampled)
    y_pred = log_reg.predict(X_test_scaled)
    
    # Compute metrics
    metrics = {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }
    
    # Save model
    with open(model_filepath, 'wb') as model_file:
        pickle.dump(log_reg, model_file)
    
    # Save scaler
    with open(scaler_filepath, 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    
    # Save metrics
    with open(metrics_filepath, 'w') as metrics_file:
        json.dump(metrics, metrics_file, indent=4)
    
    # Save feature list
    with open(feature_list_filepath, 'w') as feature_file:
        json.dump(features, feature_file)
    
    # Explain the Logistic Regression Model using SHAP
    explainer_log = shap.Explainer(log_reg, X_train_scaled)
    shap_values_log = explainer_log(X_test_scaled)
    
    # Convert SHAP values to a DataFrame for better readability
    shap_df = pd.DataFrame(shap_values_log.values, columns=features)
    shap_df['shap_expected_value'] = shap_values_log.base_values  # Add expected value (mean prediction)
    
    # Save SHAP values to CSV
    shap_df.to_csv(shap_filepath, index=False)
    
    print(f"Model training completed. Model, scaler, metrics, features, and SHAP values saved. {shap_filepath}")

if __name__ == "__main__":
    # Define folder path
    model_folder = 'saved_models/'

    # Ensure the folder exists
    os.makedirs(model_folder, exist_ok=True)

    # Set file paths
    churn_filepath = 'data/churn_data (6).csv'
    cpi_filepath = 'data/cpi_data_2023.csv'
    output_filepath = 'data/churn_data_ext.csv'
    model_filepath = os.path.join(model_folder, 'model.pkl')
    scaler_filepath = os.path.join(model_folder, 'scaler.pkl')
    metrics_filepath = os.path.join(model_folder, 'metrics.json')
    feature_list_filepath = os.path.join(model_folder, 'features.json')
    shap_filepath = os.path.join(model_folder, 'shap_values.csv')
    
    # Process and train model
    df_processed = preprocess_churn_data(churn_filepath, cpi_filepath, output_filepath)
    train_churn_model(df_processed, model_filepath, scaler_filepath, metrics_filepath, feature_list_filepath, shap_filepath)
