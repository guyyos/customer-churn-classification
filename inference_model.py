import pandas as pd
import pickle
import json
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from train_model import preprocess_churn_data, rolling_slope


def predict_churn(df, model, scaler, features):
    """
    Predict churn for the given dataframe.
    """
    df = df.copy()
    X = df[features].fillna(df[features].median())
    X_scaled = scaler.transform(X)
    df["predicted_churn"] = model.predict(X_scaled)
    return df

def predict_future_churn(df, model, scaler, features, months_ahead=2):
    """
    Predict churn for future months by extending the dataset.
    """
    df = df.copy()
    future_predictions = []
    
    # Ensure 'date' column is in datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    df["customer_id"] = df["customer_id"].astype(str)
    
    for i in range(1, months_ahead + 1):
        future_date = df["date"].max() + pd.DateOffset(months=1)
        df_future = df[df["date"] == df["date"].max()].copy()
        df_future["date"] = future_date
        df_future["month"] = future_date.month
        df_future["months_since_issuing"] += 1
        
        for feature in ["transaction_amount", "plan_type_encoded", "cpi"]:
            df_future[f"{feature}_lag_1m"] = df.groupby("customer_id")[feature].shift(1)
            df_future[f"{feature}_mean_prev_3m"] = (
                df.groupby("customer_id")[feature].shift(1).rolling(window=3, min_periods=1).mean()
            )
            df_future[f"{feature}_slope_3m"] = (
                df.groupby("customer_id")[feature].shift(1).rolling(window=3, min_periods=2).apply(rolling_slope, raw=True)
            )
        
        df_future = predict_churn(df_future, model, scaler, features)
        future_predictions.append(df_future)
        df = pd.concat([df, df_future], ignore_index=True)
    
    return pd.concat(future_predictions, ignore_index=True)

def main():
    # Load the model, scaler, and feature list
    model_filepath = 'saved_models/model.pkl'
    scaler_filepath = 'saved_models/scaler.pkl'
    feature_list_filepath = 'saved_models/features.json'

    # Load the trained model
    with open(model_filepath, 'rb') as model_file:
        model = pickle.load(model_file)

    # Load the scaler
    with open(scaler_filepath, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    # Load the feature list
    with open(feature_list_filepath, 'r') as feature_file:
        features = json.load(feature_file)

    # Load the input data
    churn_filepath = 'data/churn_data (6).csv'
    cpi_filepath = 'data/cpi_data_2023.csv'
    output_filepath = 'data/churn_data_ext1.csv'

    df = preprocess_churn_data(churn_filepath, cpi_filepath, output_filepath)

    # Run predictions for the current data
    df = predict_churn(df, model, scaler, features)

    # Run predictions for the next 2 months
    df_predicted = predict_future_churn(df, model, scaler, features, months_ahead=2)

    # Save results
    df.to_csv("data/df_with_predictions.csv", index=False)
    df_predicted.to_csv("data/df_predicted_future.csv", index=False)

    print("Predictions saved successfully.")

if __name__ == "__main__":
    main()
