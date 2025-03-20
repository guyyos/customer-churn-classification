# Purpose

This repository contains a script for training a logistic regression model to predict customer churn based on various features. The script preprocesses churn and CPI (Consumer Price Index) data, creates lag and rolling trend features, handles class imbalance using SMOTE, and provides SHAP (Shapley Additive Explanations) values for model interpretability.
This Python script, inference_model.py, is used to predict customer churn based on historical data and extend predictions for future months. 


# Setup

1. **Requirements**: Install dependencies via `pip`:
    ```bash
    pip install pandas numpy scikit-learn imbalanced-learn shap
    ```

2. **Data**: Ensure the following datasets are available:
    - `churn_data.csv`: Customer churn data with transaction history.
    - `cpi_data.csv`: CPI data for merging.

3. **Directory Structure**:
    ```
    - data/
        - churn_data.csv
        - cpi_data.csv
    - saved_models/
        - model.pkl
        - scaler.pkl
        - metrics.json
        - features.json
        - shap_values.csv
    - inference_model.py
    - train_model.py
    ```

# Usage - Train

Run the script to preprocess data, train the model, and save results:

```bash
python train_model.py
```
## Functions

- **preprocess_churn_data**: Preprocesses churn and CPI data by encoding categorical variables, generating lag and rolling features, and merging external data sources.
- **train_churn_model**: Trains a logistic regression model, applies SMOTE for class balancing, standardizes features, and saves the model, scaler, metrics, feature list, and SHAP values.

## Outputs:
- **Model**: Logistic regression model (`model.pkl`).
- **Scaler**: Standard scaler for feature scaling (`scaler.pkl`).
- **Metrics**: Precision, recall, and F1 scores (`metrics.json`).
- **Features**: List of features used in the model (`features.json`).
- **SHAP Values**: Model explainability using SHAP (`shap_values.csv`).

# Usage - Inference

## Functions

### `predict_churn(df, model, scaler, features)`
Predicts churn for the given dataframe.

### `predict_future_churn(df, model, scaler, features, months_ahead=2)`
Predicts churn for future months by extending the dataset and applying the model to future data.

1. Place the trained model (`model.pkl`), scaler (`scaler.pkl`), and features list (`features.json`) in the `saved_models/` directory.
2. Provide the churn data (`churn_data.csv`) and CPI data (`cpi_data_2023.csv`) in the `data/` directory.
3. Run the script:

```bash
python inference_model.py
```

## Explanations:

### Removing Post-Churn Data in training
1. A customer’s first churn event (churn=1) is identified.
2. All interactions after the first churn event are removed to prevent data leakage.
3. This ensures that the model only sees data up to the point of churn for each customer.

### Why Logistic Regression?
Logistic regression was chosen as the model for this churn prediction task due to its simplicity, interpretability, and effectiveness when the target variable is binary (churn = 0 or 1). It also performs well when features are linearly separable, and it provides a probabilistic output, making it suitable for a business application like predicting customer churn.

## Features Used

The following features were included in the model:

- **Month**: The month of the transaction, which may capture seasonal patterns.
- **Months Since Issuing**: The number of months since the customer subscribed to the service, which is important for understanding customer behavior over time.
- **Plan Type (Encoded)**: The type of subscription plan (Basic, Standard, Premium), encoded as numeric values (1, 2, 3) to capture the effects of different plans on churn likelihood.
- **CPI (Consumer Price Index)**: The CPI value, merged from external data, accounts for external economic conditions that may influence churn.
- **Lag Features**: These represent the previous month’s values for transaction amount, plan type, and CPI, capturing short-term trends and customer behavior.
- **Rolling Mean and Slope Features**: These features represent the trend over the past three months, helping the model capture both short-term and longer-term patterns in customer behavior.

## Imputation

In cases where data is missing (NaNs), we used **median imputation** (`X.fillna(X.median(), inplace=True)`) to handle missing values. The median was chosen as it is less sensitive to outliers compared to the mean, making it more robust when dealing with missing values in transaction amounts or CPI.

## Handling Class Imbalance (SMOTE)

Since the dataset is imbalanced (with a minority class representing churned customers), we used **SMOTE** (Synthetic Minority Over-sampling Technique) to oversample the minority class during training. This helps improve the model’s ability to correctly classify the minority class and prevent the model from being biased towards the majority class.

## Standardization

Feature scaling was done using **StandardScaler**, which standardizes the features to have a mean of 0 and a standard deviation of 1. This is important for logistic regression, as the model is sensitive to the scale of input features. Standardizing ensures that all features are treated equally, preventing bias toward variables with larger scales.

## SHAP for Model Explanation

**SHAP** values were computed to explain the model's predictions. SHAP provides a way to understand the impact of each feature on individual predictions, making the model more interpretable and transparent. This is crucial for understanding how each feature contributes to the prediction of churn for each customer.

## Future Work

- **Imputation Strategies**: Investigate different imputation techniques, such as filling missing values with 0 or using an indicator variable to flag missing data, to assess their impact on model performance.
  
- **Model Enhancement**: Explore the use of tree-based classifiers, like XGBoost, to capture non-linear relationships in the data, which may improve the accuracy of churn predictions.

- **Feature Engineering**: Experiment with additional feature transformations, such as polynomial features or interaction terms, to capture more complex patterns in the data.

- **Model Evaluation**: Implement more advanced model evaluation techniques, including cross-validation and hyperparameter tuning, to optimize the performance of the model.

- **Temporal Analysis**: Investigate incorporating more temporal features, such as seasonality or time decay, to improve future churn predictions and capture changing trends over time.

- **Interpretability**: Explore model interpretability techniques (e.g., SHAP, LIME) to better understand feature importance and provide actionable insights for business decision-making.
