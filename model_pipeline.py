import polars as pl
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from preprocessing import count_by_values, dev_feats, reconstruct_essay, get_essay_df, word_feats, sent_feats, parag_feats, product_to_keys, get_keys_pressed_per_second
from preprocessing import transform_and_clean_data, clean_column_names
from sklearn.preprocessing import StandardScaler
import math


def load_and_prepare_data(file_path):
    """Load testing data and perform initial preprocessing."""
    logs = pl.scan_csv(file_path)
    features = dev_feats(logs).collect().to_pandas()
    logs = logs.collect().to_pandas()

    essays = get_essay_df(logs)
    features = features.merge(word_feats(essays), on='id', how='left')
    features = features.merge(sent_feats(essays), on='id', how='left')
    features = features.merge(parag_feats(essays), on='id', how='left')
    features = features.merge(get_keys_pressed_per_second(logs), on='id', how='left')
    features = features.merge(product_to_keys(logs, essays), on='id', how='left')

    return features

def standardize_and_clean_data(X, X_test):
    """
    Scale the features using StandardScaler, replace NaNs with zero, and clean the column names.

    Args:
        X (pd.DataFrame): train dataset
        X_test (pd.DataFrame): test dataset

    Returns:
        pd.DataFrame: The standardized and cleaned features DataFrame.
    """
    scaler = StandardScaler()
    
    # Scale the data
    
    X_tran = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_test_tran = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
       
    # Replace NaNs with zero
    X_tran.fillna(0, inplace=True)
    X_test_tran.fillna(0, inplace=True)
    
    # Clean the column names
    X_tran = clean_column_names(X_tran)
    X_test_tran = clean_column_names(X_test_tran)

    return X_tran, X_test_tran


def feature_selection_and_scaling(features, target, n_features, scaler=None):
    """Select features using statistical tests and apply scaling.
    
    Args:
        features (DataFrame): The feature data.
        target (Series): The target data.
        n_features (int): The number of features to select.
        scaler (StandardScaler, optional): An instance of a pre-fitted StandardScaler.

    Returns:
        DataFrame: The scaled features after selection.
    """
    if features.shape[0] != target.shape[0]:
        raise ValueError(f"Inconsistent number of samples: features has {features.shape[0]} samples, target has {target.shape[0]} samples.")
    
    # Ensure the scaler is provided or initialize and fit a new one
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(features)

    # Fill NaNs with zero or another imputation strategy before feature selection
    features_filled = features.fillna(0)
    
    # Initialize and fit the SelectKBest
    selector = SelectKBest(score_func=f_regression, k=n_features)
    selector.fit(features_filled, target)

    # Select the features and transform
    selected_features = features.iloc[:, selector.get_support(indices=True)]
    scaled_features = pd.DataFrame(scaler.transform(selected_features), columns=selected_features.columns)
    
    # Clean column names
    scaled_features = clean_column_names(scaled_features)

    return scaled_features


def train_and_predict(models, X_train, y_train, X_test):
    """Train models and make predictions on the test set."""
    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions[name] = model.predict(X_test)

    return predictions

def ensemble_predictions(preds, weights):
    """Combine predictions using weighted average."""
    # Ensure the weights sum to 1
    if not math.isclose(sum(weights), 1.0, rel_tol=1e-9):
        raise ValueError("The weights must sum to 1.")
    weighted_preds = sum(preds[name] * weight for name, weight in zip(preds.keys(), weights))
    final_pred = weighted_preds / sum(weights)
    return final_pred
