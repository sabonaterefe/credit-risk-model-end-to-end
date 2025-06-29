import os
import sys
import pandas as pd
import numpy as np
import pytest
import warnings

# Suppress pandas warnings that are expected in our test cases
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data_processing import preprocess_data

@pytest.fixture
def sample_raw_data():
    """Fixture providing sample raw data with various edge cases"""
    return pd.DataFrame({
        'TransactionId': ['T1', 'T2', 'T3', 'T1'],  # duplicate
        'BatchId': ['B1', 'B2', None, 'B1'],
        'CurrencyCode': ['UGX', None, 'UGX', 'UGX'],
        'CountryCode': [256, 256, 256, 256],
        'ProviderId': ['P1', 'P2', None, 'P1'],
        'ProductId': ['Prod1', 'Prod2', 'Prod3', 'Prod1'],
        'ProductCategory': ['airtime', 'utility', None, 'airtime'],
        'ChannelId': ['Android', 'Web', None, 'Android'],
        'Amount': [1000.0, -50.0, np.nan, 1000.0],
        'Value': [1000.0, 50.0, 2000.0, 1000.0],
        'TransactionStartTime': [
            '2018-11-15 02:18:49', 
            '2018-11-15 03:00:00', 
            'invalid',  # invalid datetime
            '2018-11-15 02:18:49'
        ],
        'PricingStrategy': [2, 2, 2, 2],
        'FraudResult': [0, 1, None, 0]  # missing value
    })

def test_preprocessing_removes_duplicates(sample_raw_data):
    """Test that duplicate transactions are removed"""
    cleaned = preprocess_data(sample_raw_data.copy())
    assert cleaned['TransactionId'].duplicated().sum() == 0, \
        "Duplicate TransactionIds should be removed"

def test_missing_categorical_filled(sample_raw_data):
    """Test that missing categorical values are filled with 'Unknown'"""
    cleaned = preprocess_data(sample_raw_data.copy())
    categorical_cols = ['BatchId', 'ProviderId', 'ProductCategory', 'ChannelId']
    for col in categorical_cols:
        assert 'Unknown' in cleaned[col].values, \
            f"Missing values in {col} should be filled with 'Unknown'"

def test_missing_numerical_filled(sample_raw_data):
    """Test that missing numerical values are filled"""
    cleaned = preprocess_data(sample_raw_data.copy())
    numerical_cols = ['Amount', 'Value']
    for col in numerical_cols:
        assert cleaned[col].isnull().sum() == 0, \
            f"Missing values in {col} should be filled"

def test_datetime_conversion_and_features(sample_raw_data):
    """Test datetime conversion and feature extraction"""
    cleaned = preprocess_data(sample_raw_data.copy())
    
    # Check TransactionHour exists and is integer type
    assert 'TransactionHour' in cleaned.columns, \
        "TransactionHour feature should be created"
    
    # Check type is actually integer (not just int64)
    assert pd.api.types.is_integer_dtype(cleaned['TransactionHour']), \
        "TransactionHour should be integer dtype"
    
    # Check all values are actually integers
    assert all(isinstance(h, (int, np.integer)) for h in cleaned['TransactionHour']), \
        "All TransactionHour values should be integers"

def test_outlier_capping(sample_raw_data):
    """Test that outliers are properly capped"""
    cleaned = preprocess_data(sample_raw_data.copy())
    amount = cleaned['Amount']
    
    # Calculate expected bounds using IQR
    q1, q3 = amount.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    
    # Check all values are within bounds
    assert amount.between(lower, upper).all(), \
        "Amount values should be capped at IQR bounds"

def test_fraudresult_filled(sample_raw_data):
    """Test that missing FraudResult values are filled"""
    cleaned = preprocess_data(sample_raw_data.copy())
    assert cleaned['FraudResult'].isnull().sum() == 0, \
        "Missing FraudResult values should be filled"