import pytest
import pandas as pd
import os
import shutil
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.data_processing import load_raw_data, DateTimeExtractor, FeatureAggregator, RFMCalculator, RiskProfiler, create_feature_engineering_pipeline, create_full_processed_dataframe

@pytest.fixture(scope="session")
def actual_raw_data_path():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(project_root, 'data', 'raw', 'data.csv')
    if not os.path.exists(data_path):
        pytest.fail(f"Required data file not found: {data_path}. Please ensure 'data.csv' exists in 'data/raw/'.")
    return data_path

def test_load_raw_data_actual(actual_raw_data_path):
    df = load_raw_data(actual_raw_data_path)
    assert not df.empty
    assert 'Amount' in df.columns
    assert 'TransactionId' in df.columns
    assert 'FraudResult' in df.columns
    assert df.shape[0] > 0 

def test_datetime_extractor():
    data = {
        'TransactionStartTime': ['2023-01-01 10:30:00', '2023-02-15 14:00:00', '2023-03-20 08:00:00', '2023-04-10 20:00:00'],
        'Amount': [100, 200, 150, 50]
    }
    df = pd.DataFrame(data)

    extractor = DateTimeExtractor()
    transformed_df = extractor.transform(df)

    assert 'TransactionHour' in transformed_df.columns
    assert 'TransactionDay' in transformed_df.columns
    assert 'TransactionMonth' in transformed_df.columns
    assert 'TransactionYear' in transformed_df.columns
    assert 'TransactionDayOfWeek' in transformed_df.columns
    assert 'TransactionIsWeekend' in transformed_df.columns

    assert transformed_df['TransactionHour'].iloc[0] == 10
    assert transformed_df['TransactionDay'].iloc[1] == 15
    assert transformed_df['TransactionMonth'].iloc[2] == 3
    assert transformed_df['TransactionYear'].iloc[0] == 2023
    assert transformed_df['TransactionDayOfWeek'].iloc[0] == 6 
    assert transformed_df['TransactionIsWeekend'].iloc[0] == 1 

    assert transformed_df['TransactionDayOfWeek'].iloc[1] == 2
    assert transformed_df['TransactionIsWeekend'].iloc[1] == 0

    assert transformed_df['TransactionDayOfWeek'].iloc[3] == 0
    assert transformed_df['TransactionIsWeekend'].iloc[3] == 0

def test_feature_aggregator():
    data = {
        'TransactionId': ['T1', 'T2', 'T3', 'T4'],
        'AccountId': ['A1', 'A2', 'A1', 'A3'],
        'SubscriptionId': ['S1', 'S2', 'S1', 'S3'],
        'CustomerId': ['C1', 'C2', 'C1', 'C3'],
        'Amount': [10, 20, 15, 30]
    }
    df = pd.DataFrame(data)

    aggregator = FeatureAggregator()
    transformed_df = aggregator.transform(df)

    assert 'TotalTransactionAmount_AccountId' in transformed_df.columns
    assert transformed_df[transformed_df['AccountId'] == 'A1']['TotalTransactionAmount_AccountId'].iloc[0] == 25
    assert transformed_df[transformed_df['AccountId'] == 'A2']['TotalTransactionAmount_AccountId'].iloc[0] == 20
    assert transformed_df[transformed_df['AccountId'] == 'A3']['TotalTransactionAmount_AccountId'].iloc[0] == 30

    assert 'TotalTransactionAmount' in transformed_df.columns
    assert transformed_df['TotalTransactionAmount'].iloc[0] == 75

def test_rfm_calculator():
    data = {
        'TransactionId': ['T1', 'T2', 'T3', 'T4', 'T5'],
        'CustomerId': ['C1', 'C2', 'C1', 'C3', 'C2'],
        'TransactionStartTime': ['2023-01-01', '2023-01-05', '2023-01-03', '2023-01-10', '2023-01-07'],
        'Amount': [100, 200, 50, 300, 150]
    }
    df = pd.DataFrame(data)

    rfm_calc = RFMCalculator()
    transformed_df = rfm_calc.transform(df)

    assert 'Recency' in transformed_df.columns
    assert 'Frequency' in transformed_df.columns
    assert 'Monetary' in transformed_df.columns
    assert 'RFM_Cluster' in transformed_df.columns

    assert transformed_df[transformed_df['CustomerId'] == 'C1']['Recency'].iloc[0] == 8
    assert transformed_df[transformed_df['CustomerId'] == 'C1']['Frequency'].iloc[0] == 2
    assert transformed_df[transformed_df['CustomerId'] == 'C1']['Monetary'].iloc[0] == 150

    assert transformed_df['RFM_Cluster'].isin(['Champions', 'Loyal Customers', 'Potential Loyalists', 'At Risk', 'Others']).all()

def test_risk_profiler():
    data = {
        'Amount': [100, 200, 50, 300, 10],
        'AverageTransactionAmount': [150, 150, 150, 150, 150],
        'TransactionCount': [10, 2, 7, 1, 6],
        'TransactionHour': [23, 10, 1, 15, 0]
    }
    df = pd.DataFrame(data)

    profiler = RiskProfiler()
    transformed_df = profiler.transform(df)

    assert 'is_high_risk' in transformed_df.columns
    assert transformed_df['is_high_risk'].iloc[0] == 0
    assert transformed_df['is_high_risk'].iloc[1] == 0
    assert transformed_df['is_high_risk'].iloc[2] == 0
    assert transformed_df['is_high_risk'].iloc[3] == 0

    data_high_risk = {
        'Amount': [200],
        'AverageTransactionAmount': [100],
        'TransactionCount': [10],
        'TransactionHour': [23]
    }
    df_high_risk = pd.DataFrame(data_high_risk)
    transformed_df_high_risk = profiler.transform(df_high_risk)
    assert transformed_df_high_risk['is_high_risk'].iloc[0] == 1


def test_create_feature_engineering_pipeline():
    pipeline = create_feature_engineering_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 5
    assert pipeline.named_steps['datetime_extractor'].__class__.__name__ == 'DateTimeExtractor'
    assert pipeline.named_steps['feature_aggregator'].__class__.__name__ == 'FeatureAggregator'
    assert pipeline.named_steps['rfm_calculator'].__class__.__name__ == 'RFMCalculator'
    assert pipeline.named_steps['risk_profiler'].__class__.__name__ == 'RiskProfiler'
    assert pipeline.named_steps['final_preprocessor'].__class__.__name__ == 'ColumnTransformer'


def test_create_full_processed_dataframe_actual(actual_raw_data_path):
    raw_df = load_raw_data(actual_raw_data_path)
    processed_df = create_full_processed_dataframe(raw_df.copy())

    assert 'num__Amount' in processed_df.columns
    assert 'num__TransactionHour' in processed_df.columns
    assert 'cat__CurrencyCode_UGX' in processed_df.columns 

    assert 'CurrencyCode' not in processed_df.columns 
    assert 'CountryCode' not in processed_df.columns

    assert isinstance(processed_df, pd.DataFrame)
    assert not processed_df.empty
    assert processed_df.shape[1] > raw_df.shape[1] 
    assert 'FraudResult' in processed_df.columns 
