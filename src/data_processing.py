import pandas as pd
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

class FeatureAggregator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        
        df['TotalTransactionAmount'] = df.groupby('CustomerId')['Amount'].transform('sum')
        df['AverageTransactionAmount'] = df.groupby('CustomerId')['Amount'].transform('mean')
        df['TransactionCount'] = df.groupby('CustomerId')['TransactionId'].transform('count')
        df['StdTransactionAmount'] = df.groupby('CustomerId')['Amount'].transform('std')
        df['MaxTransactionAmount'] = df.groupby('CustomerId')['Amount'].transform('max')
        df['MinTransactionAmount'] = df.groupby('CustomerId')['Amount'].transform('min')
        
        df['StdTransactionAmount'] = df['StdTransactionAmount'].fillna(0)
        
        return df

class DateTimeExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
        
        df['TransactionHour'] = df['TransactionStartTime'].dt.hour
        df['TransactionDay'] = df['TransactionStartTime'].dt.day
        df['TransactionMonth'] = df['TransactionStartTime'].dt.month
        df['TransactionYear'] = df['TransactionStartTime'].dt.year
        df['TransactionDayOfWeek'] = df['TransactionStartTime'].dt.dayofweek
        df['TransactionIsWeekend'] = df['TransactionDayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
        
        return df

def load_raw_data(file_name):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, 'data', 'raw', file_name)
    
    df = pd.read_csv(file_path)
    print(f"Loaded data from: {file_path}")
    return df

def create_feature_engineering_pipeline():
    numerical_cols = ['Amount', 'Value', 'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear',
                      'TransactionDayOfWeek', 'TotalTransactionAmount', 'AverageTransactionAmount',
                      'TransactionCount', 'StdTransactionAmount', 'MaxTransactionAmount', 'MinTransactionAmount']
    
    categorical_cols = ['CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId', 
                    'ProductCategory', 'ChannelId', 'PricingStrategy']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num_imputer', SimpleImputer(strategy='mean'), numerical_cols),
            ('cat_imputer', SimpleImputer(strategy='most_frequent'), categorical_cols),
            ('num_scaler', StandardScaler(), numerical_cols),
            ('cat_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('datetime_extractor', DateTimeExtractor()),
        ('feature_aggregator', FeatureAggregator()),
        ('preprocessor', preprocessor)
    ])
    
    return pipeline