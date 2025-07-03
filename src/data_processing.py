import pandas as pd
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
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

class RFMCalculator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')

        snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)

        rfm = df.groupby('CustomerId').agg(
            Recency=('TransactionStartTime', lambda date: (snapshot_date - date.max()).days),
            Frequency=('TransactionId', 'count'),
            Monetary=('Amount', 'sum')
        )

        rfm = rfm.reset_index()

        return df.merge(rfm, on='CustomerId', how='left')

class RiskProfiler(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init='auto')
        self.scaler = MinMaxScaler()
        self.high_risk_cluster_label = None

    def fit(self, X, y=None):
        rfm_features = X[['Recency', 'Frequency', 'Monetary']].copy()
        rfm_scaled = self.scaler.fit_transform(rfm_features)
        self.kmeans.fit(rfm_scaled)

        cluster_centers = self.kmeans.cluster_centers_

        r_center = cluster_centers[:, 0]
        f_center = cluster_centers[:, 1]
        m_center = cluster_centers[:, 2]

        risk_score = r_center - f_center - m_center
        self.high_risk_cluster_label = np.argmin(risk_score)

        return self

    def transform(self, X):
        df = X.copy()

        rfm_features = df[['Recency', 'Frequency', 'Monetary']].copy()

        rfm_scaled = self.scaler.transform(rfm_features)

        df['RFM_Cluster'] = self.kmeans.predict(rfm_scaled)

        df['is_high_risk'] = df['RFM_Cluster'].apply(lambda x: 1 if x == self.high_risk_cluster_label else 0)

        return df

def load_raw_data(file_name):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, 'data', 'raw', file_name)

    df = pd.read_csv(file_path)
    print(f"Loaded data from: {file_path}")
    return df

def create_feature_engineering_pipeline():
    numerical_cols_for_preprocessor = ['Amount', 'Value', 'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear',
                                       'TransactionDayOfWeek', 'TransactionIsWeekend',
                                       'Recency', 'Frequency', 'Monetary',
                                       'TotalTransactionAmount', 'AverageTransactionAmount',
                                       'TransactionCount', 'StdTransactionAmount', 'MaxTransactionAmount', 'MinTransactionAmount']

    categorical_cols_for_preprocessor = ['CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId', 
                                         'ProductCategory', 'ChannelId', 'PricingStrategy']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num_imputer', SimpleImputer(strategy='mean'), numerical_cols_for_preprocessor),
            ('cat_imputer', SimpleImputer(strategy='most_frequent'), categorical_cols_for_preprocessor),
            ('num_scaler', StandardScaler(), numerical_cols_for_preprocessor),
            ('cat_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols_for_preprocessor)
        ],
        remainder='drop' # Explicitly drop any columns not specified
    )

    pipeline = Pipeline(steps=[
        ('datetime_extractor', DateTimeExtractor()),
        ('feature_aggregator', FeatureAggregator()),
        ('rfm_calculator', RFMCalculator()),
        ('risk_profiler', RiskProfiler(n_clusters=3, random_state=42)),
        ('final_preprocessor', preprocessor) # Renamed to avoid confusion, but it's the same object
    ])

    return pipeline