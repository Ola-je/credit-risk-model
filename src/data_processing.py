import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import os

class DateTimeExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy['TransactionHour'] = pd.to_datetime(X_copy['TransactionStartTime']).dt.hour
        X_copy['TransactionDay'] = pd.to_datetime(X_copy['TransactionStartTime']).dt.day
        X_copy['TransactionMonth'] = pd.to_datetime(X_copy['TransactionStartTime']).dt.month
        X_copy['TransactionYear'] = pd.to_datetime(X_copy['TransactionStartTime']).dt.year
        X_copy['TransactionDayOfWeek'] = pd.to_datetime(X_copy['TransactionStartTime']).dt.dayofweek
        X_copy['TransactionIsWeekend'] = ((pd.to_datetime(X_copy['TransactionStartTime']).dt.dayofweek) // 5 == 1).astype(int)
        return X_copy

class FeatureAggregator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in ['AccountId', 'SubscriptionId', 'CustomerId']:
            X_copy[f'TotalTransactionAmount_{col}'] = X_copy.groupby(col)['Amount'].transform('sum')
            X_copy[f'AverageTransactionAmount_{col}'] = X_copy.groupby(col)['Amount'].transform('mean')
            X_copy[f'TransactionCount_{col}'] = X_copy.groupby(col)['Amount'].transform('count')
            X_copy[f'StdTransactionAmount_{col}'] = X_copy.groupby(col)['Amount'].transform('std').fillna(0)
            X_copy[f'MaxTransactionAmount_{col}'] = X_copy.groupby(col)['Amount'].transform('max')
            X_copy[f'MinTransactionAmount_{col}'] = X_copy.groupby(col)['Amount'].transform('min')

        X_copy['TotalTransactionAmount'] = X_copy['Amount'].sum()
        X_copy['AverageTransactionAmount'] = X_copy['Amount'].mean()
        X_copy['TransactionCount'] = X_copy['Amount'].count()
        X_copy['StdTransactionAmount'] = X_copy['Amount'].std()
        X_copy['MaxTransactionAmount'] = X_copy['Amount'].max()
        X_copy['MinTransactionAmount'] = X_copy['Amount'].min()
        return X_copy

class RFMCalculator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy['TransactionStartTime'] = pd.to_datetime(X_copy['TransactionStartTime'])
        snapshot_date = X_copy['TransactionStartTime'].max() + pd.Timedelta(days=1)

        rfm = X_copy.groupby('CustomerId').agg(
            Recency=('TransactionStartTime', lambda date: (snapshot_date - date.max()).days),
            Frequency=('TransactionId', 'count'),
            Monetary=('Amount', 'sum')
        ).reset_index()

        rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, duplicates='drop').cat.codes + 1
        rfm['F_Score'] = pd.qcut(rfm['Frequency'], 5, duplicates='drop').cat.codes + 1
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, duplicates='drop').cat.codes + 1
        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

        def rfm_cluster(row):
            if row['R_Score'] >= 4 and row['F_Score'] >= 4 and row['M_Score'] >= 4:
                return 'Champions'
            elif row['R_Score'] >= 4 and row['F_Score'] >= 3:
                return 'Loyal Customers'
            elif row['R_Score'] >= 3 and row['F_Score'] >= 3 and row['M_Score'] >= 3:
                return 'Potential Loyalists'
            elif row['R_Score'] <= 2 and row['F_Score'] >= 3:
                return 'At Risk'
            else:
                return 'Others'

        rfm['RFM_Cluster'] = rfm.apply(rfm_cluster, axis=1)

        X_copy = pd.merge(X_copy, rfm[['CustomerId', 'Recency', 'Frequency', 'Monetary', 'RFM_Cluster']], on='CustomerId', how='left')
        return X_copy

class RiskProfiler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
    
        X_copy['is_high_risk'] = ((X_copy['Amount'] > X_copy['AverageTransactionAmount']) &
                                  (X_copy['TransactionCount'] > 5) &
                                  (X_copy['TransactionHour'].isin([22, 23, 0, 1, 2, 3]))).astype(int)
        return X_copy

def create_feature_engineering_pipeline():
    numerical_cols = [
        'Amount', 'Value', 'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear',
        'TransactionDayOfWeek', 'TransactionIsWeekend',
        'TotalTransactionAmount', 'AverageTransactionAmount',
        'TransactionCount', 'StdTransactionAmount', 'MaxTransactionAmount', 'MinTransactionAmount',
        'Recency', 'Frequency', 'Monetary',
        'TotalTransactionAmount_AccountId', 'AverageTransactionAmount_AccountId', 'TransactionCount_AccountId',
        'StdTransactionAmount_AccountId', 'MaxTransactionAmount_AccountId', 'MinTransactionAmount_AccountId',
        'TotalTransactionAmount_SubscriptionId', 'AverageTransactionAmount_SubscriptionId', 'TransactionCount_SubscriptionId',
        'StdTransactionAmount_SubscriptionId', 'MaxTransactionAmount_SubscriptionId', 'MinTransactionAmount_SubscriptionId',
        'TotalTransactionAmount_CustomerId', 'AverageTransactionAmount_CustomerId', 'TransactionCount_CustomerId',
        'StdTransactionAmount_CustomerId', 'MaxTransactionAmount_CustomerId', 'MinTransactionAmount_CustomerId'
    ]

    categorical_cols = [
        'CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId',
        'ProductCategory', 'ChannelId', 'PricingStrategy', 'RFM_Cluster'
    ]

    final_preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        remainder='drop'
    )

    pipeline = Pipeline(steps=[
        ('datetime_extractor', DateTimeExtractor()),
        ('feature_aggregator', FeatureAggregator()),
        ('rfm_calculator', RFMCalculator()),
        ('risk_profiler', RiskProfiler()),
        ('final_preprocessor', final_preprocessor)
    ])
    return pipeline

def load_raw_data(file_name):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, 'data', 'raw', file_name)

    df = pd.read_csv(file_path)
    print(f"Loaded data from: {file_path}")
    return df

def create_full_processed_dataframe(raw_df):
    datetime_extractor = DateTimeExtractor()
    feature_aggregator = FeatureAggregator()
    rfm_calculator = RFMCalculator()
    risk_profiler = RiskProfiler()

    df_temp = datetime_extractor.transform(raw_df.copy())
    df_temp = feature_aggregator.transform(df_temp)
    df_temp = rfm_calculator.transform(df_temp)
    df_temp = risk_profiler.transform(df_temp)

    numerical_cols = [
        'Amount', 'Value', 'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear',
        'TransactionDayOfWeek', 'TransactionIsWeekend',
        'TotalTransactionAmount', 'AverageTransactionAmount',
        'TransactionCount', 'StdTransactionAmount', 'MaxTransactionAmount', 'MinTransactionAmount',
        'Recency', 'Frequency', 'Monetary',
        'TotalTransactionAmount_AccountId', 'AverageTransactionAmount_AccountId', 'TransactionCount_AccountId',
        'StdTransactionAmount_AccountId', 'MaxTransactionAmount_AccountId', 'MinTransactionAmount_AccountId',
        'TotalTransactionAmount_SubscriptionId', 'AverageTransactionAmount_SubscriptionId', 'TransactionCount_SubscriptionId',
        'StdTransactionAmount_SubscriptionId', 'MaxTransactionAmount_SubscriptionId', 'MinTransactionAmount_SubscriptionId',
        'TotalTransactionAmount_CustomerId', 'AverageTransactionAmount_CustomerId', 'TransactionCount_CustomerId',
        'StdTransactionAmount_CustomerId', 'MaxTransactionAmount_CustomerId', 'MinTransactionAmount_CustomerId'
    ]

    categorical_cols = [
        'CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId',
        'ProductCategory', 'ChannelId', 'PricingStrategy', 'RFM_Cluster'
    ]

    numerical_cols = [col for col in numerical_cols if col in df_temp.columns]
    categorical_cols = [col for col in categorical_cols if col in df_temp.columns]

    final_preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        remainder='drop'
    )

    X_features_to_transform = df_temp[numerical_cols + categorical_cols]
    transformed_features_array = final_preprocessor.fit_transform(X_features_to_transform)
    preprocessor_output_names = final_preprocessor.get_feature_names_out()

    processed_X_df = pd.DataFrame(transformed_features_array, columns=preprocessor_output_names, index=df_temp.index)

    final_df = pd.concat([
        processed_X_df,
        df_temp[['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId', 'FraudResult', 'is_high_risk']]
    ], axis=1)

    return final_df