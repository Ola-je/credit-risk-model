from fastapi import FastAPI, HTTPException
from pydantic_models import TransactionFeatures, PredictionResponse
import mlflow
import os
import pandas as pd
from src.data_processing import create_feature_engineering_pipeline, DateTimeExtractor, FeatureAggregator, RFMCalculator, RiskProfiler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

app = FastAPI(title="Credit Risk Model API")

model = None
feature_pipeline = None

def load_model():
    global model
    global feature_pipeline
    try:
        # Construct the path to the local mlruns directory
        local_mlruns_path = os.path.abspath("mlruns")
        database_uri = f"sqlite:///{os.path.join(local_mlruns_path, 'mlruns.db')}"
        mlflow.set_tracking_uri(database_uri)

        # Assuming you've registered a model named 'CreditRiskModel'
        # with version 1 (or the latest version)
        model_name = "CreditRiskModel"
        model_uri = f"models:/{model_name}/latest"
        
        # Check if the model exists in the registry
        client = mlflow.tracking.MlflowClient()
        try:
            client.get_latest_versions(model_name, stages=["None"])
        except Exception as e:
            raise RuntimeError(f"Model '{model_name}' not found in MLflow registry. Please ensure it's registered. Error: {e}")

        model = mlflow.pyfunc.load_model(model_uri)
        feature_pipeline = create_feature_engineering_pipeline()
        print(f"Successfully loaded model from: {model_uri}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
async def read_root():
    return {"message": "Credit Risk Model API is running. Go to /docs for API documentation."}

@app.post("/predict", response_model=PredictionResponse)
async def predict_risk(data: TransactionFeatures):
    if model is None or feature_pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Server startup issue.")

    input_df = pd.DataFrame([data.model_dump()])

    try:
        # Re-apply relevant data processing steps that happen before the final ColumnTransformer
        # These are the same steps as in create_full_processed_dataframe up to RiskProfiler
        datetime_extractor = DateTimeExtractor()
        feature_aggregator = FeatureAggregator()
        rfm_calculator = RFMCalculator()
        risk_profiler = RiskProfiler()

        processed_input_df = datetime_extractor.transform(input_df.copy())
        processed_input_df = feature_aggregator.transform(processed_input_df)
        processed_input_df = rfm_calculator.transform(processed_input_df)
        processed_input_df = risk_profiler.transform(processed_input_df)

        # Get the final preprocessor from the feature_pipeline
        final_preprocessor = feature_pipeline.named_steps['final_preprocessor']

        # Define numerical and categorical columns for the preprocessor based on your data_processing.py
        # These lists should match exactly what's used in create_full_processed_dataframe
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
        
        # Filter columns to only those expected by the preprocessor
        numerical_cols_present = [col for col in numerical_cols if col in processed_input_df.columns]
        categorical_cols_present = [col for col in categorical_cols if col in processed_input_df.columns]

        features_for_prediction = processed_input_df[numerical_cols_present + categorical_cols_present]
        
        # Transform the features using the fitted preprocessor
        transformed_features = final_preprocessor.transform(features_for_prediction)

        # Predict probability
        risk_probability = model.predict_proba(transformed_features)[:, 1][0]
        return PredictionResponse(risk_probability=risk_probability)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")