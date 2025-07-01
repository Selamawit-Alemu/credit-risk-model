import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta
from src.proxy_target import calculate_rfm, scale_rfm, perform_kmeans_clustering
from src.feature_engineering import AggregateFeaturesAdder, DatetimeFeaturesExtractor

# Sample test data
data = pd.DataFrame({
    "CustomerId": ["C1", "C1", "C2", "C2", "C3"],
    "TransactionId": [1, 2, 3, 4, 5],
    "TransactionStartTime": [
        datetime.now() - timedelta(days=1),
        datetime.now() - timedelta(days=5),
        datetime.now() - timedelta(days=30),
        datetime.now() - timedelta(days=60),
        datetime.now() - timedelta(days=90),
    ],
    "Amount": [100, 200, 300, 400, 500]
})


def test_calculate_rfm():
    snapshot_date = datetime.now() + timedelta(days=1)
    rfm = calculate_rfm(data, snapshot_date)
    assert set(["CustomerId", "recency", "frequency", "monetary"]).issubset(rfm.columns)
    assert rfm.shape[0] == data["CustomerId"].nunique()


def test_scale_rfm():
    snapshot_date = datetime.now() + timedelta(days=1)
    rfm = calculate_rfm(data, snapshot_date)
    rfm_scaled, scaler = scale_rfm(rfm)
    assert rfm_scaled.shape == (3, 3)
    assert np.isclose(np.mean(rfm_scaled[:, 0]), 0, atol=1e-6)


def test_perform_kmeans_clustering():
    snapshot_date = datetime.now() + timedelta(days=1)
    rfm = calculate_rfm(data, snapshot_date)
    rfm_scaled, _ = scale_rfm(rfm)
    labels, model = perform_kmeans_clustering(rfm_scaled, n_clusters=2, random_state=42)
    assert len(labels) == rfm.shape[0]
    assert hasattr(model, "predict")


from src.feature_engineering import AggregateFeaturesAdder

def test_add_aggregate_features():
    transformer = AggregateFeaturesAdder()
    enriched = transformer.fit_transform(data.copy())
    assert "total_amount" in enriched.columns
    assert enriched.shape[0] == data.shape[0]



from src.feature_engineering import DatetimeFeaturesExtractor

def test_add_datetime_features():
    transformer = DatetimeFeaturesExtractor()
    enriched = transformer.fit_transform(data.copy())
    assert "transaction_hour" in enriched.columns
    assert "transaction_day" in enriched.columns

from sklearn.pipeline import Pipeline

def test_pipeline_integration():
    pipeline = Pipeline([
        ("datetime", DatetimeFeaturesExtractor()),
        ("aggregate", AggregateFeaturesAdder())
    ])
    result = pipeline.fit_transform(data.copy())
    assert "transaction_hour" in result.columns
    assert "total_amount" in result.columns


