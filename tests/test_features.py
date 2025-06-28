import pytest
import pandas as pd
import numpy as np
from src.data_processing import build_pipeline
from src.transformers import TemporalFeatureExtractor

def test_time_feature_extraction():
    test_data = pd.DataFrame({
        'TransactionStartTime': ['2025-01-01 14:30:00', '2025-01-07 09:15:00'],
        'CustomerId': [1, 2],
        'Amount': [100, 200]
    })
    
    transformer = TemporalFeatureExtractor()
    transformed = transformer.transform(test_data)
    
    assert 'hour' in transformed.columns
    assert transformed.loc[0, 'hour'] == 14
    assert transformed.loc[1, 'is_weekend'] == True

def test_full_pipeline():
    sample_data = pd.DataFrame({
        'TransactionStartTime': ['2025-01-01 12:00:00']*3,
        'CustomerId': [1,1,2],
        'Amount': [100,150,200],
        'ProductCategory': ['A','B','A'],
        'ChannelId': ['Web','App','Web']
    })
    
    pipeline = build_pipeline()
    result = pipeline.fit_transform(sample_data)
    
    assert result.shape[0] == 3  # Same number of rows
    assert result.shape[1] > 5   # Should have more columns after encoding