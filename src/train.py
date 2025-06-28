
from src.feature_engineering import add_aggregate_features, add_datetime_features, get_preprocessing_pipeline
import pandas as pd

df = pd.read_csv("data/raw/data.csv")
df = add_aggregate_features(df)
df = add_datetime_features(df)

pipeline = get_preprocessing_pipeline()
processed = pipeline.fit_transform(df)
