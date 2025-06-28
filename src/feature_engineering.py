import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder, 
    StandardScaler, 
    OrdinalEncoder,
    MinMaxScaler
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin

# Optional WOE support
# from xverse.transformer import WOE  # Uncomment if WOE is to be used with target variable

# --- Custom Transformers ---

class AggregateFeaturesAdder(BaseEstimator, TransformerMixin):
    """Adds aggregate transaction features per customer."""
    def __init__(self, group_col='CustomerId', agg_col='Amount'):
        self.group_col = group_col
        self.agg_col = agg_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        agg = X.groupby(self.group_col)[self.agg_col].agg(
            total_amount='sum',
            avg_amount='mean',
            count_transactions='count',
            std_amount='std'
        ).reset_index()
        return X.merge(agg, on=self.group_col, how='left')


class DatetimeFeaturesExtractor(BaseEstimator, TransformerMixin):
    """Extracts hour, day, month, and year from TransactionStartTime."""
    def __init__(self, datetime_col='TransactionStartTime'):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.datetime_col] = pd.to_datetime(X[self.datetime_col])
        X['transaction_hour'] = X[self.datetime_col].dt.hour
        X['transaction_day'] = X[self.datetime_col].dt.day
        X['transaction_month'] = X[self.datetime_col].dt.month
        X['transaction_year'] = X[self.datetime_col].dt.year
        return X


# --- Main Feature Engineering Pipeline ---
def get_feature_engineering_pipeline():
    """
    Builds and returns the full feature engineering pipeline.
    Includes:
        - Aggregate feature creation
        - Datetime feature extraction
        - Handling missing values (SimpleImputer/KNNImputer)
        - Scaling (StandardScaler)
        - Encoding (OneHotEncoder, OrdinalEncoder)
        - WOE optional (not included in pipeline)
    """
    # Define features
    numeric_features = [
        "Amount", "PricingStrategy",
        "total_amount", "avg_amount", "count_transactions", "std_amount",
        "transaction_hour", "transaction_day", "transaction_month", "transaction_year"
    ]
    categorical_features = ["ProductCategory", "ChannelId", "ProviderId"]
    ordinal_features = ["ProductCategory"]  # Include only if true ordinal nature assumed

    # --- Numeric processing pipeline ---
    numeric_transformer = Pipeline(steps=[
        ("imputer", KNNImputer(n_neighbors=5)),
        ("scaler", StandardScaler())
    ])

    # --- Categorical processing pipeline ---
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # --- Ordinal encoding (optional) ---
    ordinal_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder())
    ])

    # --- ColumnTransformer to combine all ---
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
        ("ordinal", ordinal_transformer, ordinal_features),
    ])

    # --- Final full pipeline ---
    full_pipeline = Pipeline(steps=[
        ("agg_features", AggregateFeaturesAdder()),
        ("datetime_features", DatetimeFeaturesExtractor()),
        ("preprocessing", preprocessor),
        # Optional WOE encoding (can be applied outside of pipeline)
        # ("woe", WOE())
    ])

    return full_pipeline
