import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder, 
    StandardScaler, 
    OrdinalEncoder,
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin

class AggregateFeaturesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, group_col='CustomerId', agg_col='Amount'):
        self.group_col = group_col
        self.agg_col = agg_col
        self.agg_df_ = pd.DataFrame()  # ensure it's always defined

    def fit(self, X, y=None):
        if self.group_col in X.columns:
            self.agg_df_ = X.groupby(self.group_col)[self.agg_col].agg(
                total_amount='sum',
                avg_amount='mean',
                count_transactions='count',
                std_amount='std'
            ).reset_index()
        else:
            # Empty dataframe with expected columns
            self.agg_df_ = pd.DataFrame(columns=[
                self.group_col, 'total_amount', 'avg_amount', 'count_transactions', 'std_amount'
            ])
        return self

    def transform(self, X):
        X = X.copy()
        if self.group_col in X.columns and not self.agg_df_.empty:
            return X.merge(self.agg_df_, on=self.group_col, how='left')
        else:
            # Add zero-filled default columns
            X['total_amount'] = 0
            X['avg_amount'] = 0
            X['count_transactions'] = 0
            X['std_amount'] = 0
            return X



class DatetimeFeaturesExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col='TransactionStartTime'):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.datetime_col in X.columns:
            X[self.datetime_col] = pd.to_datetime(X[self.datetime_col])
            X['transaction_hour'] = X[self.datetime_col].dt.hour
            X['transaction_day'] = X[self.datetime_col].dt.day
            X['transaction_month'] = X[self.datetime_col].dt.month
            X['transaction_year'] = X[self.datetime_col].dt.year
        return X



class DropCustomerId(BaseEstimator, TransformerMixin):
    """Drop CustomerId column after feature engineering"""
    def __init__(self, col_name='CustomerId'):
        self.col_name = col_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.col_name in X.columns:
            return X.drop(columns=[self.col_name])
        return X


def get_feature_engineering_pipeline():
    numeric_features = [
        "Amount", "PricingStrategy",
        "total_amount", "avg_amount", "count_transactions", "std_amount",
        "transaction_hour", "transaction_day", "transaction_month", "transaction_year"
    ]
    categorical_features = ["ProductCategory", "ChannelId", "ProviderId"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", KNNImputer(n_neighbors=5)),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    full_pipeline = Pipeline(steps=[
        ("agg_features", AggregateFeaturesAdder()),
        ("datetime_features", DatetimeFeaturesExtractor()),
        ("drop_customer_id", DropCustomerId()),
        ("preprocessing", preprocessor),
    ])

    return full_pipeline
