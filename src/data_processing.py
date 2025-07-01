import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class AggregateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, group_col='CustomerId', agg_col='Amount'):
        self.group_col = group_col
        self.agg_col = agg_col
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Aggregate per customer
        agg_df = X.groupby(self.group_col)[self.agg_col].agg(
            total_amount='sum',
            avg_amount='mean',
            count_transactions='count',
            std_amount='std'
        ).reset_index()
        # Merge aggregates back to original data on CustomerId
        X = pd.merge(X, agg_df, on=self.group_col, how='left')
        # Fill NaN std_amount with 0 (if only one transaction)
        X['std_amount'] = X['std_amount'].fillna(0)
        return X

class DatetimeFeatures(BaseEstimator, TransformerMixin):
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



def get_preprocessor(numeric_features, categorical_features):
    """
    Creates and returns a ColumnTransformer for preprocessing.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    return preprocessor

def preprocess_data(df, preprocessor, numeric_features, categorical_features):
    """
    Applies preprocessing to the DataFrame and returns a new DataFrame
    with proper column names.
    """
    # Fit and transform the data
    X_transformed = preprocessor.fit_transform(df)

    # Access the OneHotEncoder inside the ColumnTransformer
    ohe = preprocessor.named_transformers_['cat']

    # Get the one-hot encoded feature names
    onehot_feature_names = ohe.get_feature_names_out(categorical_features)

    # Combine all feature names
    all_feature_names = list(numeric_features) + list(onehot_feature_names)

    # Convert sparse matrix to dense if necessary (ColumnTransformer often outputs sparse)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    # Create DataFrame
    X_df = pd.DataFrame(X_transformed, columns=all_feature_names)

    return X_df

# Example of how you might define features if they are constant for your project
# numeric_features = ['Amount', 'PricingStrategy']
# categorical_features = ['ProductCategory', 'ChannelId', 'ProviderId']    



def build_feature_engineering_pipeline():
    pipeline = Pipeline([
        ('aggregate_features', AggregateFeatures()),
        ('datetime_features', DatetimeFeatures()),
        # Add more steps here later: encoding, scaling etc.
    ])
    return pipeline
