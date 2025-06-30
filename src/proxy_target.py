import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- RFM Calculation ---
def calculate_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Compute Recency, Frequency, and Monetary (RFM) metrics for each customer.

    Args:
        df (pd.DataFrame): The input DataFrame with transaction data,
                           expected to have 'CustomerId', 'TransactionStartTime',
                           'TransactionId', and 'Amount' columns.
        snapshot_date (pd.Timestamp, optional): The date to calculate recency from.
                                               If None, it defaults to the day after
                                               the latest transaction in the DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with 'CustomerId', 'recency', 'frequency',
                      and 'monetary' columns.
    """
    df = df.copy() # Work on a copy to avoid modifying the original DataFrame
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    if snapshot_date is None:
        snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerId").agg(
        # Recency: Days since last transaction
        recency=("TransactionStartTime", lambda x: (snapshot_date - x.max()).days),
        # Frequency: Total number of transactions
        frequency=("TransactionId", "count"),
        # Monetary: Sum of transaction amounts
        monetary=("Amount", "sum")
    ).reset_index()

    return rfm

# --- RFM Scaling ---
def scale_rfm(rfm_df: pd.DataFrame):
    """
    Standardize the Recency, Frequency, and Monetary (RFM) features using StandardScaler.

    Args:
        rfm_df (pd.DataFrame): DataFrame containing 'recency', 'frequency', and 'monetary' columns.

    Returns:
        tuple: A tuple containing:
            - rfm_scaled (np.ndarray): A NumPy array of the scaled RFM features.
            - scaler (StandardScaler): The fitted StandardScaler object.
    """
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[["recency", "frequency", "monetary"]])
    return rfm_scaled, scaler

# --- K-Means Clustering ---
def perform_kmeans_clustering(rfm_scaled: pd.DataFrame, n_clusters: int = 3, random_state: int = 42):
    """
    Cluster customers based on their scaled RFM values using K-Means.

    Args:
        rfm_scaled (pd.DataFrame): Scaled RFM features (NumPy array or DataFrame).
        n_clusters (int, optional): The number of clusters to form. Defaults to 3.
        random_state (int, optional): Determines random number generation for
                                      centroid initialization. Defaults to 42.

    Returns:
        tuple: A tuple containing:
            - cluster_labels (np.ndarray): An array of cluster labels for each customer.
            - kmeans (KMeans): The fitted KMeans model object.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto') # Added n_init='auto' for newer sklearn versions
    cluster_labels = kmeans.fit_predict(rfm_scaled)
    return cluster_labels, kmeans

# --- High-Risk Customer Labeling (Rule-Based) ---
def label_high_risk_customers(rfm_df: pd.DataFrame, recency_thresh: int = 60, freq_thresh: int = 2, monetary_thresh: float = 0.0):
    """
    Label customers as high-risk (1) or not (0) based on predefined RFM thresholds.

    Args:
        rfm_df (pd.DataFrame): DataFrame containing 'recency', 'frequency', and 'monetary' columns.
        recency_thresh (int, optional): Threshold for recency (days). Customers with recency >= this are higher risk. Defaults to 60.
        freq_thresh (int, optional): Threshold for frequency. Customers with frequency <= this are higher risk. Defaults to 2.
        monetary_thresh (float, optional): Threshold for monetary value. Customers with monetary <= this are higher risk. Defaults to 0.

    Returns:
        pd.DataFrame: The input DataFrame with an added 'is_high_risk' column.
    """
    rfm_df["is_high_risk"] = (
        (rfm_df["recency"] >= recency_thresh) &
        (rfm_df["frequency"] <= freq_thresh) &
        (rfm_df["monetary"] <= monetary_thresh)
    ).astype(int)
    return rfm_df

