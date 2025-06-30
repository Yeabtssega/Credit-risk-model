import pandas as pd

def is_risk_label_valid(label):
    """Helper function to check if a risk label is 0 or 1."""
    return label in (0, 1)

def calculate_rfm(df, snapshot_date=None):
    """Calculate RFM metrics for customers from transaction data DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with columns CustomerId, TransactionStartTime, TransactionId, Amount.
        snapshot_date (str or pd.Timestamp, optional): Date to use as reference for Recency calculation.
            If None, uses the max TransactionStartTime in the data.

    Returns:
        pd.DataFrame: DataFrame with columns CustomerId, Recency, Frequency, Monetary.
    """
    # Convert TransactionStartTime to datetime
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"], errors='coerce')

    # Drop rows with missing critical values
    df_clean = df.dropna(subset=["CustomerId", "TransactionStartTime", "Amount"])

    # Determine snapshot_date
    if snapshot_date is None:
        snapshot_date = df_clean["TransactionStartTime"].max()
    else:
        snapshot_date = pd.to_datetime(snapshot_date)

    # Group by CustomerId to compute RFM
    rfm = df_clean.groupby("CustomerId").agg({
        "TransactionStartTime": lambda x: (snapshot_date - x.max()).days,
        "TransactionId": "count",
        "Amount": "sum"
    }).reset_index()

    rfm.columns = ["CustomerId", "Recency", "Frequency", "Monetary"]

    return rfm
