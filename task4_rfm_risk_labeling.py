# task4_rfm_risk_labeling.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load your data (updated path)
df = pd.read_excel(r"C:\Users\HP\credit-risk-model\data\raw\data.xlsx", sheet_name="data")

# Ensure correct datetime format
df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"], errors='coerce')

# Clean data: drop missing CustomerId, Amount, or TransactionStartTime
df_clean = df.dropna(subset=["CustomerId", "TransactionStartTime", "Amount"])

# Set snapshot date as the most recent transaction date
snapshot_date = df_clean["TransactionStartTime"].max()

# Calculate RFM metrics
rfm = df_clean.groupby("CustomerId").agg({
    "TransactionStartTime": lambda x: (snapshot_date - x.max()).days,
    "TransactionId": "count",
    "Amount": "sum"
}).reset_index()

rfm.columns = ["CustomerId", "Recency", "Frequency", "Monetary"]

# Scale RFM values
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

# Analyze clusters
cluster_summary = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean().round(2)
print("Cluster summary:\n", cluster_summary)

# Identify high-risk cluster: lowest Frequency and Monetary
high_risk_cluster = cluster_summary.sort_values(by=["Frequency", "Monetary"]).index[0]
print("High-risk cluster identified as:", high_risk_cluster)

# Create binary target variable
rfm["is_high_risk"] = (rfm["Cluster"] == high_risk_cluster).astype(int)

# Merge back to original dataset (optional, for modeling)
df_final = df_clean.merge(rfm[["CustomerId", "is_high_risk"]], on="CustomerId", how="left")

# Save to file
df_final.to_csv("labeled_data.csv", index=False)
print("✅ Final dataset with 'is_high_risk' label saved to labeled_data.csv")
