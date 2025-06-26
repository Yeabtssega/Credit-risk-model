# src/create_target.py

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# Load raw data
raw_data_path = os.path.join('..', 'data', 'raw', 'data.xlsx')
df = pd.read_excel(raw_data_path)
df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')

# Snapshot date = max date in dataset + 1
snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)

# Step 1: Calculate RFM
rfm = df.groupby('AccountId').agg({
    'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
    'TransactionId': 'count',
    'Amount': 'sum'
}).reset_index()

rfm.columns = ['AccountId', 'Recency', 'Frequency', 'Monetary']

# Step 2: Scale RFM
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# Step 3: Cluster with KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Step 4: Identify high-risk cluster
# High risk = high recency (less recent), low frequency and low monetary
cluster_summary = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
high_risk_cluster = cluster_summary.sort_values(by=['Recency', 'Frequency', 'Monetary'], ascending=[False, True, True]).index[0]

rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)

# Step 5: Save target labels
target_path = os.path.join('..', 'data', 'processed', 'target_labels.csv')
rfm[['AccountId', 'is_high_risk']].to_csv(target_path, index=False)

print(f"✅ Saved is_high_risk labels to: {target_path}")
print(rfm['is_high_risk'].value_counts())
