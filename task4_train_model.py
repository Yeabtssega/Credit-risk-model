import os
import pandas as pd
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Set MLflow tracking URI and experiment
mlflow.set_tracking_uri("file:///C:/Users/HP/credit-risk-model/mlruns")
mlflow.set_experiment("Credit Risk Model")

# Load the Excel data
data_path = "data/raw/data.xlsx"
df = pd.read_excel(data_path, sheet_name="data")

# Select features and target
features = ["Amount", "Value", "PricingStrategy"]
target = "FraudResult"

X = df[features]
y = df[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ Check class distribution
print("✅ y_train class distribution:")
print(y_train.value_counts())

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)

try:
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
except Exception as e:
    print("⚠️ Warning: Model does not support predict_proba:", e)
    y_pred_prob = [0.5 for _ in y_pred]  # Dummy probabilities

print("✅ Classification Report:")
print(classification_report(y_test, y_pred))

# MLflow logging
with mlflow.start_run() as run:
    mlflow.log_params({"n_estimators": 100, "model_type": "RandomForest"})
    mlflow.sklearn.log_model(clf, "model")

    print(f"✅ Logged model to run ID: {run.info.run_id}")

    # Save locally
    model_path = "models/mlflow_model"
    os.makedirs(model_path, exist_ok=True)
    joblib.dump(clf, os.path.join(model_path, "model.pkl"))
    print(f"✅ Model saved locally to: {model_path}")
