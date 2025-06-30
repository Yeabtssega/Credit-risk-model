import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings("ignore")

# Load labeled data
df = pd.read_csv("labeled_data.csv")

# Basic feature selection
features = ["Amount", "Value", "PricingStrategy", "FraudResult"]
df = df.dropna(subset=features + ["is_high_risk"])

X = df[features]
y = df["is_high_risk"]

# Encode categorical features
X = pd.get_dummies(X, columns=["PricingStrategy", "FraudResult"], drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Enable MLflow experiment tracking
mlflow.set_experiment("Credit Risk Model")

with mlflow.start_run():

    # Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr_params = {'C': [0.01, 0.1, 1, 10]}
    grid_lr = GridSearchCV(lr, lr_params, cv=5)
    grid_lr.fit(X_train, y_train)
    best_lr = grid_lr.best_estimator_
    
    # Predictions
    y_pred_lr = best_lr.predict(X_test)
    y_prob_lr = best_lr.predict_proba(X_test)[:, 1]

    # Evaluation
    metrics_lr = {
        "Accuracy": accuracy_score(y_test, y_pred_lr),
        "Precision": precision_score(y_test, y_pred_lr),
        "Recall": recall_score(y_test, y_pred_lr),
        "F1": f1_score(y_test, y_pred_lr),
        "ROC_AUC": roc_auc_score(y_test, y_prob_lr)
    }

    # Log to MLflow
    mlflow.log_params(grid_lr.best_params_)
    mlflow.log_metrics(metrics_lr)
    mlflow.sklearn.log_model(best_lr, "model", registered_model_name="credit-risk-model-logreg")

    print("✅ Logistic Regression logged and registered.")
    print("Logistic Regression metrics:", metrics_lr)

with mlflow.start_run():

    # Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf_params = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
    grid_rf = GridSearchCV(rf, rf_params, cv=5)
    grid_rf.fit(X_train, y_train)
    best_rf = grid_rf.best_estimator_

    y_pred_rf = best_rf.predict(X_test)
    y_prob_rf = best_rf.predict_proba(X_test)[:, 1]

    metrics_rf = {
        "Accuracy": accuracy_score(y_test, y_pred_rf),
        "Precision": precision_score(y_test, y_pred_rf),
        "Recall": recall_score(y_test, y_pred_rf),
        "F1": f1_score(y_test, y_pred_rf),
        "ROC_AUC": roc_auc_score(y_test, y_prob_rf)
    }

    mlflow.log_params(grid_rf.best_params_)
    mlflow.log_metrics(metrics_rf)
    mlflow.sklearn.log_model(best_rf, "model", registered_model_name="credit-risk-model-rf")

    print("✅ Random Forest logged and registered.")
    print("Random Forest metrics:", metrics_rf)
