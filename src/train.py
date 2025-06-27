import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import mlflow
import mlflow.sklearn

def main():
    # Load data
    X = pd.read_csv("data/processed/processed_data.csv")
    y = pd.read_csv("data/processed/target_labels.csv")

    # Merge on AccountId
    df = pd.merge(X, y, on="AccountId")

    # Prepare features and target
    X = df.drop(columns=["AccountId", "is_high_risk"])
    y = df["is_high_risk"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define models pipelines with SimpleImputer
    imputer = SimpleImputer(strategy="mean")
    lr = make_pipeline(imputer, LogisticRegression(max_iter=1000, random_state=42))
    rf = make_pipeline(imputer, RandomForestClassifier(n_estimators=100, random_state=42))

    # Train models
    print("Training Logistic Regression...")
    lr.fit(X_train, y_train)

    print("Training Random Forest...")
    rf.fit(X_train, y_train)

    # Predict on test set
    y_pred_lr = lr.predict(X_test)
    y_pred_rf = rf.predict(X_test)

    # Evaluate with F1 score
    f1_lr = f1_score(y_test, y_pred_lr)
    f1_rf = f1_score(y_test, y_pred_rf)

    print(f"Logistic Regression F1 score: {f1_lr:.4f}")
    print(f"Random Forest F1 score: {f1_rf:.4f}")

    # Determine best model
    if f1_rf >= f1_lr:
        best_model = rf
        best_score = f1_rf
        best_name = "Random Forest"
    else:
        best_model = lr
        best_score = f1_lr
        best_name = "Logistic Regression"

    print(f"Logging and registering best model: {best_name} with F1 score {best_score:.4f}")

    # Log model and metric to MLflow and register
    with mlflow.start_run():
        mlflow.sklearn.log_model(best_model, "model", registered_model_name="best_model")
        mlflow.log_metric("f1_score", best_score)

    print("Model logged and registered as 'best_model'")

if __name__ == "__main__":
    main()
