import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def main():
    # Paths (adjust if needed)
    features_path = "../data/processed/processed_data.csv"
    target_path = "../data/processed/target_labels.csv"

    # Load features and target
    X = pd.read_csv(features_path)
    y = pd.read_csv(target_path)

    # Extract label column from target (assumes single column 'is_high_risk' or first column)
    if "is_high_risk" in y.columns:
        y = y["is_high_risk"]
    else:
        y = y.iloc[:, 0]

    # Check row count match
    if len(X) != len(y):
        print(
            f"Error: Number of samples in features ({len(X)}) and target ({len(y)}) do not match."
        )
        return

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define models and hyperparameters
    models = {
        "logistic_regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {
                "C": [0.01, 0.1, 1, 10],
                "penalty": ["l2"],
                "solver": ["lbfgs"],
            },
        },
        "random_forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [50, 100],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5],
            },
        },
    }

    best_model_name = None
    best_model = None
    best_f1 = 0

    mlflow.set_experiment("credit_risk_model_training")

    for name, mp in models.items():
        print(f"Training and tuning {name}...")

        with mlflow.start_run(run_name=name):
            clf = GridSearchCV(mp["model"], mp["params"], cv=3, scoring="f1", n_jobs=-1)
            clf.fit(X_train, y_train)

            best_estimator = clf.best_estimator_

            y_pred = best_estimator.predict(X_test)
            y_proba = None
            if hasattr(best_estimator, "predict_proba"):
                y_proba = best_estimator.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

            mlflow.log_params(clf.best_params_)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            if roc_auc is not None:
                mlflow.log_metric("roc_auc", roc_auc)

            mlflow.sklearn.log_model(best_estimator, artifact_path=name + "_model")

            print(f"{name} done. F1 score: {f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_model_name = name
                best_model = best_estimator

    print(f"✅ Best model: {best_model_name} with F1 score: {best_f1:.4f}")


if __name__ == "__main__":
    main()
