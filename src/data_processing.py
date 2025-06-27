import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_and_process_data(filepath):
    # Load data
    df = pd.read_excel(filepath)

    # Drop irrelevant columns
    df = df.drop(columns=["Unnamed: 16", "Unnamed: 17"])

    # Convert TransactionStartTime to datetime
    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"], errors="coerce"
    )

    # Extract datetime features
    df["TransactionYear"] = df["TransactionStartTime"].dt.year
    df["TransactionMonth"] = df["TransactionStartTime"].dt.month
    df["TransactionDay"] = df["TransactionStartTime"].dt.day
    df["TransactionWeekday"] = df["TransactionStartTime"].dt.weekday

    # Drop original datetime column if not needed
    df = df.drop(columns=["TransactionStartTime"])

    # Define feature groups
    numeric_features = [
        "Amount",
        "Value",
        "PricingStrategy",
        "FraudResult",
        "TransactionYear",
        "TransactionMonth",
        "TransactionDay",
        "TransactionWeekday",
    ]

    categorical_features = [
        "CurrencyCode",
        "CountryCode",
        "ProviderId",
        "ProductId",
        "ProductCategory",
        "ChannelId",
    ]

    # Preprocessing pipeline
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Fit and transform
    X_processed = preprocessor.fit_transform(df)

    # Convert to DataFrame with feature names
    cat_cols = preprocessor.named_transformers_["cat"].get_feature_names_out(
        categorical_features
    )
    processed_columns = numeric_features + list(cat_cols)
    X_processed_df = pd.DataFrame(X_processed, columns=processed_columns)

    return X_processed_df, df


if __name__ == "__main__":
    features_path = "../data/raw/data.xlsx"
    X_processed, original_df = load_and_process_data(features_path)
    print(f"Processed features shape: {X_processed.shape}")
    X_processed.to_csv("../data/processed/processed_data.csv", index=False)
