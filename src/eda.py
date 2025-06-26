import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import os

def main():
    # Load data
    data_path = os.path.join('..', 'data', 'raw', 'data.xlsx')
    df = pd.read_excel(data_path)
    print(f"Data loaded. Shape: {df.shape}")

    # Basic info
    print(df.info())

    # Summary statistics
    print(df.describe(include='all').T)

    # Missing values
    print("\nMissing values count per column:")
    print(df.isnull().sum())

    # Plot missing values matrix
    plt.figure(figsize=(12, 6))
    msno.matrix(df)
    plt.title("Missing Values Matrix")
    plt.savefig('missing_values_matrix.png')
    plt.close()

    # Categorical columns to explore
    categorical_cols = ['CurrencyCode', 'CountryCode', 'ProviderId', 'ProductCategory', 'ChannelId', 'PricingStrategy', 'FraudResult']

    for col in categorical_cols:
        plt.figure(figsize=(10, 4))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index)
        plt.xticks(rotation=45)
        plt.title(f'Distribution of {col}')
        plt.tight_layout()
        filename = f'cat_dist_{col}.png'
        plt.savefig(filename)
        plt.close()
        print(f'Saved plot: {filename}')

    # Numerical columns
    numerical_cols = ['Amount', 'Value']

    for col in numerical_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], bins=50, kde=True)
        plt.title(f'Distribution of {col}')
        plt.tight_layout()
        filename = f'num_dist_{col}.png'
        plt.savefig(filename)
        plt.close()
        print(f'Saved plot: {filename}')

    # Boxplots for outliers
    for col in numerical_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f'Outliers in {col}')
        plt.tight_layout()
        filename = f'boxplot_{col}.png'
        plt.savefig(filename)
        plt.close()
        print(f'Saved plot: {filename}')

    # Correlation heatmap
    plt.figure(figsize=(6, 5))
    corr = df[numerical_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()
    print('Saved plot: correlation_heatmap.png')

if __name__ == "__main__":
    main()
