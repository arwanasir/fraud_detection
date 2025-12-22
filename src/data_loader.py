import pandas as pd


def load_fraud_data(filepath='../data/raw/Fraud_Data.csv'):

    df = pd.read_csv(filepath, parse_dates=['signup_time', 'purchase_time'])
    df = df.drop_duplicates()

    if df.isnull().sum().sum() > 0:
        print(f"Dropping {df.isnull().sum().sum()} rows with missing values")
        df = df.dropna()

    return df


def load_creditcard_data(filepath='../data/raw/creditcard.csv.zip'):
    df = pd.read_csv(filepath)
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"Found {missing} missing values in creditcard data")

    return df


def load_country_mapping(filepath='../data/raw/IpAddress_to_Country.csv'):

    df = pd.read_csv(filepath)
    return df
