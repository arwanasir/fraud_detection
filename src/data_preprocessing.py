import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.utils.validation import _is_pandas_df
from imblearn.under_sampling import RandomUnderSampler


def prepare_features_for_modeling(df, target_col='class'):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    categorical_cols = X.select_dtypes(
        include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(
        include=['int64', 'float64']).columns.tolist()

    return X, y, categorical_cols, numerical_cols


def encode_categorical_features(X, categorical_cols):

    if len(categorical_cols) == 0:
        print("No categorical columns to encode")
        return X, None

    print(f"Encoding categorical columns: {categorical_cols}")
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_array = encoder.fit_transform(X[categorical_cols])
    encoded_features = encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(
        encoded_array, columns=encoded_features, index=X.index)

    X_encoded = X.drop(columns=categorical_cols)
    X_encoded = pd.concat([X_encoded, encoded_df], axis=1)

    return X_encoded, encoder


def scale_numerical_features(X, numerical_cols, scaler_type='standard'):

    if len(numerical_cols) == 0:
        print("No numerical columns to scale")
        return X, None

    print(f"Scaling numerical columns: {numerical_cols}")
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("scaler_type must be 'standard' or 'minmax'")
    X_scaled = X.copy()
    X_scaled[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    return X_scaled, scaler


def handle_class_imbalance(X_train, y_train, method='smote', random_state=42):
    print("\n Handling Class Imbalance ")
    print(f"Before resampling - Class distribution:")
    print(y_train.value_counts())
    print(f"Fraud percentage: {y_train.mean()*100:.2f}%")

    if method == 'smote':
        print("\nApplying SMOTE (Synthetic Minority Over-sampling)...")
        sampler = SMOTE(random_state=random_state)
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

    elif method == 'undersample':
        print("\nApplying Random Under-sampling...")
        sampler = RandomUnderSampler(random_state=random_state)
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

    else:
        raise ValueError("method must be 'smote' or 'undersample'")

    print(f"\nAfter resampling - Class distribution:")
    print(pd.Series(y_resampled).value_counts())
    print(f"Fraud percentage: {y_resampled.mean()*100:.2f}%")

    return X_resampled, y_resampled, sampler


def split_data_stratified(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"\n Stratified Train-Test Split ")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"\nClass distribution in training set:")
    print(y_train.value_counts(normalize=True))
    print(f"\nClass distribution in test set:")
    print(y_test.value_counts(normalize=True))

    return X_train, X_test, y_train, y_test
