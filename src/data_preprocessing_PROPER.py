import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def efficient_one_hot_encoding(df, categorical_cols, max_categories=10):
    df_encoded = df.copy()

    for col in categorical_cols:
        if col in df_encoded.columns:
            top_categories = df_encoded[col].value_counts().head(
                max_categories).index.tolist()

            df_encoded[col] = df_encoded[col].apply(
                lambda x: x if x in top_categories else 'Other'
            )

            dummies = pd.get_dummies(
                df_encoded[col], prefix=col, drop_first=True)
            df_encoded = df_encoded.drop(columns=[col])
            df_encoded = pd.concat([df_encoded, dummies], axis=1)

    return df_encoded


def efficient_scaling(df, numerical_cols):

    df_scaled = df.copy()

    for col in numerical_cols:
        if col in df_scaled.columns:
            if df_scaled[col].std() > 0:
                df_scaled[col] = (
                    df_scaled[col] - df_scaled[col].mean()) / df_scaled[col].std()
            else:
                df_scaled[col] = 0

    return df_scaled


def efficient_undersampling(X_train, y_train, sample_fraction=0.5):
    print("\nApplying efficient undersampling...")

    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train)

    fraud_indices = y_train[y_train == 1].index
    legit_indices = y_train[y_train == 0].index

    print(f"Fraud samples: {len(fraud_indices)}")
    print(f"Legitimate samples: {len(legit_indices)}")

    n_fraud = len(fraud_indices)
    n_legit_to_sample = int(n_fraud * sample_fraction)

    if n_legit_to_sample > len(legit_indices):
        n_legit_to_sample = len(legit_indices)

    sampled_legit_indices = np.random.choice(
        legit_indices, size=n_legit_to_sample, replace=False
    )

    selected_indices = np.concatenate([fraud_indices, sampled_legit_indices])
    X_balanced = X_train.loc[selected_indices].reset_index(drop=True)
    y_balanced = y_train.loc[selected_indices].reset_index(drop=True)
    shuffle_idx = np.random.permutation(len(X_balanced))
    X_balanced = X_balanced.iloc[shuffle_idx]
    y_balanced = y_balanced.iloc[shuffle_idx]

    print(f"After undersampling:")
    print(f"  Total samples: {len(X_balanced)}")
    print(
        f"  Fraud: {(y_balanced == 1).sum()} ({(y_balanced == 1).mean()*100:.1f}%)")
    print(
        f"  Legitimate: {(y_balanced == 0).sum()} ({(y_balanced == 0).mean()*100:.1f}%)")

    return X_balanced, y_balanced


def task1(df, target_col='class'):
    print(f"\n1. Preparing data...")
    y = df[target_col]
    X = df.drop(columns=[target_col])

    print(f"   Original shape: {df.shape}")
    print(f"   Target distribution: {y.value_counts().to_dict()}")

    print(f"\n2. Selecting key features to prevent memory issues...")

    keep_columns = [
        'purchase_value', 'age', 'time_since_signup_hours',
        'hour_of_day', 'day_of_week', 'source', 'browser', 'sex'
    ]

    existing_columns = [col for col in keep_columns if col in X.columns]
    X = X[existing_columns]

    print(f"   Using columns: {existing_columns}")

    print(f"\n3. Applying efficient One-Hot Encoding...")
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(
        include=['int64', 'float64']).columns.tolist()

    print(f"   Categorical columns: {categorical_cols}")
    print(f"   Numerical columns: {numerical_cols}")

    if categorical_cols:
        X_encoded = efficient_one_hot_encoding(
            X, categorical_cols, max_categories=5)
        print(f"   Shape after encoding: {X_encoded.shape}")
    else:
        X_encoded = X
        print(f"   No categorical columns to encode")

    print(f"\n4. Applying feature scaling...")
    if numerical_cols:
        X_scaled = efficient_scaling(X_encoded, numerical_cols)
        print(f"   Scaled columns: {numerical_cols}")
    else:
        X_scaled = X_encoded

    print(f"\n5. Creating train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    print(f"   Training fraud %: {y_train.mean()*100:.2f}%")

    print(f"\n6. Handling class imbalance with undersampling...")
    X_train_bal, y_train_bal = efficient_undersampling(X_train, y_train)

    print(f"\n7. Saving processed data...")

    X_train_bal.to_csv('../data/processed/X_train_bal_small.csv', index=False)
    y_train_bal.to_csv('../data/processed/y_train_bal_small.csv', index=False)
    X_test.to_csv('../data/processed/X_test_small.csv', index=False)
    y_test.to_csv('../data/processed/y_test_small.csv', index=False)

    print(f"   Saved: X_train_bal_small.csv ({X_train_bal.shape})")
    print(f"   Saved: y_train_bal_small.csv ({len(y_train_bal)})")
    print(f"   Saved: X_test_small.csv ({X_test.shape})")
    print(f"   Saved: y_test_small.csv ({len(y_test)})")

    return X_train_bal, X_test, y_train_bal, y_test
