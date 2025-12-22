import pandas as pd
import numpy as np


def ip_to_int(ip_address):

    try:
        if pd.isna(ip_address):
            return 0

        ip_str = str(ip_address).strip()

        if '.' not in ip_str:
            return 0

        parts = ip_str.split('.')
        if len(parts) != 4:
            return 0

        for part in parts:
            if not part.isdigit():
                return 0
        return (int(parts[0]) * 16777216) + \
               (int(parts[1]) * 65536) + \
               (int(parts[2]) * 256) + \
            int(parts[3])

    except Exception as e:
        print(f"Warning: Could not convert IP {ip_address}: {e}")
        return 0


def add_ip_integer_columns(fraud_df, country_df):
    fraud_df = fraud_df.copy()
    country_df = country_df.copy()

    print(f"Converting {len(fraud_df)} fraud IPs to integers...")
    fraud_df['ip_int'] = fraud_df['ip_address'].apply(ip_to_int)

    print(f"Converting {len(country_df)} country mapping IPs to integers...")
    country_df['lower_int'] = country_df['lower_bound_ip_address'].apply(
        ip_to_int)
    country_df['upper_int'] = country_df['upper_bound_ip_address'].apply(
        ip_to_int)
    zero_count_fraud = (fraud_df['ip_int'] == 0).sum()
    zero_count_country = ((country_df['lower_int'] == 0) | (
        country_df['upper_int'] == 0)).sum()

    if zero_count_fraud > 0:
        print(
            f"Warning: {zero_count_fraud} fraud IPs could not be converted (set to 0)")

    if zero_count_country > 0:
        print(
            f"Warning: {zero_count_country} country mapping IPs could not be converted (set to 0)")

    return fraud_df, country_df


def merge_with_country(fraud_df, country_df):
    valid_fraud_mask = fraud_df['ip_int'] > 0
    valid_country_mask = (country_df['lower_int'] > 0) & (
        country_df['upper_int'] > 0)

    fraud_df_valid = fraud_df[valid_fraud_mask].copy()
    country_df_valid = country_df[valid_country_mask].copy()

    print(f"Valid fraud IPs: {len(fraud_df_valid)}/{len(fraud_df)}")
    print(f"Valid country mappings: {len(country_df_valid)}/{len(country_df)}")

    if len(fraud_df_valid) == 0 or len(country_df_valid) == 0:
        print("Error: No valid IPs to merge")
        return fraud_df.assign(country=None)
    fraud_df_valid = fraud_df_valid.sort_values('ip_int')
    country_df_valid = country_df_valid.sort_values('lower_int')

    print("Merging with country data...")
    merged_df = pd.merge_asof(
        fraud_df_valid,
        country_df_valid[['lower_int', 'upper_int', 'country']],
        left_on='ip_int',
        right_on='lower_int',
        direction='backward'
    )

    valid_mask = (merged_df['ip_int'] >= merged_df['lower_int']) & \
                 (merged_df['ip_int'] <= merged_df['upper_int'])

    merged_valid = merged_df[valid_mask].copy()
    merged_invalid = merged_df[~valid_mask].copy()

    print(f"Successfully mapped: {len(merged_valid)} transactions")
    print(f"Could not map: {len(merged_invalid)} transactions")
    if len(merged_invalid) > 0:
        merged_invalid['country'] = None
        final_df = pd.concat([merged_valid, merged_invalid], ignore_index=True)
    else:
        final_df = merged_valid
    if len(fraud_df[~valid_fraud_mask]) > 0:
        unmapped = fraud_df[~valid_fraud_mask].copy()
        unmapped['country'] = None
        unmapped['lower_int'] = 0
        unmapped['upper_int'] = 0
        final_df = pd.concat([final_df, unmapped], ignore_index=True)

    print(f"Final merged dataset: {len(final_df)} transactions")
    print(
        f"Transactions with country mapped: {(final_df['country'].notna().sum() / len(final_df) * 100):.1f}%")

    return final_df
