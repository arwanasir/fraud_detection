def create_time_features(df):
    df = df.copy()

    df['time_since_signup_hours'] = (
        df['purchase_time'] - df['signup_time']
    ).dt.total_seconds() / 3600

    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek

    return df


def analyze_fraud_by_country(df):

    if 'country' not in df.columns:
        print("Error: No country column found")
        return None
    country_stats = df.groupby('country').agg(
        total_transactions=('class', 'count'),
        fraud_count=('class', 'sum'),
        fraud_percentage=('class', 'mean')
    ).reset_index()

    country_stats['fraud_percentage'] = country_stats['fraud_percentage'] * 100
    country_stats = country_stats.sort_values(
        'fraud_percentage', ascending=False)

    return country_stats
