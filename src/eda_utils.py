import matplotlib.pyplot as plt
import seaborn as sns


def plot_class_distribution(df, target_col='class'):
    fraud_counts = df[target_col].value_counts()
    fraud_percent = df[target_col].mean() * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fraud_counts.plot(kind='bar', ax=ax1, color=['green', 'red'])
    ax1.set_title('Fraud vs Non-Fraud Transactions')
    ax1.set_xlabel('Class (0=Legit, 1=Fraud)')
    ax1.set_ylabel('Count')

    for i, v in enumerate(fraud_counts):
        ax1.text(i, v + max(fraud_counts)*0.01, str(v), ha='center')

    ax2.pie(fraud_counts, labels=['Legitimate', 'Fraud'], autopct='%1.1f%%',
            colors=['green', 'red'], startangle=90)
    ax2.set_title(f'Fraud Rate: {fraud_percent:.2f}%')

    plt.tight_layout()
    plt.show()

    print(f"Total transactions: {len(df)}")
    print(f"Legitimate (0): {fraud_counts[0]}")
    print(f"Fraudulent (1): {fraud_counts[1]}")
    print(f"Fraud percentage: {fraud_percent:.2f}%")


def plot_fraud_by_category(df, category_col):
    fraud_by_cat = df.groupby(category_col)['class'].mean() * 100
    fraud_by_cat = fraud_by_cat.sort_values(ascending=False)

    plt.figure(figsize=(10, 5))
    bars = plt.bar(fraud_by_cat.index.astype(str), fraud_by_cat.values)
    plt.title(f'Fraud Percentage by {category_col}')
    plt.xlabel(category_col)
    plt.ylabel('Fraud Percentage (%)')
    plt.xticks(rotation=45)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def plot_fraud_by_hour(df):
    if 'hour_of_day' not in df.columns:
        df = df.copy()
        df['hour_of_day'] = df['purchase_time'].dt.hour

    fraud_by_hour = df[df['class'] == 1].groupby('hour_of_day').size()
    total_by_hour = df.groupby('hour_of_day').size()
    fraud_rate_by_hour = (fraud_by_hour / total_by_hour * 100).fillna(0)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    fraud_by_hour.plot(kind='bar', color='red')
    plt.title('Number of Fraud Cases by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Fraud Count')
    plt.subplot(1, 2, 2)
    fraud_rate_by_hour.plot(kind='bar', color='orange')
    plt.title('Fraud Rate by Hour (%)')
    plt.xlabel('Hour of Day')
    plt.ylabel('Fraud Rate (%)')

    plt.tight_layout()
    plt.show()
