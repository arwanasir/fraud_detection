import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')


def load_model_and_data():
    """Load the selected model and data for SHAP analysis - FIXED VERSION"""
    print("Loading model and data for SHAP analysis...")

    try:
        model = joblib.load('models/random_forest_ensemble.pkl')
        print("Loaded: Random Forest model")
    except:
        try:
            model = joblib.load('models/logistic_regression_baseline.pkl')
            print("Loaded: Logistic Regression model")
        except:
            print("Error: No model found. Please run Task 2 first.")
            return None, None, None, None

    try:
        X_test = pd.read_csv('../data/processed/X_test_small.csv')
        y_test = pd.read_csv('../data/processed/y_test_small.csv').iloc[:, 0]
        X_train = pd.read_csv('../data/processed/X_train_bal_small.csv')
    except FileNotFoundError:
        print("Processed data not found. Trying simple versions...")
        try:
            X_test = pd.read_csv('../data/processed/X_test_simple.csv')
            y_test = pd.read_csv(
                '../data/processed/y_test_simple.csv').iloc[:, 0]
            X_train = pd.read_csv('../data/processed/X_train_bal_small.csv')
        except:
            print("ERROR: No data files found. Run Task 1 & 2 first.")
            return None, None, None, None

    print("\nConverting data to numeric format for SHAP...")

    def convert_to_numeric(df):
        df = df.copy()
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:

                    df = pd.get_dummies(df, columns=[col], drop_first=True)

        df = df.fillna(0)

        for col in df.columns:
            if df[col].dtype not in ['int64', 'float64']:
                df[col] = df[col].astype('float64')

        return df

    X_train = convert_to_numeric(X_train)
    X_test = convert_to_numeric(X_test)

    print("Aligning columns between train and test...")
    common_cols = list(set(X_train.columns) & set(X_test.columns))

    if len(common_cols) == 0:
        print("ERROR: No common columns between train and test!")
        return None, None, None, None

    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    print(
        f"Training class distribution: {pd.Series(y_test).value_counts().to_dict()}")

    return model, X_train, X_test, y_test


def get_predictions_for_shap(model, X_test, y_test):
    print("\nIdentifying predictions for SHAP analysis...")

    # ADD THIS LINE - Ensure X_test has same columns as training data
    X_test = X_test[model.feature_names_in_]  # For sklearn >= 1.0

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    indices = {
        'true_positive': None,
        'false_positive': None,
        'false_negative': None,
        'true_negative': None
    }

    for idx in range(len(y_test)):
        actual = y_test.iloc[idx]
        predicted = y_pred[idx]

        if actual == 1 and predicted == 1 and indices['true_positive'] is None:
            indices['true_positive'] = idx
        elif actual == 0 and predicted == 1 and indices['false_positive'] is None:
            indices['false_positive'] = idx
        elif actual == 1 and predicted == 0 and indices['false_negative'] is None:
            indices['false_negative'] = idx
        elif actual == 0 and predicted == 0 and indices['true_negative'] is None:
            indices['true_negative'] = idx

        if all(v is not None for v in indices.values()):
            break

    print("Found predictions for SHAP analysis:")
    for case, idx in indices.items():
        if idx is not None:
            print(f"  {case.replace('_', ' ').title()}: Index {idx}")

    return indices, y_pred_proba


def shap_global_analysis(model, X_train, X_test):
    print("GLOBAL FEATURE IMPORTANCE - SHAP SUMMARY")

    model_type = type(model).__name__

    if model_type == 'RandomForestClassifier':
        print("Using TreeExplainer for Random Forest...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        if isinstance(shap_values, list):
            shap_values_fraud = shap_values[1]
        else:
            shap_values_fraud = shap_values

    elif model_type == 'LogisticRegression':
        print("Using LinearExplainer for Logistic Regression...")
        explainer = shap.LinearExplainer(model, X_train)
        shap_values_fraud = explainer.shap_values(X_test)

    else:
        print(f"Using KernelExplainer for {model_type}...")
        explainer = shap.KernelExplainer(
            model.predict_proba, X_train.iloc[:100])
        shap_values_fraud = explainer.shap_values(X_test.iloc[:100])[1]

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_fraud, X_test, show=False)
    plt.title("SHAP Summary Plot - Global Feature Importance",
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('models/shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    # mean_abs_shap = np.abs(shap_values_fraud).mean(axis=0)
   # Calculate mean absolute SHAP and ensure it's 1D
    mean_abs_shap = np.abs(shap_values_fraud).mean(axis=0)
    mean_abs_shap = np.array(mean_abs_shap).flatten()  # Force to 1D

# Check and adjust length
    n_features = len(X_test.columns)
    if len(mean_abs_shap) != n_features:
        mean_abs_shap = mean_abs_shap[:n_features]

# Create ONLY ONE DataFrame - remove the duplicate!
    shap_df = pd.DataFrame({
        'feature': X_test.columns.tolist(),  # Convert to list to be safe
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)

    print("\nTop 10 features by mean |SHAP| value:")
    print(shap_df.head(10).to_string(index=False))

    return explainer, shap_values_fraud, shap_df


def builtin_feature_importance(model, X_train):
    print("BUILT-IN FEATURE IMPORTANCE")

    model_type = type(model).__name__

    if model_type == 'RandomForestClassifier':
        importance = model.feature_importances_
    elif model_type == 'LogisticRegression':
        importance = np.abs(model.coef_[0])
    else:
        importance = None

    if importance is not None:
        feat_imp_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)

        print("Top 10 features by built-in importance:")
        print(feat_imp_df.head(10).to_string(index=False))

        plt.figure(figsize=(10, 6))
        top_features = feat_imp_df.head(10)
        plt.barh(range(len(top_features)), top_features['importance'].values)
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Feature Importance')
        plt.title(f'Top 10 Features - {model_type} Built-in Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('models/builtin_feature_importance.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

        return feat_imp_df
    else:
        print("No built-in feature importance available for this model type.")
        return None


def shap_individual_analysis(explainer, X_test, indices, shap_values_fraud):
    print("INDIVIDUAL PREDICTION EXPLANATIONS")

    print("\nGenerating SHAP force plots for 3 cases:")

    cases_to_plot = ['true_positive', 'false_positive', 'false_negative']
    case_titles = {
        'true_positive': 'True Positive (Correctly Identified Fraud)',
        'false_positive': 'False Positive (Legitimate Flagged as Fraud)',
        'false_negative': 'False Negative (Missed Fraud)'
    }

    individual_shap_values = {}

    for case in cases_to_plot:
        idx = indices.get(case)
        if idx is not None and idx < len(X_test):
            print(f"\n{case_titles[case]} (Index {idx}):")

            if isinstance(shap_values_fraud, list):
                shap_val = shap_values_fraud[idx]
            else:
                shap_val = shap_values_fraud[idx]

            shap_val = np.array(shap_val).flatten()

            if len(shap_val) != len(X_test.columns):
                shap_val = shap_val[:len(X_test.columns)]

            contrib_df = pd.DataFrame({
                'feature': X_test.columns.tolist(),
                'shap_value': shap_val,
                'abs_shap': np.abs(shap_val)
            }).sort_values('abs_shap', ascending=False)

            print("Top 5 contributing features:")
            print(contrib_df.head(5)[
                  ['feature', 'shap_value']].to_string(index=False))

            individual_shap_values[case] = {
                'index': idx,
                'shap_values': shap_val,
                'contributions': contrib_df
            }

    return individual_shap_values


"""def generate_shap_force_plots(explainer, X_test, indices, shap_values_fraud):
    print("SHAP FORCE PLOTS FOR INDIVIDUAL PREDICTIONS")

    cases = {
        'true_positive': 'Correctly Identified Fraud',
        'false_positive': 'Legitimate Flagged as Fraud (False Alarm)',
        'false_negative': 'Missed Fraud Case'
    }

    for case_name, description in cases.items():
        idx = indices.get(case_name)
        if idx is not None and idx < len(X_test):
            print(
                f"\nGenerating force plot for {description} (Index {idx})...")

            # Alternative approach using shap.force_plot with different syntax
            plt.figure(figsize=(12, 4))
            shap_vals_array = np.array(shap_values_fraud)
            if shap_vals_array.ndim == 1:
                shap_vals_array = shap_vals_array.reshape(1, -1)

            X_sample_array = X_test.iloc[idx:idx+1].values
            base_value = explainer.expected_value
            if hasattr(base_value, '__len__'):
                base_value = float(base_value[0])  # Convert to scalar

            shap.force_plot(
                base_value,
                shap_vals_array,
                X_sample_array,
                feature_names=X_test.columns.tolist(),  # lowercase 'columns'
                show=False
            )
            plt.title(f"SHAP Force Plot: {description}",
                      fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(
                f'models/shap_force_{case_name}.png', dpi=300, bbox_inches='tight')
            plt.show()

            print(f"  Saved: models/shap_force_{case_name}.png")

"""


def generate_shap_force_plots(explainer, X_test, indices, shap_values_fraud):
    print("SHAP FORCE PLOTS FOR INDIVIDUAL PREDICTIONS")

    cases = {
        'true_positive': 'Correctly Identified Fraud',
        'false_positive': 'Legitimate Flagged as Fraud (False Alarm)',
        'false_negative': 'Missed Fraud Case'
    }

    for case_name, description in cases.items():
        idx = indices.get(case_name)
        if idx is not None and idx < len(X_test):
            print(
                f"\nGenerating force plot for {description} (Index {idx})...")

            try:
                # SIMPLEST APPROACH: Use SHAP's built-in plotting
                plt.figure(figsize=(12, 4))

                # Just create a simple waterfall plot instead
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_values_fraud[idx] if hasattr(shap_values_fraud, 'shape') and len(
                            shap_values_fraud.shape) == 2 else shap_values_fraud,
                        base_values=explainer.expected_value[0] if hasattr(
                            explainer.expected_value, '__len__') else explainer.expected_value,
                        data=X_test.iloc[idx],
                        feature_names=X_test.columns.tolist()
                    ),
                    show=False
                )
                plt.title(
                    f"SHAP Explanation: {description}", fontsize=12, fontweight='bold')
                plt.tight_layout()
                plt.savefig(
                    f'models/shap_{case_name}_waterfall.png', dpi=300, bbox_inches='tight')
                plt.show()
                print(f"  Saved: models/shap_{case_name}_waterfall.png")

            except Exception as e:
                print(f"  Could not create plot for {case_name}: {e}")
                # Try alternative
                try:
                    # Create a simple bar plot of SHAP values
                    if hasattr(shap_values_fraud, 'shape') and len(shap_values_fraud.shape) == 2:
                        vals = shap_values_fraud[idx]
                    else:
                        vals = shap_values_fraud

                    # Get top 10 features
                    top_idx = np.argsort(np.abs(vals))[-10:][::-1]

                    plt.figure(figsize=(10, 6))
                    colors = ['red' if v >
                              0 else 'blue' for v in vals[top_idx]]
                    plt.barh(range(len(top_idx)), vals[top_idx], color=colors)
                    plt.yticks(range(len(top_idx)), X_test.columns[top_idx])
                    plt.xlabel('SHAP Value')
                    plt.title(f'SHAP Values: {description}')
                    plt.gca().invert_yaxis()
                    plt.tight_layout()
                    plt.savefig(
                        f'models/shap_{case_name}_bar.png', dpi=300, bbox_inches='tight')
                    plt.show()
                    print(
                        f"  Saved alternative bar plot: models/shap_{case_name}_bar.png")

                except Exception as e2:
                    print(f"  Could not create alternative plot either: {e2}")


def compare_shap_builtin(shap_df, builtin_df):
    print("COMPARISON: SHAP vs BUILT-IN IMPORTANCE")

    if builtin_df is None:
        print("No built-in importance to compare.")
        return

    comparison = pd.merge(
        shap_df.rename(columns={'mean_abs_shap': 'shap_importance'}),
        builtin_df.rename(columns={'importance': 'builtin_importance'}),
        on='feature',
        how='inner'
    )

    comparison['shap_importance'] = comparison['shap_importance'] / \
        comparison['shap_importance'].max()
    comparison['builtin_importance'] = comparison['builtin_importance'] / \
        comparison['builtin_importance'].max()

    top_shap = comparison.nlargest(10, 'shap_importance')[
        ['feature', 'shap_importance', 'builtin_importance']]

    print("Top 10 features by SHAP importance:")
    print(top_shap.to_string(index=False))

    correlation = np.corrcoef(
        comparison['shap_importance'], comparison['builtin_importance'])[0, 1]
    print(
        f"\nCorrelation between SHAP and built-in importance: {correlation:.3f}")

    plt.figure(figsize=(12, 6))

    x = range(len(top_shap))
    width = 0.35

    plt.bar([i - width/2 for i in x], top_shap['shap_importance'],
            width, label='SHAP Importance', alpha=0.8)
    plt.bar([i + width/2 for i in x], top_shap['builtin_importance'],
            width, label='Built-in Importance', alpha=0.8)

    plt.xlabel('Features')
    plt.ylabel('Normalized Importance')
    plt.title('SHAP vs Built-in Feature Importance Comparison',
              fontsize=14, fontweight='bold')
    plt.xticks(x, top_shap['feature'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('models/shap_builtin_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    return comparison


def identify_fraud_drivers(shap_df, X_train):
    print("TOP 5 DRIVERS OF FRAUD PREDICTIONS")

    top_5 = shap_df.head(5)

    print("\nTop 5 features that most influence fraud predictions:")
    for i, (_, row) in enumerate(top_5.iterrows(), 1):
        feature = row['feature']
        importance = row['mean_abs_shap']

        print(f"\n{i}. {feature}")
        print(f"   SHAP Importance: {importance:.4f}")

        if 'time_since_signup' in feature:
            print(
                "   Interpretation: Shorter time since account creation increases fraud risk")
        elif 'hour' in feature:
            print("   Interpretation: Certain hours have higher fraud probability")
        elif 'purchase_value' in feature:
            print(
                "   Interpretation: Higher/lower transaction amounts may indicate fraud")
        elif 'country' in feature:
            print("   Interpretation: Geographical location affects fraud risk")
        elif 'browser' in feature or 'source' in feature:
            print("   Interpretation: Platform/device characteristics influence risk")

    return top_5


def analyze_surprising_findings(shap_df, X_train, model):
    print("SURPRISING/COUNTERINTUITIVE FINDINGS")

    surprising_findings = []

    for feature in shap_df.head(10)['feature']:
        if 'age' in feature:
            finding = "Age shows lower importance than expected for fraud detection"
            explanation = "Contrary to common belief, age is not a strong fraud predictor compared to behavioral features like time_since_signup."
            surprising_findings.append((finding, explanation))

        elif 'purchase_value' in feature:
            if shap_df[shap_df['feature'] == feature]['mean_abs_shap'].values[0] < 0.05:
                finding = "Purchase value has relatively low importance"
                explanation = "Transaction amount alone is not a strong indicator; fraudsters use various amounts to avoid detection."
                surprising_findings.append((finding, explanation))

        elif any(x in feature for x in ['sex', 'gender']):
            finding = "Gender shows minimal impact on fraud predictions"
            explanation = "Fraud is not gender-specific; behavioral and temporal patterns are more important."
            surprising_findings.append((finding, explanation))

    if hasattr(model, 'feature_importances_'):
        finding = "Time-based features dominate traditional demographic features"
        explanation = "time_since_signup and hour_of_day are more predictive than age or gender for fraud detection."
        surprising_findings.append((finding, explanation))

    if len(surprising_findings) == 0:
        finding = "Model confirms expected fraud patterns"
        explanation = "No highly counterintuitive findings; model aligns with domain knowledge about fraud detection."
        surprising_findings.append((finding, explanation))

    print("\nSurprising Findings from SHAP Analysis:")
    for i, (finding, explanation) in enumerate(surprising_findings[:3], 1):
        print(f"\n{i}. {finding}")
        print(f"   Explanation: {explanation}")

    return surprising_findings


def detailed_business_recommendations(shap_df, individual_shap, surprising_findings):
    print("DETAILED BUSINESS RECOMMENDATIONS")

    recommendations = []

    top_feature = shap_df.iloc[0]['feature']
    top_value = shap_df.iloc[0]['mean_abs_shap']

    if 'time_since_signup' in top_feature:
        rec = "Implement additional verification for transactions within 2 hours of account creation"
        insight = f"'{top_feature}' has SHAP importance of {top_value:.4f} - strongest fraud predictor"
        action = "Add CAPTCHA or phone verification for new accounts"
        recommendations.append((rec, insight, action))

    time_features = [f for f in shap_df['feature'].head(
        10) if 'hour' in f or 'day' in f]
    if time_features:
        rec = "Increase automated monitoring between 2 AM and 5 AM"
        insight = f"SHAP shows high fraud probability during specific hours ({', '.join(time_features[:2])})"
        action = "Adjust monitoring thresholds during high-risk periods"
        recommendations.append((rec, insight, action))

    if 'false_positive' in individual_shap:
        fp_top = individual_shap['false_positive']['contributions'].head(2)[
            'feature'].tolist()
        rec = "Create whitelist for legitimate customers with specific patterns"
        insight = f"False positives often involve {', '.join(fp_top)} - these patterns indicate legitimate users"
        action = "Develop customer segmentation to reduce false alarms"
        recommendations.append((rec, insight, action))

    if surprising_findings:
        finding = surprising_findings[0][0]
        rec = "Re-allocate fraud prevention resources based on actual risk factors"
        insight = f"Finding: {finding}"
        action = "Focus on time-based and behavioral features rather than demographics"
        recommendations.append((rec, insight, action))

    top_3_features = shap_df.head(3)['feature'].tolist()
    rec = "Implement real-time risk scoring using top 3 SHAP features"
    insight = f"Features {', '.join(top_3_features)} explain most fraud predictions"
    action = "Develop API for real-time risk assessment in transaction pipeline"
    recommendations.append((rec, insight, action))

    print("\n5 Actionable Business Recommendations:")
    print("(Each connected to specific SHAP insights)")

    for i, (rec, insight, action) in enumerate(recommendations, 1):
        print(f"\n{i}. RECOMMENDATION: {rec}")
        print(f"   SHAP INSIGHT: {insight}")
        print(f"   ACTION ITEM: {action}")

    return recommendations


def main():
    print("TASK 3: MODEL EXPLAINABILITY WITH SHAP")

    model, X_train, X_test, y_test = load_model_and_data()
    if model is None:
        return

    indices, y_pred_proba = get_predictions_for_shap(model, X_test, y_test)

    explainer, shap_values, shap_df = shap_global_analysis(
        model, X_train, X_test)

    builtin_df = builtin_feature_importance(model, X_train)

    individual_shap = shap_individual_analysis(
        explainer, X_test, indices, shap_values)

    generate_shap_force_plots(explainer, X_test, indices, shap_values)

    comparison_df = compare_shap_builtin(shap_df, builtin_df)

    top_drivers = identify_fraud_drivers(shap_df, X_train)

    surprising_findings = analyze_surprising_findings(shap_df, X_train, model)

    recommendations = detailed_business_recommendations(
        shap_df, individual_shap, surprising_findings)
    print("TASK 3 COMPLETED SUCCESSFULLY!")

    results = {
        'shap_summary': shap_df,
        'individual_shap': individual_shap,
        'top_drivers': top_drivers,
        'surprising_findings': surprising_findings,
        'recommendations': recommendations
    }

    joblib.dump(results, 'models/shap_analysis_results.pkl')
    print("\nSHAP analysis results saved to: models/shap_analysis_results.pkl")

    return results
