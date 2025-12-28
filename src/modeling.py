import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, precision_recall_curve,
                             confusion_matrix, classification_report)
import joblib
import os


def load_preprocessed_data():
    """Load the data we processed in Task 1"""
    print("Loading preprocessed data from Task 1...")

    try:
        X_train = pd.read_csv('../data/processed/X_train_bal_small.csv')
        y_train = pd.read_csv(
            '../data/processed/y_train_bal_small.csv').iloc[:, 0]
        X_test = pd.read_csv('../data/processed/X_test_small.csv')
        y_test = pd.read_csv('../data/processed/y_test_small.csv').iloc[:, 0]

    except FileNotFoundError:
        print("Balanced data not found, loading simple version...")
        X_train = pd.read_csv('../data/processed/X_train_simple.csv')
        y_train = pd.read_csv(
            '../data/processed/y_train_simple.csv').iloc[:, 0]
        X_test = pd.read_csv('../data/processed/X_test_simple.csv')
        y_test = pd.read_csv('../data/processed/y_test_simple.csv').iloc[:, 0]

    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    print(
        f"Training class distribution: {pd.Series(y_train).value_counts().to_dict()}")
    print(
        f"Test class distribution: {pd.Series(y_test).value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test, model_name="Model"):

    print(f"EVALUATING {model_name.upper()}")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    auc_pr = average_precision_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    report = classification_report(
        y_test, y_pred, target_names=['Legitimate', 'Fraud'])

    print(f"\nPerformance Metrics:")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"AUC-PR:  {auc_pr:.4f}  ← MOST IMPORTANT FOR IMBALANCED DATA")
    print(f"F1-Score: {f1:.4f}")

    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"               0       1")
    print(
        f"Actual 0   [{cm[0, 0]:>6}  {cm[0, 1]:>6}]  → False Positives: {cm[0, 1]}")
    print(
        f"        1   [{cm[1, 0]:>6}  {cm[1, 1]:>6}]  → False Negatives: {cm[1, 0]}")

    print(f"\nClassification Report:")
    print(report)

    false_positives = cm[0, 1]
    false_negatives = cm[1, 0]

    print(f"\nBusiness Impact Analysis:")
    print(f"False Positives (legitimate flagged as fraud): {false_positives}")
    print(f"  → Could alienate {false_positives} customers")
    print(f"False Negatives (fraud missed): {false_negatives}")
    print(f"  → Potential financial loss from {false_negatives} fraud cases")

    return {
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'f1_score': f1,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }


def plot_pr_curve(y_test, y_pred_proba_lr, y_pred_proba_rf, model_names=['Logistic Regression', 'Random Forest']):
    plt.figure(figsize=(10, 6))

    for i, (y_pred_proba, name) in enumerate(zip([y_pred_proba_lr, y_pred_proba_rf], model_names)):
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        auc_pr = average_precision_score(y_test, y_pred_proba)
        plt.plot(recall, precision,
                 label=f'{name} (AUC-PR = {auc_pr:.3f})', linewidth=2)

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves - Fraud Detection',
              fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('models/pr_curves_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def build_baseline_model(X_train, y_train):
    print("BUILDING BASELINE MODEL: LOGISTIC REGRESSION")

    print("\nWhy Logistic Regression as baseline:")
    print("1. Interpretable - coefficients show feature importance")
    print("2. Fast to train - good for baseline comparison")
    print("3. Provides probability scores - useful for threshold adjustment")
    print("4. Less prone to overfitting with imbalanced data")

    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )

    lr_model.fit(X_train, y_train)

    print("Model trained successfully!")
    print(f"Number of features: {X_train.shape[1]}")

    if hasattr(lr_model, 'coef_'):
        print("\nTop 10 most important features (coefficient magnitude):")
        coef_df = pd.DataFrame({
            'feature': X_train.columns,
            'coefficient': lr_model.coef_[0]
        })
        coef_df['abs_coef'] = np.abs(coef_df['coefficient'])
        print(coef_df.nlargest(10, 'abs_coef')[['feature', 'coefficient']])

    return lr_model


def build_ensemble_model(X_train, y_train):
    print("BUILDING ENSEMBLE MODEL: RANDOM FOREST")

    print("\nWhy Random Forest:")
    print("1. Handles non-linear relationships")
    print("2. Robust to outliers and irrelevant features")
    print("3. Built-in feature importance")
    print("4. Good performance on imbalanced data with class_weight")
    print("5. Less prone to overfitting than single decision trees")

    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(X_train, y_train)

    print("Model trained successfully!")
    print(f"Number of trees: {rf_model.n_estimators}")
    print(f"Number of features: {X_train.shape[1]}")

    print("\nTop 10 most important features:")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(feature_importance.head(10))

    plt.figure(figsize=(10, 6))
    top_features = feature_importance.head(10)
    plt.barh(range(len(top_features)), top_features['importance'].values)
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Feature Importance - Random Forest')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('models/rf_feature_importance.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    return rf_model


def perform_cross_validation(model, X_train, y_train, model_name="Model", cv=5):
    print(f"\n{'='*60}")
    print(f"STRATIFIED {cv}-FOLD CROSS VALIDATION - {model_name.upper()}")
    print('='*60)

    scoring = {
        'auc_pr': 'average_precision',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }

    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    print(f"Performing {cv}-fold stratified cross-validation...")
    print(f"Training samples per fold: ~{len(X_train)//cv}")

    cv_results = {}
    for metric_name, metric_scorer in scoring.items():
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv_strategy,
            scoring=metric_scorer,
            n_jobs=-1
        )
        cv_results[metric_name] = scores

    for metric_name, scores in cv_results.items():
        print(f"\n{metric_name.upper()}:")
        print(f"  Fold scores: {['%.4f' % s for s in scores]}")
        print(f"  Mean: {scores.mean():.4f}")
        print(f"  Std:  {scores.std():.4f}")
        print(f"  95% CI: [{scores.mean() - 1.96*scores.std():.4f}, "
              f"{scores.mean() + 1.96*scores.std():.4f}]")

    return cv_results


def compare_models(results_lr, results_rf):
    print("MODEL COMPARISON AND SELECTION")

    comparison = pd.DataFrame({
        'Metric': ['AUC-PR', 'AUC-ROC', 'F1-Score', 'False Positives', 'False Negatives'],
        'Logistic Regression': [
            results_lr['auc_pr'],
            results_lr['auc_roc'],
            results_lr['f1_score'],
            results_lr['confusion_matrix'][0, 1],
            results_lr['confusion_matrix'][1, 0]
        ],
        'Random Forest': [
            results_rf['auc_pr'],
            results_rf['auc_roc'],
            results_rf['f1_score'],
            results_rf['confusion_matrix'][0, 1],
            results_rf['confusion_matrix'][1, 0]
        ]
    })

    def get_winner(row):
        if row['Metric'] in ['False Positives', 'False Negatives']:

            return 'Logistic Regression' if row['Logistic Regression'] < row['Random Forest'] else 'Random Forest'
        else:
            return 'Logistic Regression' if row['Logistic Regression'] > row['Random Forest'] else 'Random Forest'

    comparison['Winner'] = comparison.apply(get_winner, axis=1)

    print("\nPerformance Comparison:")
    print(comparison.to_string(index=False))

    print("FINAL MODEL SELECTION")

    print("\nDecision Criteria:")
    print("1. AUC-PR is most important for imbalanced fraud detection")
    print("2. False negatives (missed fraud) cause financial loss")
    print("3. False positives (annoyed customers) affect user experience")
    print("4. Model interpretability for business stakeholders")

    auc_pr_diff = results_rf['auc_pr'] - results_lr['auc_pr']
    fn_diff = results_rf['confusion_matrix'][1, 0] - \
        results_lr['confusion_matrix'][1, 0]
    fp_diff = results_rf['confusion_matrix'][0, 1] - \
        results_lr['confusion_matrix'][0, 1]

    print(f"\nKey Differences:")
    print(f"AUC-PR difference (RF - LR): {auc_pr_diff:.4f}")
    print(f"Additional false negatives with RF: {fn_diff}")
    print(f"Additional false positives with RF: {fp_diff}")

    if auc_pr_diff > 0.05:
        selected = "Random Forest"
        reason = "Significantly better AUC-PR (+{:.3f}) despite slightly higher false positives".format(
            auc_pr_diff)
    elif fn_diff < -5:
        selected = "Random Forest"
        reason = "Catches {} more fraud cases (fewer false negatives)".format(
            -fn_diff)
    else:
        selected = "Logistic Regression"
        reason = "Good balance of performance and interpretability, with reasonable AUC-PR of {:.3f}".format(
            results_lr['auc_pr'])

    print(f"\nSelected Model: {selected}")
    print(f"Reason: {reason}")

    return selected, comparison


def save_model(model, model_name):
    """Save trained model to disk"""
    os.makedirs('models', exist_ok=True)
    filename = f'models/{model_name}.pkl'
    joblib.dump(model, filename)
    print(f"\nModel saved as: {filename}")
    return filename


def main():
    print("TASK 2: MODEL BUILDING AND TRAINING")
    X_train, X_test, y_train, y_test = load_preprocessed_data()

    lr_model = build_baseline_model(X_train, y_train)
    lr_results = evaluate_model(
        lr_model, X_test, y_test, "Logistic Regression")

    lr_cv_results = perform_cross_validation(
        LogisticRegression(max_iter=1000, random_state=42,
                           class_weight='balanced'),
        X_train, y_train, "Logistic Regression"
    )

    rf_model = build_ensemble_model(X_train, y_train)
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    rf_cv_results = perform_cross_validation(
        RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'),
        X_train, y_train, "Random Forest"
    )

    selected_model, comparison_df = compare_models(lr_results, rf_results)

    if selected_model == "Logistic Regression":
        save_model(lr_model, "logistic_regression_baseline")
    else:
        save_model(rf_model, "random_forest_ensemble")

    plot_pr_curve(
        y_test, lr_results['y_pred_proba'], rf_results['y_pred_proba'])

    print("TASK 2 COMPLETED SUCCESSFULLY!")

    return lr_model, rf_model, lr_results, rf_results, comparison_df, selected_model
