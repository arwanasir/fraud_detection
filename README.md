# fraud_detection

## Project Status

Task 1: Completed | Task 2: Completed | Task 3: Pending

## Task 1: Data Analysis & Preprocessing (Completed)

- Cleaned and analyzed fraud data
- Mapped IP addresses to countries
- Created features: time_since_signup, hour_of_day, day_of_week
- Handled class imbalance with undersampling
- Prepared data for modeling

## Task 2: Model Building & Training (Completed)

- Built Logistic Regression baseline model
- Built Random Forest ensemble model
- Evaluated with AUC-PR, F1-score, confusion matrix
- Performed 5-fold cross-validation
- Selected best model based on performance

## Repository Structure

data/ - Raw and processed data
notebooks/ - Analysis notebooks
src/ - Python modules
models/ - Saved machine learning models
requirements.txt - Dependencies

## How to Run

1. Install: pip install -r requirements.txt
2. Run notebooks in order:
   - eda-fraud-data.ipynb
   - feature-engineering.ipynb
   - data-preprocessing.ipynb
   - modeling.ipynb

## Next Steps

Task 3: Model explainability with SHAP analysis
