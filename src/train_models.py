import os
import warnings
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Mute warnings for a clean production console
warnings.filterwarnings("ignore")

# Helper function to format feature names cleanly for downstream charts
def format_label(col):
    mapping = {
        'person_age': 'Age', 'person_income': 'Annual Income',
        'person_emp_length': 'Employment Length', 'loan_amnt': 'Loan Amount',
        'loan_int_rate': 'Interest Rate', 'loan_percent_income': 'Loan-to-Income Ratio',
        'cb_person_cred_hist_length': 'Credit History'
    }
    return mapping.get(col, col.replace('_', ' ').title())

def main():
    print("Starting Production Model Training Pipeline...")

    # 1. Load Data
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("ERROR: Data files not found in /data. Please run 'python src/prepare_data.py' first.")
        return

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop('loan_status', axis=1)
    y_train = train_df['loan_status']
    X_test = test_df.drop('loan_status', axis=1)
    y_test = test_df['loan_status']
    print(f"Data loaded successfully. Training on {len(X_train)} records.")

    # 2. Build Preprocessing Pipeline
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object', 'string']).columns

    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), 
                                ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), 
                                ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

    # 3. Define the Models (HARDCODED WITH OPTIMIZED HYPERPARAMETERS)
    print("Assembling Model 1 (Voting Classifier) and Model 2 (Random Forest)...")
    
    voting_clf = VotingClassifier(
        estimators=[
            ('gb', GradientBoostingClassifier(
                learning_rate=0.1, n_estimators=200, max_depth=5, random_state=42)),
            ('xgb', XGBClassifier(
                learning_rate=0.1, n_estimators=200, max_depth=6, random_state=42, eval_metric='logloss')),
            ('lgbm', LGBMClassifier(
                learning_rate=0.1, n_estimators=200, num_leaves=50, random_state=42, verbose=-1))
        ], voting='soft'
    )
    
    model_1 = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', voting_clf)])
    
    model_2 = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(
        n_estimators=200, max_depth=15, random_state=42))])

    # 4. Train the Models
    print("Training Model 1 (This may take a moment due to optimized depth)...")
    model_1.fit(X_train, y_train)
    
    print("Training Model 2 (Backup Model)...")
    model_2.fit(X_train, y_train)

    # 5. Evaluate & Generate Predictions CSV
    print("Evaluating and generating predictions...")
    y_pred_proba = model_1.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"   Validation AUC Score: {auc_score:.4f}")
    
    predictions_df = pd.DataFrame({'actual': y_test, 'probability': y_pred_proba})

    # 6. Extract Feature Importances CSV
    print("Extracting clean feature importances...")
    cat_encoder = model_1.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    cat_features_encoded = cat_encoder.get_feature_names_out(categorical_features)
    all_features = list(numeric_features) + list(cat_features_encoded)
    
    # Extract importances from the Gradient Boosting component of the ensemble
    gb_fitted = model_1.named_steps['classifier'].estimators_[0] 
    
    feature_importance_df = pd.DataFrame({
        'Feature': [format_label(f) for f in all_features],
        'Importance': gb_fitted.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    # 7. Export All Artifacts
    print("Exporting all artifacts to /artifacts directory...")
    os.makedirs('artifacts', exist_ok=True)
    
    joblib.dump(model_1, 'artifacts/model_1.pkl')
    joblib.dump(model_2, 'artifacts/model_2.pkl')
    predictions_df.to_csv('artifacts/predictions.csv', index=False)
    feature_importance_df.to_csv('artifacts/feature_importance.csv', index=False)

    print("DEPLOYMENT COMPLETE. All files are ready for integration.")

if __name__ == "__main__":
    main()
    