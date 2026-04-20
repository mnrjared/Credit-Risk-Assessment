import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer # Added to fix null values

# 1. Load from the root-relative data folder
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# 2. Separate features and target
X_train = train.drop('loan_status', axis=1)
y_train = train['loan_status']
X_test = test.drop('loan_status', axis=1)
y_test = test['loan_status']

# 3. Handle Missing Values (Crucial for scaling/modeling)
imputer = SimpleImputer(strategy='median')
num_columns = [
    'person_age', 'person_income', 'person_emp_length',
    'loan_amnt', 'loan_int_rate', 'loan_percent_income',
    'cb_person_cred_hist_length'
]

X_train[num_columns] = imputer.fit_transform(X_train[num_columns])
X_test[num_columns] = imputer.transform(X_test[num_columns])

# 4. Categorical Encoding
cat_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
X_train_encoded = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)

# 5. Scaling
scaler = StandardScaler()
X_train_encoded[num_columns] = scaler.fit_transform(X_train_encoded[num_columns])
X_test_encoded[num_columns] = scaler.transform(X_test_encoded[num_columns])

# 6. Save preprocessed data (Optional, for transparency)
X_train_encoded.to_csv("data/X_train_preprocessed.csv", index=False)
print("✅ Preprocessing complete. Nulls filled, categorical encoded, and features scaled.")
print(f"Final training features shape: {X_train_encoded.shape}")