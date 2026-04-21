import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

X = df.drop('loan_status', axis=1)
y = df['loan_status']

X_encoded = pd.get_dummies(X, columns=[
    'person_home_ownership',
    'loan_intent',
    'loan_grade',
    'cb_person_default_on_file'
], drop_first=True)

scaler = StandardScaler()
num_columns = [
    'person_age', 'person_income', 'person_emp_length',
    'loan_amnt', 'loan_int_rate', 'loan_percent_income',
    'cb_person_cred_hist_length'
]

X_encoded[num_columns] = scaler.fit_transform(X_encoded[num_columns])