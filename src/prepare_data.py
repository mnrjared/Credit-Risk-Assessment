import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 1. Path relative to the root folder (where you run the command)
# Ensure you run this script as: python src/prepare_data.py
data_path = 'data/credit_risk_dataset.csv'

if not os.path.exists(data_path):
    print(f"❌ ERROR: Could not find {data_path}. Ensure you are in the root directory.")
else:
    df = pd.read_csv(data_path)
    print("✅ Raw data loaded successfully.")

    # 2. Check for missing values
    print("\nMissing values per column:\n", df.isnull().sum())

    # 3. Outlier Handling (Clipping) - EXCLUDING THE TARGET
    # We only want to clip actual feature values, not our 0/1 target labels
    num_features = [col for col in df.select_dtypes(include=['int64', 'float64']).columns 
                    if col != 'loan_status']
    
    for column in num_features:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[column] = df[column].clip(lower, upper)
    
    print(f"✅ Outlier clipping complete for {len(num_features)} feature columns.")

    # 4. Stratified Split (Mandatory for imbalanced data)
    # This preserves the ~22% default rate in both the training and testing files
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        stratify=df["loan_status"], 
        random_state=42
    )

    # 5. Save files into the local data folder (Root-relative paths)
    os.makedirs('data', exist_ok=True)
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    print("\n✅ train.csv and test.csv generated successfully in the /data folder!")
    
    # 6. Final Sanity Check for P2 Modeling
    # This should now show both 0s and 1s
    print("\n--- Distribution Check in Training Data ---")
    print(train_df["loan_status"].value_counts())