# Credit Risk Assessment

A machine learning web application that assesses credit risk for loan applicants using an ensemble of gradient boosting models. Built with Dash and deployed on Render.

---

## Live Demo

[credit-risk-assessment-de3b.onrender.com](https://credit-risk-assessment-de3b.onrender.com)

---

## Project Structure

```
├── artifacts/
│   ├── model_1.pkl                  # Voting Classifier (GB + XGBoost + LightGBM)
│   ├── model_2.pkl                  # Random Forest (backup model)
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   ├── feature_importance.csv
│   └── predictions.csv
├── data/
│   ├── credit_risk_dataset.csv      # Raw dataset
│   ├── train.csv                    # 80% stratified split
│   └── test.csv                     # 20% stratified split
├── notebooks/
│   ├── EDA.ipynb
│   ├── modeling.ipynb
│   └── web_application.ipynb
├── src/
│   ├── prepare_data.py              # Cleaning, outlier clipping, train/test split
│   ├── preprocess_data.py           # Imputation, encoding, scaling
│   ├── train_models.py              # Model training and artifact export
│   ├── web_app.py                   # Dash web application
│   └── assets/
│       └── styles.css
├── requirements.txt
└── README.md
```

---

## Models

**Model 1 — Voting Classifier (primary)**
A soft-voting ensemble of three gradient boosting models:
- Gradient Boosting Classifier
- XGBoost
- LightGBM

**Model 2 — Random Forest (backup)**
Used alongside Model 1 to produce a consensus-based risk verdict.

Risk is determined by comparing both model outputs:
- Both predict default → **HIGH RISK**
- Both predict no default → **LOW RISK**
- Models disagree → **MODERATE RISK**

---

## Features Used

| Feature | Description |
|---|---|
| Age | Applicant's age |
| Annual Income | Yearly income in ZAR |
| Employment Length | Years in current employment |
| Loan Amount | Requested loan amount in ZAR |
| Interest Rate | Loan interest rate (%) |
| Loan-to-Income Ratio | Loan amount as a fraction of income |
| Credit History Length | Years of credit history |
| Loan Grade | Lender-assigned grade (A–G) |
| Home Ownership | RENT / OWN / MORTGAGE / OTHER |
| Loan Intent | Purpose of the loan |
| Default on File | Prior default history (Y/N) |

---

## Running Locally

**1. Clone the repo**
```bash
git clone https://github.com/yourname/credit-risk-assessment.git
cd credit-risk-assessment
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Prepare data and train models** (skip if using pre-built artifacts)
```bash
python src/prepare_data.py
python src/train_models.py
```

**4. Start the app**
```bash
python src/web_app.py
```

Then open `http://localhost:8050` in your browser.

---

## Deployment (Render)

- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn src.web_app:server`

---

## Dependencies

```
dash
pandas
plotly
gunicorn
scikit-learn
joblib
xgboost
lightgbm
```
