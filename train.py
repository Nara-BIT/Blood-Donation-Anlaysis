#Import all necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib # Import joblib
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import gradio as gr # Not needed for training

from imblearn.over_sampling import SMOTE
import xgboost as xgb

from skopt import BayesSearchCV
from skopt.space import Real, Integer

print("--- Starting Model Training ---")
start_time = time.time()

#Load Dataset
path = "C:\\Users\\naras\\Desktop\\My_Files\\ML_Workshop\\Blood_Donation_Analysis\\blood-train.csv"
try:
    df = pd.read_csv(path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: '{path}' not found. Please make sure it's in the same folder.")
    exit()

# Drop unnecessary column (ID)
df.drop(df.columns[0], axis=1, inplace=True)

# Outlier Identification
cols=['Months since Last Donation', 'Number of Donations', 'Total Volume Donated (c.c.)', 'Months since First Donation']
for col in cols:
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot for outlier:{col}")
    # plt.show() # Commented out to allow script to run fully

#Outlier Treatment using IQR
print("Applying outlier treatment...")
cols_with_outliers = ['Months since Last Donation', 'Number of Donations', 'Total Volume Donated (c.c.)']
for col in cols_with_outliers:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower_bound, lower_bound,
                     np.where(df[col] > upper_bound, upper_bound, df[col]))
print("Outlier treatment complete.")

#Show no outliers after IQR
for col in cols_with_outliers:
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot for Outlier after IQR:{col}")
    # plt.show() # Commented out

#Feature Engineering
print("Performing feature engineering...")
df['donation_rate'] = df['Number of Donations'] / (df['Months since First Donation'] + 1)
df['recency_inverse'] = 1 / (df['Months since Last Donation'] + 1)
print("Feature engineering complete.")

#Define Features and Target
X = df.drop('Made Donation in March 2007', axis=1)
y = df['Made Donation in March 2007']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Handle Class Imbalance using SMOTE
print("Handling class imbalance with SMOTE...")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)
print("SMOTE complete.")

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

#Bayesian Hyperparameter Tuning for XGBoost
param_space = {
    'n_estimators': Integer(100, 500),
    'max_depth': Integer(3, 10),
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'subsample': Real(0.6, 1.0),
    'colsample_bytree': Real(0.6, 1.0),
    'min_child_weight': Integer(1, 10)
}

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=1,  # SMOTE already balances, keep 1
    eval_metric='logloss',
    random_state=42
)

print("Starting Bayesian Hyperparameter Tuning (this may take a few minutes)...")
bayes_search = BayesSearchCV(
    estimator=xgb_model,
    search_spaces=param_space,
    n_iter=30,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0,
    random_state=42
)

bayes_search.fit(x_train, y_train)
best_model = bayes_search.best_estimator_

print("\n✅ Best Parameters Found:")
print(bayes_search.best_params_)

#Predictions and Evaluation
y_pred = best_model.predict(x_test)
print("\n📊 XGBoost Model Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- IMPORTANT PART: SAVE THE MODEL AND SCALER ---
print("Saving model and scaler...")
joblib.dump(best_model, 'blood_donation_xgb_model.joblib')
joblib.dump(scaler, 'blood_donation_scaler.joblib')

end_time = time.time()
print(f"\n--- Model Training Complete in {end_time - start_time:.2f} seconds ---")
print("✅ Model saved as 'blood_donation_xgb_model.joblib'")
print("✅ Scaler saved as 'blood_donation_scaler.joblib'")
