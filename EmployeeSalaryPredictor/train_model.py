import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

# Load dataset
df = pd.read_csv('data/cleaned_adult.csv')

# Drop rows with missing values or ?
df = df.replace('?', np.nan).dropna()

# Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split dataset
X = df.drop('income', axis=1)
y = df['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# Train XGBoost
xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)

# Select best model
best_model = rf_model if rf_acc > xgb_acc else xgb_model
print("Best Model Accuracy:", max(rf_acc, xgb_acc))

# Save the model
os.makedirs("model", exist_ok=True)
joblib.dump(best_model, 'model/salary_model.pkl')
joblib.dump(label_encoders, 'model/label_encoders.pkl')

from sklearn.metrics import classification_report, confusion_matrix

# Get predictions
y_pred = best_model.predict(X_test)

# Evaluation metrics
report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)

# Save evaluation report and confusion matrix
joblib.dump(report, 'model/evaluation_report.pkl')
joblib.dump(conf_matrix, 'model/confusion_matrix.pkl')
