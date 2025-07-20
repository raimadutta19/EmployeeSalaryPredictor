import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load model and encoders
import os
model_path = os.path.join(os.path.dirname(__file__), 'model', 'salary_model.pkl')
model = joblib.load(model_path)

import os
label_encoder_path = os.path.join(os.path.dirname(__file__), 'model', 'label_encoders.pkl')
label_encoders = joblib.load(label_encoder_path)

report = joblib.load('model/evaluation_report.pkl')
conf_matrix = joblib.load('model/confusion_matrix.pkl')

st.title("💼 Employee Salary Predictor")
st.markdown("Predict whether an individual's income is **>50K** or **<=50K** using ML.")

# --- SIDEBAR INPUT ---
st.sidebar.header("📥 Enter Employee Details")

age = st.sidebar.slider('Age', 18, 80, 30)
workclass = st.sidebar.selectbox('Workclass', label_encoders['workclass'].classes_)
education = st.sidebar.selectbox('Education', label_encoders['education'].classes_)
marital_status = st.sidebar.selectbox('Marital Status', label_encoders['marital-status'].classes_)
occupation = st.sidebar.selectbox('Occupation', label_encoders['occupation'].classes_)
relationship = st.sidebar.selectbox('Relationship', label_encoders['relationship'].classes_)
race = st.sidebar.selectbox('Race', label_encoders['race'].classes_)
gender = st.sidebar.selectbox('Gender', label_encoders['gender'].classes_)
capital_gain = st.sidebar.number_input('Capital Gain', 0, 99999, 0)
capital_loss = st.sidebar.number_input('Capital Loss', 0, 99999, 0)
hours_per_week = st.sidebar.slider('Hours Per Week', 1, 100, 40)
native_country = st.sidebar.selectbox('Native Country', label_encoders['native-country'].classes_)

# --- MODEL INPUT PREPARATION ---
input_data = pd.DataFrame({
    'age': [age],
    'workclass': [label_encoders['workclass'].transform([workclass])[0]],
    'fnlwgt': [0],  # dummy
    'education': [label_encoders['education'].transform([education])[0]],
    'educational-num': [0],  # dummy
    'marital-status': [label_encoders['marital-status'].transform([marital_status])[0]],
    'occupation': [label_encoders['occupation'].transform([occupation])[0]],
    'relationship': [label_encoders['relationship'].transform([relationship])[0]],
    'race': [label_encoders['race'].transform([race])[0]],
    'gender': [label_encoders['gender'].transform([gender])[0]],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [label_encoders['native-country'].transform([native_country])[0]]
})

# --- PREDICTION ---
st.subheader("🔍 Prediction Result")

if st.button("Predict Income"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    income = label_encoders['income'].inverse_transform([prediction])[0]
    confidence = round(max(proba) * 100, 2)

    col1, col2 = st.columns(2)
    col1.success(f"💰 Predicted Income: **{income}**")
    col2.info(f"🧠 Confidence: **{confidence}%**")

# --- METRICS SECTION ---
with st.expander("📊 View Model Evaluation Metrics"):
    st.write(f"**Accuracy**: {round(report['accuracy'] * 100, 2)}%")
    st.write(f"**Precision (>50K)**: {round(report['1']['precision'] * 100, 2)}%")
    st.write(f"**Recall (>50K)**: {round(report['1']['recall'] * 100, 2)}%")
    st.write(f"**F1-Score (>50K)**: {round(report['1']['f1-score'] * 100, 2)}%")

    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

# --- FOOTER ---
st.markdown("---")
st.markdown("Made with ❤️ by Raima Dutta")
