# 💼 Employee Salary Predictor

A machine learning-powered web application that predicts employee salaries based on job features such as role, experience, company size, remote ratio, and location. Built using **Streamlit**, **scikit-learn**, **XGBoost**, **matplotlib**, and **seaborn**.

🔗 [Live App on Streamlit](https://employeesalarypredictor-8uxxkb2bc4ubexuwgrptcs.streamlit.app/)  
📊 Dataset: [Data Science Salaries 2023 – Kaggle](https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2023)

---

## 🚀 Features

- Predict salaries using job title, company size, remote ratio, location, and experience level.
- Supports currency switch (USD, INR, EUR, etc.).
- Shows model evaluation metrics: **Accuracy, Precision, Recall, F1-Score**.
- Includes model performance charts: Confusion matrix and feature importance.
- Clean and interactive UI using **Streamlit**.
- Built with proper ML pipeline, preprocessing, and optimized models (RandomForest, XGBoost).

---

## 🧠 Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python (scikit-learn, XGBoost)
- **Libraries**: Pandas, NumPy, Seaborn, Matplotlib, Joblib
- **Deployment**: Streamlit Cloud
- **Model Files**: Saved using `joblib`

---

## 🗂️ Project Structure

EmployeeSalaryPredictor/
│
├── data/ # Raw and cleaned dataset files
├── model/ # Trained model and supporting files
│ ├── salary_model.pkl
│ ├── label_encoders.pkl
│ ├── confusion_matrix.pkl
│ └── evaluation_report.pkl
│
├── predictor_app.py # Main Streamlit app
├── train_model.py # Script for model training & saving
├── requirements.txt # Project dependencies
└── README.md # You're here!

📈 Model Performance
✅ R² Score: Above 0.90

📊 Metrics: Accuracy, Precision, Recall, F1-Score

🔍 Feature Importance visualized using Matplotlib
