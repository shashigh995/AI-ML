# Explainable Finance AI for Fraud & Risk Detection System

A complete full-stack AI-powered web application that detects fraudulent financial transactions using Machine Learning and explains the predictions using SHAP (Explainable AI).

## 🚀 Features
- **Fraud Detection**: Predicts if a transaction is 'Fraud' or 'Safe'.
- **Risk Scoring Engine**: Assigns a risk score (0-100) to each transaction.
- **SHAP-powered Explainable AI**: Visualizes feature importance and explains why a transaction was flagged.
- **Counterfactual Explanations**: Provides suggestions on why a transaction could be considered safe.
- **Analytics Dashboard**: Comprehensive charts (Chart.js) showing fraud trends and risk distribution.
- **Admin Control Panel**: Upload CSV datasets, clean the database, and re-train models.

## 🛠️ Tech Stack
- **Frontend**: HTML5, Vanilla CSS3, JavaScript (ES6+), Chart.js
- **Backend**: Python Flask, SQLite, SQLAlchemy
- **Machine Learning**: Scikit-learn (Random Forest, Logistic Regression)
- **Explainable AI**: SHAP (SHapley Additive exPlanations)

## 📦 Project Structure
- `app.py`: Main Flask server and API routes.
- `models/database.py`: SQLite database schema.
- `models/model_engine.py`: ML pipeline (Training, Prediction, Explanation).
- `static/`: Frontend assets (CSS/Images).
- `templates/`: HTML pages.
- `data/`: CSV datasets and saved models.

## 🏃 Run Instructions
1. **Clone the project** or copy the folder.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Start the server**:
   ```bash
   python app.py
   ```
4. **Access the application**:
   Open your browser and navigate to `http://127.0.0.1:5000/`

## 🛡️ Admin Login
- **Username**: `admin`
- **Password**: `password123`

## 📊 Sample Data
A sample CSV is provided in `data/sample_data.csv`. You can upload this file in the **Upload Data** page to populate the system.

---
Developed with focus on Explainable AI (XAI) for secure banking systems.
