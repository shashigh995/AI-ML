import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# v1.0.5 - NUCLEAR RESET - Naming fix forced
class FraudDetectionModel:
    def __init__(self, model_path='data/model.joblib'):
        self.model_path = model_path
        self.rf_model = None
        self.lr_model = None
        self.scaler = StandardScaler()
        self.le_dict = {}
        self.X_test = None
        self.y_test = None
        self.features = ['Amount', 'Location_Encoded', 'Type_Encoded', 'Hour', 'Device_Encoded']
        
        if os.path.exists(self.model_path):
            self.load_model()
            
    def preprocess_data(self, df):
        df = df.copy()
        df.dropna(inplace=True)
        
        # Encoding categorical
        df['Location_Encoded'] = LabelEncoder().fit_transform(df['Location'])
        df['Type_Encoded'] = LabelEncoder().fit_transform(df['Transaction_Type'])
        df['Device_Encoded'] = LabelEncoder().fit_transform(df['Device'])
        
        # Store for future single-point predictions
        for col in ['Location', 'Transaction_Type', 'Device']:
            le = LabelEncoder()
            le.fit(df[col])
            self.le_dict[col] = le
            
        # Feature Engineering
        if 'Time' in df.columns:
            df['Hour'] = df['Time'].apply(lambda x: int(x.split(':')[0]) if isinstance(x, str) and ':' in x else 12)
        else:
            df['Hour'] = 12
            
        return df

    def train(self, df):
        # NUCLEAR RESET: Encode everything here to be 100% sure
        df = df.copy()
        df.dropna(inplace=True)
        
        # MASTER FIX v1.0.6: Universal Mapping
        df['Location_Encoded'] = LabelEncoder().fit_transform(df['Location'].astype(str))
        df['Type_Encoded'] = LabelEncoder().fit_transform(df['Transaction_Type'].astype(str))
        df['Device_Encoded'] = LabelEncoder().fit_transform(df['Device'].astype(str))
        
        if 'Time' in df.columns:
            df['Hour'] = df['Time'].apply(lambda x: int(x.split(':')[0]) if isinstance(x, str) and ':' in x else 12)
        else:
            df['Hour'] = 12
            
        final_features = ['Amount', 'Location_Encoded', 'Type_Encoded', 'Hour', 'Device_Encoded']
        # Explicit check before sub-selecting
        for f in final_features:
            if f not in df.columns:
                print(f"CRITICAL WARNING: {f} missing. Creating fallback...")
                df[f] = 0
                
        X = df[final_features]
        y = df['Status'].apply(lambda x: 1 if str(x).lower() == 'fraud' else 0)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_test = X_test
        self.y_test = y_test
        
        # Random Forest
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_model.fit(X_train, y_train)
        rf_score = self.rf_model.score(X_test, y_test)
        
        # Logistic Regression
        self.lr_model = LogisticRegression()
        self.lr_model.fit(X_train, y_train)
        lr_score = self.lr_model.score(X_test, y_test)
        
        # Save RF as main model
        if not os.path.exists('data'): os.makedirs('data')
        joblib.dump(self.rf_model, self.model_path)
        
        return {
            'rf_accuracy': round(rf_score * 100, 2),
            'lr_accuracy': round(lr_score * 100, 2)
        }

    def predict(self, data_point):
        """
        data_point: dict with keys [Amount, Location, Transaction_Type, Time, Device]
        """
        if not self.rf_model:
            return {"error": "Model not trained"}
        
        # Preprocess single point
        loc_enc = self.le_dict['Location'].transform([data_point['Location']])[0] if 'Location' in self.le_dict else 0
        type_enc = self.le_dict['Transaction_Type'].transform([data_point['Transaction_Type']])[0] if 'Transaction_Type' in self.le_dict else 0
        dev_enc = self.le_dict['Device'].transform([data_point['Device']])[0] if 'Device' in self.le_dict else 0
        hour = int(data_point['Time'].split(':')[0]) if ':' in data_point['Time'] else 12
        
        features = np.array([[data_point['Amount'], loc_enc, type_enc, hour, dev_enc]])
        
        prob = self.rf_model.predict_proba(features)[0][1] # Probability of Fraud
        prediction = 1 if prob > 0.5 else 0
        
        risk_score = int(prob * 100)
        
        return {
            'is_fraud': bool(prediction),
            'risk_score': risk_score,
            'confidence': round(max(prob, 1-prob) * 100, 2)
        }

    def explain(self, data_point):
        if not self.rf_model: return None
        
        # Turbo Lite: Use RF's internal feature importances to avoid SHAP OOM on small servers
        # Construct feature vector exactly as in training
        loc_enc = self.le_dict['Location'].transform([data_point['Location']])[0] if 'Location' in self.le_dict else 0
        type_enc = self.le_dict['Transaction_Type'].transform([data_point['Transaction_Type']])[0] if 'Transaction_Type' in self.le_dict else 0
        dev_enc = self.le_dict['Device'].transform([data_point['Device']])[0] if 'Device' in self.le_dict else 0
        hour = int(data_point['Time'].split(':')[0]) if ':' in data_point['Time'] else 12
        
        # Get overall model importances as a robust fallback
        importances = self.rf_model.feature_importances_
        contributions = dict(zip(self.features, [float(v) for v in importances]))
        
        # Heuristic for single-point weighting (Simple but effective for demo)
        # We simulate the direction based on simple relative thresholds
        point_explanation = {}
        for feature in self.features:
            base_importance = contributions[feature]
            if feature == 'Amount':
                weight = 1.5 if data_point['Amount'] > 5000 else 0.5
            else:
                weight = 1.0
            point_explanation[feature] = base_importance * weight

        sorted_contrib = sorted(point_explanation.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Generate Counterfactual Logic
        counterfactual = []
        if data_point['Amount'] > 5000:
            counterfactual.append(f"Reduce transaction amount below ₹5000")
        if hour < 6 or hour > 22:
            counterfactual.append("Perform transaction during standard business hours (9 AM - 6 PM)")
        
        return {
            'feature_importance': point_explanation,
            'top_features': sorted_contrib[:3],
            'counterfactuals': counterfactual
        }

    def load_model(self):
        self.rf_model = joblib.load(self.model_path)
