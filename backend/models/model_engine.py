import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import traceback

# v2.0.0 - COMPLETE REWRITE - All numpy, no DataFrame feature names
class FraudDetectionModel:
    def __init__(self, model_path='data/model.joblib'):
        self.model_path = model_path
        self.rf_model = None
        self.lr_model = None
        self.le_dict = {}
        self.features = ['Amount', 'Location', 'Transaction_Type', 'Hour', 'Device']
        
        # DO NOT load old model - it has stale feature names that crash sklearn
        # Model will be trained fresh each session
            
    def train(self, df):
        """v2.0.0: Complete rewrite using pure numpy to avoid all pandas index issues."""
        print("=== TRAIN v2.0.0 START ===")
        
        df = df.copy()
        df.dropna(inplace=True)
        print(f"  Rows after dropna: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        
        # Step 1: Fit label encoders and store them
        for col in ['Location', 'Transaction_Type', 'Device']:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            self.le_dict[col] = le
        print("  LabelEncoders fitted OK")
        
        # Step 2: Encode everything into plain python lists
        amounts = pd.to_numeric(df['Amount'], errors='coerce').fillna(0).tolist()
        locations = self.le_dict['Location'].transform(df['Location'].astype(str)).tolist()
        types = self.le_dict['Transaction_Type'].transform(df['Transaction_Type'].astype(str)).tolist()
        devices = self.le_dict['Device'].transform(df['Device'].astype(str)).tolist()
        
        hours = []
        for t in df['Time']:
            try:
                hours.append(int(str(t).split(':')[0]))
            except:
                hours.append(12)
        print("  Feature encoding OK")
        
        # Step 3: Build X as a PURE NUMPY ARRAY (no column names = no sklearn feature name issues)
        X = np.column_stack([amounts, locations, types, hours, devices])
        y = np.array([1 if str(s).lower() == 'fraud' else 0 for s in df['Status']])
        print(f"  X shape: {X.shape}, y shape: {y.shape}")
        print(f"  Fraud count: {sum(y)}, Safe count: {len(y) - sum(y)}")
        
        # Step 4: Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Step 5: Train Random Forest
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_model.fit(X_train, y_train)
        rf_score = self.rf_model.score(X_test, y_test)
        print(f"  RF accuracy: {rf_score}")
        
        # Step 6: Train Logistic Regression
        self.lr_model = LogisticRegression(max_iter=1000)
        self.lr_model.fit(X_train, y_train)
        lr_score = self.lr_model.score(X_test, y_test)
        print(f"  LR accuracy: {lr_score}")
        
        # Step 7: Save model
        if not os.path.exists('data'):
            os.makedirs('data')
        joblib.dump(self.rf_model, self.model_path)
        print("=== TRAIN v2.0.0 COMPLETE ===")
        
        return {
            'rf_accuracy': round(rf_score * 100, 2),
            'lr_accuracy': round(lr_score * 100, 2)
        }

    def predict(self, data_point):
        """Predict fraud probability for a single transaction."""
        if not self.rf_model:
            return {"error": "Model not trained", "risk_score": 0}
        
        try:
            loc_enc = self.le_dict['Location'].transform([str(data_point['Location'])])[0] if 'Location' in self.le_dict else 0
            type_enc = self.le_dict['Transaction_Type'].transform([str(data_point['Transaction_Type'])])[0] if 'Transaction_Type' in self.le_dict else 0
            dev_enc = self.le_dict['Device'].transform([str(data_point['Device'])])[0] if 'Device' in self.le_dict else 0
            
            hour = 12
            time_val = data_point.get('Time', '')
            if isinstance(time_val, str) and ':' in time_val:
                try:
                    hour = int(time_val.split(':')[0])
                except:
                    hour = 12
            
            # PURE NUMPY - no feature names
            features = np.array([[float(data_point['Amount']), loc_enc, type_enc, hour, dev_enc]])
            
            prob = self.rf_model.predict_proba(features)[0][1]
            prediction = 1 if prob > 0.5 else 0
            risk_score = int(prob * 100)
            
            return {
                'is_fraud': bool(prediction),
                'risk_score': risk_score,
                'confidence': round(max(prob, 1 - prob) * 100, 2)
            }
        except Exception as e:
            print(f"  PREDICT ERROR: {str(e)}")
            traceback.print_exc()
            return {"error": str(e), "risk_score": 0}

    def explain(self, data_point):
        """Generate explanation for a prediction."""
        if not self.rf_model:
            return None
        
        try:
            loc_enc = self.le_dict['Location'].transform([str(data_point['Location'])])[0] if 'Location' in self.le_dict else 0
            type_enc = self.le_dict['Transaction_Type'].transform([str(data_point['Transaction_Type'])])[0] if 'Transaction_Type' in self.le_dict else 0
            dev_enc = self.le_dict['Device'].transform([str(data_point['Device'])])[0] if 'Device' in self.le_dict else 0
            
            hour = 12
            time_val = data_point.get('Time', '')
            if isinstance(time_val, str) and ':' in time_val:
                try:
                    hour = int(time_val.split(':')[0])
                except:
                    hour = 12
            
            feature_names = ['Amount', 'Location', 'Transaction_Type', 'Hour', 'Device']
            importances = self.rf_model.feature_importances_
            contributions = dict(zip(feature_names, [float(v) for v in importances]))
            
            point_explanation = {}
            for feature in feature_names:
                base_importance = contributions[feature]
                if feature == 'Amount':
                    weight = 1.5 if float(data_point['Amount']) > 5000 else 0.5
                else:
                    weight = 1.0
                point_explanation[feature] = base_importance * weight

            sorted_contrib = sorted(point_explanation.items(), key=lambda x: abs(x[1]), reverse=True)
            
            counterfactual = []
            if float(data_point['Amount']) > 5000:
                counterfactual.append("Reduce transaction amount below ₹5000")
            if hour < 6 or hour > 22:
                counterfactual.append("Perform transaction during standard business hours (9 AM - 6 PM)")
            
            return {
                'feature_importance': point_explanation,
                'top_features': sorted_contrib[:3],
                'counterfactuals': counterfactual
            }
        except Exception as e:
            print(f"  EXPLAIN ERROR: {str(e)}")
            traceback.print_exc()
            return None
