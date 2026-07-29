"""
agents/failure_prediction_agent.py
AURA Smart Manufacturing AI

Handles machine learning task: dynamic Random Forest training to predict
whether a machine is at risk of fault / requires maintenance.
Provides feature importance and evaluation metrics.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class FailurePredictionAgent:
    """
    Handles machine learning task: dynamic Random Forest training to predict
    whether a machine is at risk of fault / requires maintenance.
    Provides feature importance and evaluation metrics.
    """
    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.model = None
        self.feature_columns = [
            'Operating_Hours', 'Temperature_C', 'Vibration_mm_s',
            'Pressure_bar', 'Power_Consumption_kW', 'Load_Percentage',
            'Oil_Level_Percentage', 'Humidity_Percentage', 'RPM'
        ]
        self.target_column = 'Fault_Status'
        self.metrics = {}
        self.feature_importances = pd.DataFrame()

    def train_and_predict(self, df: pd.DataFrame) -> dict:
        results = {}

        # Prepare target
        # Ensure target is binary
        y_raw = df[self.target_column]
        # Robust binary mapping for target columns to prevent parsing exceptions
        y = y_raw.apply(lambda x: 1 if str(x).strip().lower() in ['1', '1.0', 'yes', 'true', 'fault', 'critical', 'fail', 'y'] else 0)

        X = df[self.feature_columns]

        # Fill numerical missing values in features just in case
        X = X.fillna(X.median())

        # Check if we have multiple classes to train on
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            # Fallback if the dataset does not have binary classes
            # Artificially define faults based on a combination of extreme metrics
            y = ((df['Temperature_C'] > 85) | (df['Vibration_mm_s'] > 5.5) | (df['Remaining_Useful_Life_Days'] < 10)).astype(int)
            unique_classes = np.unique(y)

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Initialize and train Random Forest Classifier
        self.model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        self.model.fit(X_train, y_train)

        # Generate Test Set Predictions and Evaluate
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        self.metrics = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1_Score': f1,
            'Confusion_Matrix': cm.tolist()
        }

        # Extract Feature Importances
        importances = self.model.feature_importances_
        self.feature_importances = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        # Compute Failure Probability on the Entire Dataset
        df_predictions = df.copy()
        df_predictions['Predicted_Fault_Probability'] = self.model.predict_proba(X)[:, 1]
        df_predictions['Predicted_Fault_Class'] = self.model.predict(X)

        results['metrics'] = self.metrics
        results['feature_importances'] = self.feature_importances
        results['df_with_predictions'] = df_predictions

        return results

    def predict_custom(self, features_dict: dict) -> dict:
        """
        Predicts failure probability for a single set of manual inputs
        """
        if self.model is None:
            return {"error": "Model not trained yet."}

        input_df = pd.DataFrame([features_dict])
        input_data = input_df[self.feature_columns]

        prob = self.model.predict_proba(input_data)[0][1]
        pred_class = self.model.predict(input_data)[0]

        return {
            'Failure_Probability': round(float(prob) * 100, 2),
            'Prediction': "Fault Imminent / Maintenance Required" if pred_class == 1 else "Normal Operations"
        }
