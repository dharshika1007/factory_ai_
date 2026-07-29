"""
agents/machine_health_agent.py
AURA Smart Manufacturing AI

Analyzes physical machine metrics (Temperature, Vibration, Pressure, RPM,
Humidity, Oil Level) to detect anomalies, assign health scores, and flag
health status.
"""

import pandas as pd
import numpy as np


class MachineHealthAgent:
    """
    Analyzes physical machine metrics (Temperature, Vibration, Pressure, RPM, Humidity, Oil Level)
    to detect anomalies, assign health scores, and flag health status.
    """
    def __init__(self, coordinator):
        self.coordinator = coordinator

    def analyze(self, df: pd.DataFrame) -> dict:
        results = {}

        # Telemetry thresholds for anomalies (based on typical heavy manufacturing standard ranges)
        thresholds = {
            'Temperature_C': {'high': 80.0, 'critical': 95.0},
            'Vibration_mm_s': {'high': 4.5, 'critical': 7.0},
            'Pressure_bar': {'low': 20.0, 'high': 120.0, 'critical_high': 150.0},
            'Oil_Level_Percentage': {'low': 25.0, 'critical_low': 15.0},
            'RPM': {'high': 2800, 'critical_high': 3200}
        }

        # Make a copy of dataframe to manipulate
        df_health = df.copy()

        # Calculate anomaly indicators
        df_health['Temp_Anomaly'] = df_health['Temperature_C'] > thresholds['Temperature_C']['high']
        df_health['Vib_Anomaly'] = df_health['Vibration_mm_s'] > thresholds['Vibration_mm_s']['high']
        df_health['Pres_Anomaly'] = (df_health['Pressure_bar'] > thresholds['Pressure_bar']['high']) | (df_health['Pressure_bar'] < thresholds['Pressure_bar']['low'])
        df_health['Oil_Anomaly'] = df_health['Oil_Level_Percentage'] < thresholds['Oil_Level_Percentage']['low']

        df_health['Anomaly_Count'] = (
            df_health['Temp_Anomaly'].astype(int) +
            df_health['Vib_Anomaly'].astype(int) +
            df_health['Pres_Anomaly'].astype(int) +
            df_health['Oil_Anomaly'].astype(int)
        )

        # Group metrics by Machine
        machine_groups = df_health.groupby('Machine_ID')
        machine_summaries = []

        for name, group in machine_groups:
            machine_name = group['Machine_Name'].iloc[0]
            avg_temp = group['Temperature_C'].mean()
            max_temp = group['Temperature_C'].max()
            avg_vib = group['Vibration_mm_s'].mean()
            max_vib = group['Vibration_mm_s'].max()
            avg_pres = group['Pressure_bar'].mean()
            min_oil = group['Oil_Level_Percentage'].min()
            avg_load = group['Load_Percentage'].mean()
            avg_rpm = group['RPM'].mean()
            avg_rul = group['Remaining_Useful_Life_Days'].mean()
            total_anomalies = group['Anomaly_Count'].sum()

            # Formulate robust Machine Health Score (0-100)
            # Subtract points for average values exceeding parameters, low oil level, and anomalies count
            temp_penalty = max(0, (avg_temp - 65) * 0.8) if avg_temp > 65 else 0
            vib_penalty = max(0, (avg_vib - 3.0) * 10.0) if avg_vib > 3.0 else 0
            oil_penalty = max(0, (50 - min_oil) * 1.5) if min_oil < 50 else 0
            anomaly_penalty = total_anomalies * 5.0

            health_score = 100.0 - (temp_penalty + vib_penalty + oil_penalty + anomaly_penalty)
            # Adjust health score to bounds [0, 100]
            health_score = max(0.0, min(100.0, health_score))

            # Map health score to categories
            if health_score < 50.0:
                health_category = 'Critical'
            elif health_score < 80.0:
                health_category = 'Fair'
            else:
                health_category = 'Good'

            machine_summaries.append({
                'Machine_ID': name,
                'Machine_Name': machine_name,
                'Health_Score': round(health_score, 1),
                'Health_Category': health_category,
                'Avg_Temperature_C': round(avg_temp, 2),
                'Max_Temperature_C': round(max_temp, 2),
                'Avg_Vibration_mm_s': round(avg_vib, 2),
                'Max_Vibration_mm_s': round(max_vib, 2),
                'Avg_Pressure_bar': round(avg_pres, 2),
                'Min_Oil_Level_Percentage': round(min_oil, 1),
                'Avg_Load_Percentage': round(avg_load, 1),
                'Avg_RPM': round(avg_rpm, 1),
                'Avg_RUL_Days': round(avg_rul, 1),
                'Total_Anomalies': int(total_anomalies)
            })

        machine_summary_df = pd.DataFrame(machine_summaries)

        # Store results
        results['df_with_anomalies'] = df_health
        results['machine_summary'] = machine_summary_df
        results['thresholds'] = thresholds
        results['total_anomalies'] = int(df_health['Anomaly_Count'].sum())
        results['critical_count'] = int(len(machine_summary_df[machine_summary_df['Health_Category'] == 'Critical']))
        results['fair_count'] = int(len(machine_summary_df[machine_summary_df['Health_Category'] == 'Fair']))
        results['good_count'] = int(len(machine_summary_df[machine_summary_df['Health_Category'] == 'Good']))

        return results
