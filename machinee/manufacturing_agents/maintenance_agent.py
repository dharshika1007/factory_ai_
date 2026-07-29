"""
agents/maintenance_agent.py
AURA Smart Manufacturing AI

Coordinates predictive maintenance planning. Ranks machines by urgency
combining health score, failure probability, and remaining useful life.
Recommends specific maintenance tasks.
"""

import pandas as pd


class MaintenanceAgent:
    """
    Coordinates predictive maintenance planning. Rank machines by urgency combining health score,
    failure probability, and remaining useful life. Recommends specific maintenance tasks.
    """
    def __init__(self, coordinator):
        self.coordinator = coordinator

    def schedule(self, df: pd.DataFrame, health_results: dict, prediction_results: dict) -> dict:
        results = {}

        # Get machine health summaries and predictions
        health_summary = health_results['machine_summary']
        df_preds = prediction_results['df_with_predictions']

        # Group predictions by Machine
        pred_summary = df_preds.groupby('Machine_ID')['Predicted_Fault_Probability'].mean().reset_index()

        # Merge health summaries with average predictions
        merged = pd.merge(health_summary, pred_summary, on='Machine_ID')

        # Calculate Maintenance Urgency Score (0 to 100)
        # Urgency is higher if Health Score is low, Failure Probability is high, and Remaining Useful Life (RUL) is low
        # Urgency = (100 - Health_Score)*0.4 + (Fail_Prob*100)*0.4 + (max(0, 100 - RUL_Days*2))*0.2
        merged['Urgency_Score'] = (
            (100.0 - merged['Health_Score']) * 0.45 +
            (merged['Predicted_Fault_Probability'] * 100.0) * 0.35 +
            (merged['Avg_RUL_Days'].apply(lambda x: max(0.0, 100.0 - x * 2.0))) * 0.20
        )
        merged['Urgency_Score'] = merged['Urgency_Score'].apply(lambda x: round(min(100.0, max(0.0, x)), 1))

        # Classify Urgency Rank
        def get_priority(score):
            if score >= 75.0:
                return 'CRITICAL (Immediate Action)'
            elif score >= 45.0:
                return 'HIGH (Schedule within 48h)'
            elif score >= 20.0:
                return 'MEDIUM (Schedule within 7 Days)'
            else:
                return 'LOW (Routine Inspection)'

        merged['Priority'] = merged['Urgency_Score'].apply(get_priority)

        # Generate targeted maintenance actions
        maintenance_queue = []
        for idx, row in merged.iterrows():
            m_id = row['Machine_ID']
            m_name = row['Machine_Name']
            oil = row['Min_Oil_Level_Percentage']
            vib = row['Max_Vibration_mm_s']
            temp = row['Max_Temperature_C']
            rul = row['Avg_RUL_Days']

            actions = []
            if oil < 30.0:
                actions.append("Replenish/Flush gear oil immediately.")
            if vib > 4.5:
                actions.append("Check bearings alignment, check structural mount stiffness.")
            if temp > 80.0:
                actions.append("Inspect cooling fan and radiator. Clear ventilation pathways.")
            if rul < 15.0:
                actions.append("Schedule mechanical assembly overhaul (low remaining life).")
            if not actions:
                actions.append("Perform routine cleaning and standard lubrication checks.")

            maintenance_queue.append({
                'Machine_ID': m_id,
                'Machine_Name': m_name,
                'Urgency_Score': row['Urgency_Score'],
                'Priority': row['Priority'],
                'Health_Score': row['Health_Score'],
                'RUL_Days': rul,
                'Recommended_Actions': " | ".join(actions)
            })

        queue_df = pd.DataFrame(maintenance_queue).sort_values(by='Urgency_Score', ascending=False)

        results['maintenance_queue'] = queue_df
        results['critical_maintenance_count'] = int(len(queue_df[queue_df['Priority'].str.contains('CRITICAL')]))
        results['high_maintenance_count'] = int(len(queue_df[queue_df['Priority'].str.contains('HIGH')]))

        return results
