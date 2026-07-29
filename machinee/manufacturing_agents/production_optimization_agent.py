"""
agents/production_optimization_agent.py
AURA Smart Manufacturing AI

Analyzes operating load distribution, RPM patterns, power usage efficiency
and suggests production optimization actions (de-rating, scheduling shift,
load balancing).
"""

import pandas as pd


class ProductionOptimizationAgent:
    """
    Analyzes operating load distribution, RPM patterns, power usage efficiency
    and suggests production optimization actions (de-rating, scheduling shift, load balancing).
    """
    def __init__(self, coordinator):
        self.coordinator = coordinator

    def optimize(self, df: pd.DataFrame) -> dict:
        results = {}

        # Compute operational indices
        # Load efficiency = Load_Percentage / Power_Consumption_kW (Work per kW)
        df_opt = df.copy()
        # Avoid division by zero
        df_opt['Power_Consumption_kW'] = df_opt['Power_Consumption_kW'].replace(0, 0.1)
        df_opt['Load_Efficiency'] = df_opt['Load_Percentage'] / df_opt['Power_Consumption_kW']

        # Analyze average efficiency per machine
        machine_stats = df_opt.groupby('Machine_ID').agg({
            'Machine_Name': 'first',
            'Load_Percentage': 'mean',
            'Power_Consumption_kW': 'mean',
            'Load_Efficiency': 'mean',
            'Operating_Hours': 'max',
            'Remaining_Useful_Life_Days': 'mean'
        }).reset_index()

        machine_stats.columns = ['Machine_ID', 'Machine_Name', 'Avg_Load_Pct', 'Avg_Power_kW', 'Avg_Load_Efficiency', 'Total_Hours', 'Avg_RUL']
        machine_stats = machine_stats.round(2)

        # Identify overloaded or highly inefficient machines
        # High load, high hours, low remaining life, or low efficiency
        optimization_insights = []

        for idx, row in machine_stats.iterrows():
            m_id = row['Machine_ID']
            m_name = row['Machine_Name']
            eff = row['Avg_Load_Efficiency']
            load = row['Avg_Load_Pct']
            rul = row['Avg_RUL']

            status_tag = "Optimal"
            recs = []

            if load > 85.0 and rul < 30:
                status_tag = "Risk of Overload Degradation"
                recs.append(f"De-rate load by 15-20% to preserve Remaining Useful Life ({rul} days left).")
            if eff < 0.5:
                status_tag = "Low Energy Efficiency"
                recs.append("Check internal friction drag, inspect mechanical drives, or replace worn motors.")
            if load < 30.0 and row['Avg_Power_kW'] > 15.0:
                status_tag = "Underloaded Energy Drain"
                recs.append("Consolidate workloads onto other active machines. Power down or enter Eco-standby mode.")

            if not recs:
                recs.append("Maintain current operational schedule. Keep load balanced.")

            optimization_insights.append({
                'Machine_ID': m_id,
                'Machine_Name': m_name,
                'Load_Efficiency': eff,
                'Status_Tag': status_tag,
                'Actionable_Insight': " | ".join(recs)
            })

        opt_insights_df = pd.DataFrame(optimization_insights)

        # Suggest Load Balancing
        # Find healthiest/most efficient machines to transfer load TO
        # and least healthy/least efficient to transfer load FROM
        healthy_high_eff = machine_stats[machine_stats['Avg_RUL'] > 60].sort_values(by='Avg_Load_Efficiency', ascending=False)
        degrading_high_load = machine_stats[(machine_stats['Avg_RUL'] < 30) & (machine_stats['Avg_Load_Pct'] > 70)]

        balancing_actions = []
        for idx, deg in degrading_high_load.iterrows():
            if not healthy_high_eff.empty:
                target = healthy_high_eff.iloc[0]
                balancing_actions.append({
                    'Source_Machine_ID': deg['Machine_ID'],
                    'Source_Machine_Name': deg['Machine_Name'],
                    'Target_Machine_ID': target['Machine_ID'],
                    'Target_Machine_Name': target['Machine_Name'],
                    'Action': f"Transfer 15% operational load from {deg['Machine_Name']} to {target['Machine_Name']}.",
                    'Reason': f"{deg['Machine_Name']} is degrading rapidly (RUL: {deg['Avg_RUL']} days), while {target['Machine_Name']} has high efficiency and ample life."
                })

        results['machine_stats'] = machine_stats
        results['opt_insights'] = opt_insights_df
        results['balancing_actions'] = pd.DataFrame(balancing_actions)
        results['avg_factory_efficiency'] = round(float(machine_stats['Avg_Load_Efficiency'].mean()), 2)

        return results
