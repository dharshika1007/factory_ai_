"""
agents/coordinator_agent.py
AURA Smart Manufacturing AI

Central orchestration agent. Manages data loading, validation, preprocessing,
and runs all sub-agents sequentially to construct the intelligence output.
"""

import pandas as pd
import numpy as np
import os

from .machine_health_agent import MachineHealthAgent
from .failure_prediction_agent import FailurePredictionAgent
from .maintenance_agent import MaintenanceAgent
from .production_optimization_agent import ProductionOptimizationAgent
from .report_agent import ReportAgent


class CoordinatorAgent:
    """
    Central orchestration agent. Manages loading, validation, preprocessing,
    and runs sub-agents sequentially to construct the intelligence output.
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.df = None
        self.is_processed = False

        # Instantiate sub-agents
        self.health_agent = MachineHealthAgent(self)
        self.failure_agent = FailurePredictionAgent(self)
        self.maintenance_agent = MaintenanceAgent(self)
        self.optimization_agent = ProductionOptimizationAgent(self)
        self.report_agent = ReportAgent(self, self.api_key)

        # Pipeline results stores
        self.health_results = None
        self.prediction_results = None
        self.maintenance_results = None
        self.optimization_results = None
        self.ai_report = None

    def load_and_preprocess_data(self, uploaded_file) -> tuple:
        try:
            df = pd.read_csv(uploaded_file)

            # Column requirements verification
            required_cols = [
                'Machine_ID', 'Machine_Name', 'Start_Time', 'End_Time',
                'Operating_Hours', 'Temperature_C', 'Vibration_mm_s',
                'Pressure_bar', 'Power_Consumption_kW', 'Load_Percentage',
                'Oil_Level_Percentage', 'Humidity_Percentage', 'RPM',
                'Remaining_Useful_Life_Days', 'Machine_Health',
                'Fault_Status', 'Maintenance_Required'
            ]

            # Map case-insensitive column headers and replace underscores if needed
            cols_map = {}
            for col in df.columns:
                cleaned_col = str(col).strip().lower().replace("_", "").replace(" ", "")
                for req in required_cols:
                    cleaned_req = req.lower().replace("_", "")
                    if cleaned_col == cleaned_req:
                        cols_map[col] = req
                        break

            df = df.rename(columns=cols_map)

            # Find missing columns
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return False, f"Missing required columns in dataset: {', '.join(missing_cols)}"

            # Perform basic data cleaning and parsing
            # Handle timestamps
            df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
            df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')

            # Numerical column sanitization
            numeric_cols = [
                'Operating_Hours', 'Temperature_C', 'Vibration_mm_s',
                'Pressure_bar', 'Power_Consumption_kW', 'Load_Percentage',
                'Oil_Level_Percentage', 'Humidity_Percentage', 'RPM',
                'Remaining_Useful_Life_Days'
            ]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Fill missing with median
                df[col] = df[col].fillna(df[col].median())

            # Clean categorical columns
            df['Machine_ID'] = df['Machine_ID'].astype(str)
            df['Machine_Name'] = df['Machine_Name'].astype(str)
            df['Machine_Health'] = df['Machine_Health'].fillna('Good').astype(str)

            self.df = df
            self.is_processed = True
            return True, "Data successfully ingested, cleaned and normalized."

        except Exception as e:
            return False, f"Ingestion Error: {str(e)}"

    def run_pipeline(self) -> bool:
        if not self.is_processed or self.df is None:
            return False

        try:
            # 1. Analyze Machine Health
            self.health_results = self.health_agent.analyze(self.df)

            # 2. Train Random Forest and Predict Failures
            self.prediction_results = self.failure_agent.train_and_predict(self.df)

            # 3. Schedule Preventive Maintenance
            self.maintenance_results = self.maintenance_agent.schedule(
                self.df, self.health_results, self.prediction_results
            )

            # 4. Analyze Production Load & Power
            self.optimization_results = self.optimization_agent.optimize(self.df)

            # Reset generated report (will be built on request)
            self.ai_report = None
            return True
        except Exception as e:
            # Import streamlit here only for the error display, avoiding a top-level circular dep
            try:
                import streamlit as st
                st.error(f"Pipeline Pipeline Execution Failure: {str(e)}")
            except Exception:
                print(f"Pipeline Execution Failure: {str(e)}")
            return False

    def generate_ai_report(self) -> str:
        if not self.health_results:
            return "Please run the pipeline analysis first."

        if self.ai_report is None:
            self.ai_report = self.report_agent.generate_report(
                self.health_results,
                self.prediction_results,
                self.maintenance_results,
                self.optimization_results
            )
        return self.ai_report
