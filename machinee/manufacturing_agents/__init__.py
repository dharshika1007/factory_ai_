"""
agents/__init__.py
AURA Smart Manufacturing AI — Agent Package
Re-exports all agent classes and utility functions for clean top-level imports.
"""

from .machine_health_agent import MachineHealthAgent
from .failure_prediction_agent import FailurePredictionAgent
from .maintenance_agent import MaintenanceAgent
from .production_optimization_agent import ProductionOptimizationAgent
from .report_agent import ReportAgent, NumberedCanvas, generate_matplotlib_charts, generate_pdf_report, compute_financial_loss_data
from .coordinator_agent import CoordinatorAgent

__all__ = [
    "MachineHealthAgent",
    "FailurePredictionAgent",
    "MaintenanceAgent",
    "ProductionOptimizationAgent",
    "ReportAgent",
    "NumberedCanvas",
    "generate_matplotlib_charts",
    "generate_pdf_report",
    "compute_financial_loss_data",
    "CoordinatorAgent",
]
