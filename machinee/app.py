"""
app.py
AURA Smart Manufacturing AI — Streamlit Dashboard
Pure UI controller. All agent logic lives in the manufacturing_agents/ package.
"""

import sys
import os

# Ensure the project root is on sys.path so the local package is
# always found, regardless of the working directory Streamlit was launched from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import datetime
import tempfile
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Import agents package
from manufacturing_agents import (
    CoordinatorAgent,
    generate_matplotlib_charts,
    generate_pdf_report,
)

# Load environment variables
load_dotenv()


# ==========================================
# ==========================================
# HELPER — load Font Awesome icon library
# ==========================================
def load_icons() -> str:
    """Read icons.html and return its content for injection into Streamlit."""
    icons_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons.html")
    with open(icons_path, "r", encoding="utf-8") as f:
        return f.read()


# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="FACTORY AI | Agentic AI Smart Manufacturing Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Font Awesome + icon CSS
st.markdown(load_icons(), unsafe_allow_html=True)

# ==========================================
# GLOBAL CSS STYLING
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Orbitron:wght@400;500;700;900&family=JetBrains+Mono:wght@400;700&display=swap');

    /* Force Dark background globally with high-end dashboard colors */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #0b0f19 !important;
        background-image: 
            radial-gradient(at 0% 0%, rgba(30, 41, 59, 0.25) 0, transparent 50%),
            radial-gradient(at 50% 0%, rgba(99, 102, 241, 0.04) 0, transparent 50%),
            radial-gradient(at 100% 0%, rgba(15, 23, 42, 0.3) 0, transparent 50%) !important;
        background-attachment: fixed !important;
    }

    [data-testid="stSidebar"] {
        background-color: #0d1220 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.04) !important;
    }

    /* Base text colors */
    h1, h2, h3, h4, h5, h6, p, li, label, span, div, .stMarkdown, .stText, .stDataFrame, .stTable, .stSelectbox, .stSlider, .stButton, .stFileUploader, .stCaption, .stHelp {
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif;
    }

    /* Secondary text */
    p, li, .stCaption, .header-subtitle {
        color: #94a3b8 !important;
    }

    /* Header styling (Grafana style) */
    .header-container {
        background: linear-gradient(135deg, rgba(19, 27, 46, 0.8) 0%, rgba(13, 18, 32, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    .header-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 2rem;
        font-weight: 900;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: 1.5px;
        text-shadow: 0 0 30px rgba(56, 189, 248, 0.15);
    }
    .header-subtitle {
        font-size: 0.9rem;
        color: #94a3b8 !important;
        margin-top: 6px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Status Badge styling */
    .status-badge {
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
    }
    .status-good {
        background-color: rgba(16, 185, 129, 0.1) !important;
        color: #10b981 !important;
        border-color: rgba(16, 185, 129, 0.25) !important;
        box-shadow: 0 0 8px rgba(16, 185, 129, 0.1);
    }
    .status-fair {
        background-color: rgba(245, 158, 11, 0.1) !important;
        color: #fbbf24 !important;
        border-color: rgba(245, 158, 11, 0.25) !important;
        box-shadow: 0 0 8px rgba(245, 158, 11, 0.1);
    }
    .status-critical {
        background-color: rgba(239, 68, 68, 0.1) !important;
        color: #f87171 !important;
        border-color: rgba(239, 68, 68, 0.25) !important;
        box-shadow: 0 0 8px rgba(239, 68, 68, 0.1);
    }

    /* KPI Card styling (dark mode) */
    .kpi-card {
        background: linear-gradient(135deg, rgba(21, 28, 44, 0.7) 0%, rgba(13, 18, 32, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
    }
    .kpi-card:hover {
        border-color: rgba(56, 189, 248, 0.25);
        transform: translateY(-2px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.4), 0 0 10px rgba(56, 189, 248, 0.1);
    }
    .kpi-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }
    .kpi-value {
        font-family: 'Orbitron', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc !important;
        margin-top: 8px;
        letter-spacing: 0.5px;
    }
    .kpi-trend {
        font-size: 0.8rem;
        margin-top: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Section Headers */
    .section-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: #f8fafc !important;
        margin-bottom: 24px;
        margin-top: 10px;
        border-left: 4px solid #38bdf8;
        padding-left: 12px;
        letter-spacing: 1px;
    }

    /* Dataframe tables */
    [data-testid="stDataFrame"] > div, [data-testid="stTable"] {
        background-color: #131b2e !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }
    [data-testid="stDataFrame"] th, [data-testid="stDataFrame"] td, [data-testid="stTable"] th, [data-testid="stTable"] td {
        color: #cbd5e1 !important;
        background-color: #131b2e !important;
        border-color: rgba(255, 255, 255, 0.05) !important;
    }

    /* Custom menu styling for sidebar radio selection */
    [data-testid="stRadio"] div[role="radiogroup"] {
        gap: 6px !important;
    }
    [data-testid="stRadio"] label {
        padding: 10px 14px !important;
        border-radius: 8px !important;
        background-color: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        color: #94a3b8 !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
        font-weight: 500 !important;
        display: flex !important;
        align-items: center !important;
    }
    [data-testid="stRadio"] label:hover {
        background-color: rgba(56, 189, 248, 0.05) !important;
        color: #f8fafc !important;
        border-color: rgba(56, 189, 248, 0.15) !important;
        transform: translateX(2px);
    }
    [data-testid="stRadio"] label[data-checked="true"] {
        background: linear-gradient(135deg, rgba(30, 64, 175, 0.25) 0%, rgba(30, 58, 138, 0.45) 100%) !important;
        border-color: #38bdf8 !important;
        color: #38bdf8 !important;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.15) !important;
        font-weight: 600 !important;
    }
    [data-testid="stRadio"] label span {
        color: inherit !important;
    }
    [data-testid="stRadio"] label input {
        display: none !important;
    }

    /* Buttons */
    div.stButton > button, div.stDownloadButton > button, 
    button[kind="secondary"], button[kind="primary"], 
    [data-testid="baseButton-secondary"], [data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-family: 'Inter', sans-serif !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        min-height: 44px !important;
        padding: 0 20px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        letter-spacing: 0.5px !important;
    }
    
    div.stButton > button:hover, div.stDownloadButton > button:hover, 
    button[kind="secondary"]:hover, button[kind="primary"]:hover,
    [data-testid="baseButton-secondary"]:hover, [data-testid="baseButton-primary"]:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.4) !important;
        transform: translateY(-1.5px) !important;
        color: #ffffff !important;
    }
    
    div.stButton > button:active, div.stDownloadButton > button:active,
    button[kind="secondary"]:active, button[kind="primary"]:active,
    [data-testid="baseButton-secondary"]:active, [data-testid="baseButton-primary"]:active {
        transform: translateY(1px) !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3) !important;
    }

    div.stButton > button:disabled, div.stDownloadButton > button:disabled, 
    button[kind="secondary"]:disabled, button[kind="primary"]:disabled,
    [data-testid="baseButton-secondary"]:disabled, [data-testid="baseButton-primary"]:disabled {
        background: #151c2c !important;
        color: #475569 !important;
        border: 1px solid rgba(255, 255, 255, 0.02) !important;
        box-shadow: none !important;
        transform: none !important;
        cursor: not-allowed !important;
    }

    /* File Uploader */
    [data-testid="stFileUploader"] > div, [data-testid="stFileUploader"] > section {
        background-color: #131b2e !important;
        border-radius: 10px !important;
    }
    [data-testid="stFileUploadDropzone"] {
        background-color: #131b2e !important;
        border: 1px dashed rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
    }
    [data-testid="stFileUploader"] svg {
        fill: #38bdf8 !important;
        color: #38bdf8 !important;
    }
    [data-testid="stFileUploader"] small, 
    [data-testid="stFileUploader"] span, 
    [data-testid="stFileUploader"] label, 
    [data-testid="stFileUploaderFileName"] {
        color: #f8fafc !important;
    }
    [data-testid="stFileUploader"] button {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stFileUploader"] button:hover {
        background-color: #334155 !important;
        border-color: #38bdf8 !important;
    }

    /* Inputs: Text Input, Text Area, Selectbox, Slider */
    [data-testid="stTextInput"] input, 
    [data-testid="stTextArea"] textarea, 
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div, 
    [data-testid="stSlider"] div {
        background-color: #131b2e !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }
    [data-testid="stTextInput"] label, 
    [data-testid="stTextArea"] label, 
    [data-testid="stSelectbox"] label {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
    }
    ::placeholder {
        color: #475569 !important;
        opacity: 1 !important;
    }

    /* Tabs */
    [data-testid="stTabs"] button {
        color: #94a3b8 !important;
        background-color: transparent !important;
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.5px !important;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom-color: #38bdf8 !important;
    }

    /* Expander */
    [data-testid="stExpander"] {
        background-color: #131b2e !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
    }
    [data-testid="stExpander"] summary {
        border-bottom: 1px solid rgba(255, 255, 255, 0.03) !important;
    }
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary p {
        color: #f8fafc !important;
        font-weight: 600 !important;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 700 !important;
        color: #f8fafc !important;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }

    /* Alerts */
    [data-testid="stAlert"] {
        color: #f8fafc !important;
        background-color: rgba(21, 28, 44, 0.8) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 10px !important;
    }
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"], 
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {
        color: #cbd5e1 !important;
    }

    /* Asset status chip styling for grid mapping */
    .machine-status-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-bottom: 24px;
        background: rgba(19, 27, 46, 0.4);
        padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.03);
    }
    .machine-status-chip {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 10px 16px;
        border-radius: 8px;
        background: #131b2e;
        border: 1px solid rgba(255, 255, 255, 0.05);
        font-size: 0.85rem;
        color: #f8fafc;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .machine-status-chip:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.3);
        box-shadow: 0 6px 14px rgba(0, 0, 0, 0.3), 0 0 8px rgba(56, 189, 248, 0.1);
    }
    .machine-status-name {
        font-weight: 600;
        font-family: 'Inter', sans-serif;
    }
    .machine-status-score {
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        font-size: 0.8rem;
        padding: 2px 6px;
        border-radius: 4px;
        background: rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if 'coordinator' not in st.session_state:
    st.session_state.coordinator = None
if 'temp_coordinator' not in st.session_state:
    st.session_state.temp_coordinator = None
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
if 'download_timestamp' not in st.session_state:
    st.session_state.download_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')

# ==========================================
# CUSTOM HEADER
# ==========================================
st.markdown("""
<div class="header-container">
    <h1 class="header-title">FACTORY AI // SMART MANUFACTURING AI</h1>
    <p class="header-subtitle">Edge Telemetry Analytics • Multi-Agent Cognitive Pipeline • Predictive Reliability Core</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR — DATASET UPLOAD & ANALYZE
# ==========================================
active_tab = "Landing"

with st.sidebar:
    st.markdown(
        '<h3><i class="fa-solid fa-file-import aura-icon c-info"></i>Telemetry Ingestion</h3>',
        unsafe_allow_html=True
    )
    uploaded_file = st.file_uploader(
        "Upload Factory Dataset (CSV)",
        type=["csv"],
        help="Expected columns include Machine_ID, Temperature_C, Vibration_mm_s, Operating_Hours, Fault_Status, etc."
    )

    if uploaded_file is not None:
        if 'last_uploaded' not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
            st.session_state.last_uploaded = uploaded_file.name
            st.session_state.coordinator = None
            st.session_state.temp_coordinator = None
            st.session_state.analysis_run = False

            api_key = os.getenv("GROQ_API_KEY", "")
            coordinator = CoordinatorAgent(api_key=api_key)
            success, msg = coordinator.load_and_preprocess_data(uploaded_file)

            if success:
                st.session_state.temp_coordinator = coordinator
                st.success(msg)
            else:
                st.error(msg)
    else:
        st.session_state.last_uploaded = None
        st.session_state.coordinator = None
        st.session_state.temp_coordinator = None
        st.session_state.analysis_run = False

    # Analyze Button
    if st.session_state.temp_coordinator is not None and not st.session_state.get('analysis_run', False):
        st.markdown("---")
        if st.button(
            "Analyze Telemetry Data",
            help="Run the multi-agent cognitive pipeline on the uploaded telemetry dataset.",
            use_container_width=True
        ):
            with st.spinner("Executing agent intelligence layers..."):
                pipeline_success = st.session_state.temp_coordinator.run_pipeline()
                if pipeline_success:
                    st.session_state.coordinator = st.session_state.temp_coordinator
                    st.session_state.analysis_run = True
                    st.success("Analysis Completed!")
                    st.rerun()
                else:
                    st.error("Agent Pipeline execution failed.")
                    
    # Control Console Navigation
    if st.session_state.coordinator is not None and st.session_state.get('analysis_run', False):
        st.markdown("<br><hr style='border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
        st.markdown('<h3 class="sidebar-heading" style="color: #38bdf8; font-size: 1.1rem; font-family: \'Orbitron\', sans-serif; letter-spacing: 0.5px; margin-bottom: 12px;"><i class="fa-solid fa-compass aura-icon c-info"></i>Control Console</h3>', unsafe_allow_html=True)
        menu_options = [
            "📊 Factory Control Center",
            "🩺 Asset Diagnostics",
            "🧠 Predictive Reliability",
            "🔧 Maintenance Scheduler",
            "⚡ Grid & Power Optimization",
            "📋 Executive AI Report"
        ]
        active_tab = st.radio("SELECT DASHBOARD VIEW", menu_options, label_visibility="collapsed")

# ==========================================
# MAIN CONTENT
# ==========================================
if st.session_state.coordinator is None or not st.session_state.get('analysis_run', False):

    # ---- LANDING PAGE ----
    if uploaded_file is not None and st.session_state.temp_coordinator is not None:
        st.markdown("""
        <div class="kpi-card" style="margin-bottom: 24px;">
            <h3 style="font-family: 'Orbitron', sans-serif; color: #38bdf8; margin-top: 0; display: flex; align-items: center; gap: 8px;">
                <i class="fa-solid fa-square-check aura-icon c-success"></i> Telemetry Ingestion Complete
            </h3>
            <p style="font-size: 0.95rem; margin-top: 8px; color: #cbd5e1;">
                Your factory telemetry dataset was parsed successfully and is ready in-memory.
            </p>
            <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2); padding: 8px 16px; border-radius: 8px; color: #10b981; font-weight: 500; display: inline-flex; align-items: center; gap: 8px; margin-top: 8px; font-size: 0.85rem;">
                <span class="status-dot-glowing status-dot-good" style="margin: 0;"></span> Ready for Cognitive Analysis
            </div>
        </div>
        
        <h4 style="font-family: 'Orbitron', sans-serif; color: #f8fafc; font-size: 1.1rem; margin-bottom: 16px; border-left: 3px solid #818cf8; padding-left: 8px;">
            Multi-Agent Pipeline Sequencing
        </h4>
        
        <div style="display: flex; flex-direction: column; gap: 12px; margin-bottom: 24px;">
            <div class="kpi-card" style="padding: 16px; background: rgba(30, 41, 59, 0.15);">
                <strong style="color: #38bdf8; font-family: 'Orbitron', sans-serif; font-size: 0.9rem;"><i class="fa-solid fa-thermometer-half aura-icon"></i> 1. Machine Health Agent</strong>
                <p style="margin: 6px 0 0 0; font-size: 0.85rem; color: #94a3b8; line-height: 1.4;">Evaluates temperature, vibration, and pressure readings for threshold violations and flags sensor anomalies.</p>
            </div>
            <div class="kpi-card" style="padding: 16px; background: rgba(30, 41, 59, 0.15);">
                <strong style="color: #a855f7; font-family: 'Orbitron', sans-serif; font-size: 0.9rem;"><i class="fa-solid fa-brain aura-icon"></i> 2. Failure Prediction Agent</strong>
                <p style="margin: 6px 0 0 0; font-size: 0.85rem; color: #94a3b8; line-height: 1.4;">Trains a dynamic Random Forest Classifier on telemetry data to predict failure probability and calculate feature importance.</p>
            </div>
            <div class="kpi-card" style="padding: 16px; background: rgba(30, 41, 59, 0.15);">
                <strong style="color: #fbbf24; font-family: 'Orbitron', sans-serif; font-size: 0.9rem;"><i class="fa-solid fa-screwdriver-wrench aura-icon"></i> 3. Maintenance Agent</strong>
                <p style="margin: 6px 0 0 0; font-size: 0.85rem; color: #94a3b8; line-height: 1.4;">Ranks machine maintenance urgency and compiles a prioritized scheduler queue based on active risk profiles.</p>
            </div>
            <div class="kpi-card" style="padding: 16px; background: rgba(30, 41, 59, 0.15);">
                <strong style="color: #10b981; font-family: 'Orbitron', sans-serif; font-size: 0.9rem;"><i class="fa-solid fa-bolt aura-icon"></i> 4. Production Optimization Agent</strong>
                <p style="margin: 6px 0 0 0; font-size: 0.85rem; color: #94a3b8; line-height: 1.4;">Maps real-time power consumption to load percentages and devises active energy-load balancing configurations.</p>
            </div>
            <div class="kpi-card" style="padding: 16px; background: rgba(30, 41, 59, 0.15);">
                <strong style="color: #818cf8; font-family: 'Orbitron', sans-serif; font-size: 0.9rem;"><i class="fa-solid fa-robot aura-icon"></i> 5. GenAI Report Agent</strong>
                <p style="margin: 6px 0 0 0; font-size: 0.85rem; color: #94a3b8; line-height: 1.4;">Synthesizes floor parameters and ML results to generate a professional executive report via Groq LLM API.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.info("👈 **Click the 'Analyze Telemetry Data' button in the sidebar** to start the cognitive processing pipeline.")
        
    else:
        st.markdown("""
        <div class="kpi-card" style="margin-bottom: 24px;">
            <h3 style="font-family: 'Orbitron', sans-serif; color: #38bdf8; margin-top: 0; display: flex; align-items: center; gap: 8px;">
                <i class="fa-solid fa-industry aura-icon"></i> FACTORY AI Smart Manufacturing AI Platform
            </h3>
            <p style="font-size: 0.95rem; margin-top: 8px; color: #cbd5e1; line-height: 1.5;">
                Welcome to FACTORY AI. To initialize the multi-agent cognitive telemetry analysis pipeline, please upload your factory CSV dataset in the sidebar.
            </p>
            <div style="background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.2); padding: 8px 16px; border-radius: 8px; color: #38bdf8; font-weight: 500; display: inline-flex; align-items: center; gap: 8px; margin-top: 8px; font-size: 0.85rem;">
                <i class="fa-solid fa-info-circle"></i> Awaiting CSV Dataset Upload
            </div>
        </div>
        
        <h4 style="font-family: 'Orbitron', sans-serif; color: #f8fafc; font-size: 1.1rem; margin-bottom: 16px; border-left: 3px solid #818cf8; padding-left: 8px;">
            CSV Schema Guidelines
        </h4>
        <p style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 16px;">
            The ingestion agent validates the dataset structure. Below is the list of expected telemetry attributes:
        </p>
        """, unsafe_allow_html=True)

        schema_cols = [
            ("Machine_ID", "String unique identifier of the machine/node"),
            ("Machine_Name", "Operational name/model type of the hardware"),
            ("Start_Time / End_Time", "Timestamps of operational logging interval"),
            ("Operating_Hours", "Total cumulative runtime hours of the machine"),
            ("Temperature_C", "Ambient and engine thermal sensor reading (Celsius)"),
            ("Vibration_mm_s", "Rotational housing vibration peak-to-peak amplitude (mm/s)"),
            ("Pressure_bar", "Hydraulic core or pneumatic line pressure (bar)"),
            ("Power_Consumption_kW", "Current electrical grid draw rate (kW)"),
            ("Load_Percentage", "Current capacity usage vs peak rating (0-100%)"),
            ("Oil_Level_Percentage", "Internal lubrication fluids volume check (0-100%)"),
            ("Humidity_Percentage", "Ambient humidity levels inside the workshop"),
            ("RPM", "Rotations per minute speed of the main shaft"),
            ("Remaining_Useful_Life_Days", "Estimated remaining days before catastrophic component damage"),
            ("Machine_Health", "Manual health inspection status label ('Good', 'Fair', 'Critical')"),
            ("Fault_Status", "Target binary marker representing active failures or damage (0=Normal, 1=Faulted)"),
            ("Maintenance_Required", "Target binary marker of scheduled check requirement (0=No, 1=Yes)")
        ]

        cols_df = pd.DataFrame(schema_cols, columns=["Attribute Column", "Operational Description"])
        st.dataframe(cols_df, use_container_width=True, hide_index=True)
        st.info("👈 **Upload your CSV file in the sidebar** to begin dynamic modeling and multi-agent report creation.")

else:
    coord = st.session_state.coordinator

    # ============================================================
    # SIDEBAR NAVIGATION ROUTING
    # ============================================================
    if active_tab == "📊 Factory Control Center":
        st.markdown(
            '<div class="section-title">'
            '<i class="fa-solid fa-industry aura-icon c-info"></i>'
            'Factory Control Center (Overview)'
            '</div>',
            unsafe_allow_html=True
        )

        # KPI Row
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

        total_m    = len(coord.health_results['machine_summary'])
        avg_health = coord.health_results['machine_summary']['Health_Score'].mean()
        tot_anom   = coord.health_results['total_anomalies']
        critical_m = coord.health_results['critical_count']

        with kpi_col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Monitored Assets</div>
                <div class="kpi-value">{total_m}</div>
                <div class="kpi-trend" style="color: #38bdf8;">
                    <i class="fa-solid fa-circle-check aura-icon c-inherit"></i> Active Node Network
                </div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_col2:
            health_color = "#10b981" if avg_health > 75 else ("#f59e0b" if avg_health > 50 else "#ef4444")
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Mean Health Index</div>
                <div class="kpi-value" style="color: {health_color};">{avg_health:.1f}%</div>
                <div class="kpi-trend" style="color: {health_color};">
                    <i class="fa-solid fa-heart-pulse aura-icon c-inherit"></i> Dynamic Floor Score
                </div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_col3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Sensor Anomalies</div>
                <div class="kpi-value" style="color: #fbbf24;">{tot_anom}</div>
                <div class="kpi-trend" style="color: #fbbf24;">
                    <i class="fa-solid fa-triangle-exclamation aura-icon c-inherit"></i> Aggregated Outliers
                </div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_col4:
            crit_color = "#ef4444" if critical_m > 0 else "#10b981"
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Critical Level Risks</div>
                <div class="kpi-value" style="color: {crit_color};">{critical_m}</div>
                <div class="kpi-trend" style="color: {crit_color};">
                    <i class="fa-solid fa-circle-exclamation aura-icon c-inherit"></i> Requires Immediate Action
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts Row (Gauge, Pie, Trend Selection)
        chart_col1, chart_col2, chart_col3 = st.columns([1.1, 0.9, 1.4])

        with chart_col1:
            st.markdown("##### Mean Fleet Health Index")
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = avg_health,
                domain = {'x': [0, 1], 'y': [0, 1]},
                number = {'suffix': "%", 'font': {'size': 32, 'family': 'Orbitron', 'color': health_color}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                    'bar': {'color': health_color},
                    'bgcolor': "rgba(21, 28, 44, 0.4)",
                    'borderwidth': 1,
                    'bordercolor': "rgba(255, 255, 255, 0.05)",
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.1)'},
                        {'range': [50, 75], 'color': 'rgba(245, 158, 11, 0.1)'},
                        {'range': [75, 100], 'color': 'rgba(16, 185, 129, 0.1)'}
                    ],
                    'threshold': {
                        'line': {'color': "#38bdf8", 'width': 3},
                        'thickness': 0.75,
                        'value': avg_health
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#f8fafc"),
                margin=dict(l=10, r=10, t=10, b=10),
                height=260
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with chart_col2:
            st.markdown("##### Health Categories")
            summary_df = coord.health_results['machine_summary']
            fig_pie = px.pie(
                summary_df,
                names='Health_Category',
                values='Health_Score',
                color='Health_Category',
                color_discrete_map={'Good': '#10b981', 'Fair': '#f59e0b', 'Critical': '#ef4444'},
                hole=0.55
            )
            fig_pie.update_traces(
                textposition='inside',
                textinfo='percent+label',
                marker=dict(line=dict(color='#0b0f19', width=2)),
                hoverinfo='label+percent',
                showlegend=False
            )
            fig_pie.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#cbd5e1"),
                margin=dict(l=10, r=10, t=10, b=10),
                height=260
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with chart_col3:
            st.markdown("##### Real-Time Sensor Telemetry Trends")
            sensor_opt = st.selectbox(
                "Choose Sensor Variable for Aggregated Historical Trend",
                ["Temperature_C", "Vibration_mm_s", "Pressure_bar", "RPM", "Power_Consumption_kW"],
                key="overview_sensor_select",
                label_visibility="collapsed"
            )
            
            fig_trend = px.line(
                coord.df.sort_values(by='Start_Time'),
                x='Start_Time',
                y=sensor_opt,
                color='Machine_Name',
                labels={'Start_Time': 'Timestamp', sensor_opt: sensor_opt.replace('_', ' ')}
            )
            fig_trend.update_traces(line=dict(width=2))
            fig_trend.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#cbd5e1"),
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(
                    gridcolor="rgba(255,255,255,0.05)",
                    linecolor="rgba(255,255,255,0.1)",
                ),
                yaxis=dict(
                    gridcolor="rgba(255,255,255,0.05)",
                    linecolor="rgba(255,255,255,0.1)",
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(size=9, color="#94a3b8")
                ),
                height=260
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        # Assets Table Overview
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<h5><i class="fa-solid fa-desktop aura-icon c-info"></i>Digital Twin Fleet Summary</h5>',
            unsafe_allow_html=True
        )
        disp_df = summary_df[['Machine_ID', 'Machine_Name', 'Health_Score', 'Health_Category',
                               'Avg_Temperature_C', 'Avg_Vibration_mm_s', 'Avg_RUL_Days', 'Total_Anomalies']].copy()
        
        disp_df.columns = [col.replace('_', ' ') for col in disp_df.columns]
        
        st.dataframe(
            disp_df.style
                .background_gradient(subset=['Health Score'], cmap='RdYlGn', vmin=0, vmax=100)
                .format({
                    'Health Score': '{:.1f}%',
                    'Avg Temperature C': '{:.1f}°C',
                    'Avg Vibration mm s': '{:.2f} mm/s',
                    'Avg RUL Days': '{:.0f} days',
                    'Total Anomalies': '{:d}'
                }),
            use_container_width=True,
            hide_index=True
        )

    elif active_tab == "🩺 Asset Diagnostics":
        st.markdown(
            '<div class="section-title">'
            '<i class="fa-solid fa-thermometer-half aura-icon c-info"></i>'
            'Asset Diagnostics (Machine Health)'
            '</div>',
            unsafe_allow_html=True
        )

        selected_m = st.selectbox(
            "Select Machine Asset to Inspect Details",
            coord.df['Machine_Name'].unique(),
            key="health_machine_select"
        )
        m_df = coord.df[coord.df['Machine_Name'] == selected_m].sort_values(by='Start_Time')
        h_row = coord.health_results['machine_summary'][
            coord.health_results['machine_summary']['Machine_Name'] == selected_m
        ].iloc[0]

        st.markdown(f"### Diagnostics Drill-down: <span style='font-family: \"Orbitron\", sans-serif; color: #38bdf8;'>{selected_m}</span>", unsafe_allow_html=True)

        det_col1, det_col2, det_col3, det_col4 = st.columns(4)
        det_col1.metric("Calculated Health Score", f"{h_row['Health_Score']:.1f}%", delta=f"{h_row['Health_Category']}")
        det_col2.metric("Total Flagged Outliers", h_row['Total_Anomalies'])
        det_col3.metric("Operating Hours", f"{m_df['Operating_Hours'].max():.0f} Hrs")
        det_col4.metric("Avg RUL Days Remaining", f"{h_row['Avg_RUL_Days']:.0f} Days")

        st.markdown("---")

        st.markdown("##### Telemetry Sensor Time-series Comparison")
        fig_multi = go.Figure()
        fig_multi.add_trace(go.Scatter(x=m_df['Start_Time'], y=m_df['Temperature_C'],       name="Temperature (°C)",     line=dict(color="#ff4b4b", width=2.5, shape='spline')))
        fig_multi.add_trace(go.Scatter(x=m_df['Start_Time'], y=m_df['Vibration_mm_s'] * 10, name="Vibration x10 (mm/s)", line=dict(color="#00a2ff", width=2.5, shape='spline')))
        fig_multi.add_trace(go.Scatter(x=m_df['Start_Time'], y=m_df['Pressure_bar'],        name="Pressure (bar)",       line=dict(color="#00e676", width=2.5, shape='spline')))
        fig_multi.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#94a3b8"),
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=10, color="#f8fafc")
            ),
            height=380
        )
        st.plotly_chart(fig_multi, use_container_width=True)

        # Anomaly highlights table
        anom_data   = coord.health_results['df_with_anomalies']
        machine_anom = anom_data[
            (anom_data['Machine_Name'] == selected_m) & (anom_data['Anomaly_Count'] > 0)
        ].sort_values(by='Start_Time', ascending=False)

        st.markdown(f"##### Recent Sensor Violations for {selected_m} (Out of Normal Bounds)")
        if not machine_anom.empty:
            disp_anom = machine_anom[['Start_Time', 'Temperature_C', 'Vibration_mm_s',
                                      'Pressure_bar', 'Oil_Level_Percentage', 'Anomaly_Count']].copy()
            disp_anom.columns = [c.replace('_', ' ') for c in disp_anom.columns]
            st.dataframe(
                disp_anom.style.format({
                    'Temperature C': '{:.1f}°C',
                    'Vibration mm s': '{:.2f} mm/s',
                    'Pressure bar': '{:.1f} bar',
                    'Oil Level Percentage': '{:.1f}%',
                    'Anomaly Count': '{:d}'
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success("No sensor threshold violations detected for this machine asset during this log sequence.")

    elif active_tab == "🧠 Predictive Reliability":
        st.markdown(
            '<div class="section-title">'
            '<i class="fa-solid fa-brain aura-icon c-info"></i>'
            'Predictive Reliability (ML classifier)'
            '</div>',
            unsafe_allow_html=True
        )

        metrics = coord.prediction_results['metrics']

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Model Training Accuracy",  f"{metrics['Accuracy']:.2%}")
        col_m2.metric("Precision (Reliability)",  f"{metrics['Precision']:.2%}")
        col_m3.metric("Recall (Sensitivity)",     f"{metrics['Recall']:.2%}")
        col_m4.metric("F1 Performance Score",     f"{metrics['F1_Score']:.2%}")

        st.markdown("---")

        c_ml1, c_ml2 = st.columns([1, 1])

        with c_ml1:
            st.markdown("##### Sensor Predictive Feature Importance")
            feat_imp_df = coord.prediction_results['feature_importances']
            
            fig_bar = px.bar(
                feat_imp_df,
                x='Importance',
                y='Feature',
                orientation='h',
                color='Importance',
                color_continuous_scale=[[0, '#818cf8'], [1, '#a855f7']]
            )
            fig_bar.update_traces(
                marker=dict(line=dict(color='rgba(0,0,0,0)', width=0)),
                hoverinfo='x+y'
            )
            fig_bar.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#94a3b8"),
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
                yaxis=dict(gridcolor="rgba(0,0,0,0)", linecolor="rgba(255,255,255,0.1)", categoryorder='total ascending'),
                coloraxis_showscale=False,
                height=320
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with c_ml2:
            st.markdown(
                '<h5 style="margin-top:0;">'
                '<i class="fa-solid fa-sliders aura-icon c-info"></i>'
                'Simulator: Custom Telemetry Inputs'
                '</h5>',
                unsafe_allow_html=True
            )
            st.write("Modify telemetry settings to observe the Random Forest real-time failure model predictions:")

            sim_temp = st.slider("Temperature (°C)",    min_value=10.0,  max_value=150.0, value=65.0,  key="sim_temp_slider")
            sim_vib  = st.slider("Vibration (mm/s)",    min_value=0.0,   max_value=15.0,  value=2.5,   key="sim_vib_slider")
            sim_pres = st.slider("Pressure (bar)",      min_value=0.0,   max_value=200.0, value=90.0,  key="sim_pres_slider")
            sim_oil  = st.slider("Lubricant Level (%)", min_value=0.0,   max_value=100.0, value=75.0,  key="sim_oil_slider")

            input_feats = {
                'Operating_Hours':      5000,
                'Temperature_C':        sim_temp,
                'Vibration_mm_s':       sim_vib,
                'Pressure_bar':         sim_pres,
                'Power_Consumption_kW': 45.0,
                'Load_Percentage':      75.0,
                'Oil_Level_Percentage': sim_oil,
                'Humidity_Percentage':  55.0,
                'RPM':                  1800.0
            }

            pred_out   = coord.failure_agent.predict_custom(input_feats)
            prob_color = "#ef4444" if pred_out['Failure_Probability'] > 50 else "#10b981"
            st.markdown(f"""
            <div style="background: rgba(21, 28, 44, 0.8); padding: 16px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.06); margin-top: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
                <h6 style="margin: 0 0 8px 0; color: #a855f7; font-family: 'Orbitron', sans-serif; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">
                    <i class="fa-solid fa-robot aura-icon"></i> SIMULATOR DIAGNOSIS STATUS
                </h6>
                <h4 style="margin: 5px 0 5px 0; color: {prob_color}; font-family: 'Orbitron', sans-serif; font-weight: 700; font-size: 1.3rem;">{pred_out['Prediction']}</h4>
                <p style="margin: 0; font-size: 0.85rem; color: #cbd5e1;">Predicted Failure Risk: <strong style="color:{prob_color}; font-family:'Orbitron',sans-serif;">{pred_out['Failure_Probability']}%</strong></p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        
        cm_col1, cm_col2 = st.columns([1, 1.2])
        
        with cm_col1:
            st.markdown("##### Confusion Matrix (Validation Split)")
            cm_data = metrics['Confusion_Matrix']
            cm_df   = pd.DataFrame(cm_data, index=["Actual Normal", "Actual Fault"], columns=["Predicted Normal", "Predicted Fault"])
            st.dataframe(
                cm_df.style
                    .background_gradient(cmap='Purples')
                    .format('{:d}'),
                use_container_width=True
            )
            
        with cm_col2:
            st.markdown("##### Sensor Cross-Correlation Heatmap")
            numeric_cols = [
                'Temperature_C', 'Vibration_mm_s', 'Pressure_bar', 
                'Power_Consumption_kW', 'Load_Percentage', 'Oil_Level_Percentage', 
                'RPM', 'Remaining_Useful_Life_Days'
            ]
            corr_matrix = coord.df[numeric_cols].corr()
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=[c.replace('_', ' ') for c in corr_matrix.columns],
                y=[c.replace('_', ' ') for c in corr_matrix.index],
                colorscale='electric',
                zmin=-1, zmax=1
            ))
            fig_corr.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#94a3b8"),
                margin=dict(l=20, r=20, t=10, b=20),
                height=260
            )
            st.plotly_chart(fig_corr, use_container_width=True)



    elif active_tab == "🔧 Maintenance Scheduler":
        st.markdown(
            '<div class="section-title">'
            '<i class="fa-solid fa-screwdriver-wrench aura-icon c-info"></i>'
            'Maintenance Scheduler'
            '</div>',
            unsafe_allow_html=True
        )
        m_queue    = coord.maintenance_results['maintenance_queue']
        crit_maint = coord.maintenance_results['critical_maintenance_count']
        high_maint = coord.maintenance_results['high_maintenance_count']

        col_sch1, col_sch2 = st.columns(2)
        if crit_maint > 0:
            col_sch1.warning(f"⚠️ **{crit_maint}** Machine Asset(s) require **Immediate Critical Maintenance**.")
        else:
            col_sch1.success("✅ Zero immediate critical-level repairs scheduled on the floor.")

        if high_maint > 0:
            col_sch2.info(f"ℹ️ **{high_maint}** Machine Asset(s) listed on **High Priority Action Queue**.")

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            '<h5><i class="fa-solid fa-calendar-days aura-icon c-info"></i>Asset Action Registry (Sorted by Urgency Index)</h5>',
            unsafe_allow_html=True
        )
        styled_queue = m_queue.style.background_gradient(subset=['Urgency_Score'], cmap='OrRd', vmin=0, vmax=100)
        st.dataframe(styled_queue, use_container_width=True, hide_index=True)

        # Download work orders card
        st.markdown("""
        <div class="kpi-card" style="margin-top: 24px; border-left: 4px solid #38bdf8;">
            <h5 style="margin: 0 0 8px 0; color: #38bdf8; font-family: 'Orbitron', sans-serif;"><i class="fa-solid fa-file-csv aura-icon"></i> Export Work Orders</h5>
            <p style="margin: 0 0 16px 0; font-size: 0.85rem; color: #cbd5e1;">Generate and download a CSV registry of all pending maintenance tickets sorted by urgency index for shop floor technicians.</p>
        </div>
        """, unsafe_allow_html=True)
        
        csv_data = m_queue.to_csv(index=False).encode('utf-8')
        
        # Write copy directly to local workspace path
        local_csv_filename = "Machine_Health_Analysis.csv"
        with open(local_csv_filename, "wb") as f_out:
            f_out.write(csv_data)

        # Export copy to Streamlit static path for guaranteed same-origin download in webviews
        try:
            streamlit_static_dir = os.path.join(os.path.dirname(st.__file__), "static")
            static_csv_path = os.path.join(streamlit_static_dir, "Machine_Health_Analysis.csv")
            with open(static_csv_path, "wb") as f_static:
                f_static.write(csv_data)
        except Exception:
            pass

        st.success("💾 **Work orders registry generated!**")

        st.download_button(
            label="Download Scheduled Work Orders (CSV)",
            data=csv_data,
            file_name="Machine_Health_Analysis.csv",
            mime="text/csv",
            key="maintenance_download_btn"
        )

    elif active_tab == "⚡ Grid & Power Optimization":
        st.markdown(
            '<div class="section-title">'
            '<i class="fa-solid fa-bolt aura-icon c-info"></i>'
            'Grid & Power Optimization'
            '</div>',
            unsafe_allow_html=True
        )

        opt_res = coord.optimization_results

        o_col1, o_col2 = st.columns(2)
        o_col1.metric("Mean Energy Efficiency Index (Load / kW)", f"{opt_res['avg_factory_efficiency']:.2f}")

        with o_col2:
            balancing_actions_df = opt_res['balancing_actions']
            st.metric("Suggested Operational Load-Transfers", len(balancing_actions_df))

        st.markdown("---")

        st.markdown("##### Load Capacity vs Power Consumption Efficiency")
        stats_df = opt_res['machine_stats']
        
        fig_scatter = px.scatter(
            stats_df,
            x='Avg_Load_Pct',
            y='Avg_Power_kW',
            size='Total_Hours',
            color='Avg_Load_Efficiency',
            hover_name='Machine_Name',
            color_continuous_scale=[[0, '#00a2ff'], [0.5, '#00d2ff'], [1, '#00e676']],
            size_max=35
        )
        fig_scatter.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#94a3b8"),
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)", title="Average Load (%)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)", title="Average Power (kW)"),
            coloraxis_colorbar=dict(
                title="Efficiency",
                title_font=dict(size=10, color="#f8fafc"),
                tickfont=dict(size=9, color="#94a3b8")
            ),
            height=340
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        col_i1, col_i2 = st.columns([3, 2])

        with col_i1:
            st.markdown(
                '<h5><i class="fa-solid fa-plug aura-icon c-info"></i>Energy-degradation Optimization Status</h5>',
                unsafe_allow_html=True
            )
            opt_insights = opt_res['opt_insights'].copy()
            opt_insights.columns = [c.replace('_', ' ') for c in opt_insights.columns]
            st.dataframe(opt_insights, use_container_width=True, hide_index=True)

        with col_i2:
            st.markdown(
                '<h5><i class="fa-solid fa-scale-balanced aura-icon c-info"></i>Suggested Grid Load Balancing</h5>',
                unsafe_allow_html=True
            )
            if not balancing_actions_df.empty:
                for idx, row in balancing_actions_df.iterrows():
                    st.markdown(f"""
                    <div class="kpi-card" style="margin-bottom: 12px; border-left: 4px solid #38bdf8; background: rgba(30, 41, 59, 0.1);">
                        <strong style="color: #38bdf8; font-family: 'Orbitron', sans-serif;"><i class="fa-solid fa-rotate aura-icon"></i> {row['Action']}</strong>
                        <p style="margin: 6px 0 0 0; font-size: 0.85rem; color: #cbd5e1; line-height: 1.4;">{row['Reason']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("Power distribution balance is within nominal bounds. No routing transfers required.")

    elif active_tab == "📋 Executive AI Report":
        st.markdown(
            '<div class="section-title">'
            '<i class="fa-solid fa-robot aura-icon c-info"></i>'
            'Executive Operations Report'
            '</div>',
            unsafe_allow_html=True
        )

        if st.button(
            "Generate / Synthesize Operations Report",
            help="Sends telemetry summaries and ML metrics to Groq to generate a professional overview report.",
            key="generate_report_button",
            use_container_width=True
        ):
            with st.spinner("Synthesizing telemetry profiles and building smart recommendations with Groq Agent..."):
                report_content = coord.generate_ai_report()
                st.session_state.current_ai_report = report_content

        if 'current_ai_report' in st.session_state:
            st.markdown("---")
            
            rep_col1, rep_col2 = st.columns([1.2, 1])
            
            with rep_col1:
                st.markdown("##### Preview Executive Report")
                st.markdown(f"""
                <div style="background: rgba(21, 28, 44, 0.4); border: 1px solid rgba(255, 255, 255, 0.05); padding: 24px; border-radius: 12px; max-height: 600px; overflow-y: auto; color: #cbd5e1;">
                    {st.session_state.current_ai_report}
                </div>
                """, unsafe_allow_html=True)
                
            with rep_col2:
                st.markdown("##### Interactive Report Editor")
                updated_report = st.text_area(
                    "Interactive Report Editor",
                    value=st.session_state.current_ai_report,
                    height=350,
                    key="ai_report_text_area",
                    label_visibility="collapsed"
                )
                st.session_state.current_ai_report = updated_report

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("##### Download compiled Report")

                try:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        charts_folder = None
                        try:
                            if generate_matplotlib_charts(coord, temp_dir):
                                charts_folder = temp_dir
                        except Exception as chart_err:
                            st.warning(f"Note: Charts couldn't be embedded in PDF: {str(chart_err)}")

                        pdf_bytes = generate_pdf_report(
                            coord,
                            st.session_state.current_ai_report,
                            charts_dir=charts_folder
                        )

                    st.success("💾 **PDF Report compiled successfully!**")

                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_bytes,
                        file_name="Smart_Manufacturing_Executive_Report.pdf",
                        mime="application/pdf",
                        key="download_pdf_report_btn",
                        use_container_width=True
                    )
                except Exception as pdf_err:
                    st.error(f"Error compiling PDF Report: {str(pdf_err)}")

            # On-screen Financial Exposure Visualization Section
            st.markdown("<br><hr style='border-color: rgba(255,255,255,0.06);'><br>", unsafe_allow_html=True)
            st.markdown("### 📊 Predicted Machine Failures & Financial Loss Analysis")
            st.markdown(
                "This section evaluates plant-floor financial exposure based on predictive failure models. "
                "Estimated downtime, repair costs, and production loss rates are calculated dynamically from power usage, load profiles, "
                "and mechanical health indices."
            )

            import importlib
            import manufacturing_agents.report_agent
            importlib.reload(manufacturing_agents.report_agent)
            from manufacturing_agents.report_agent import compute_financial_loss_data
            records, summary = compute_financial_loss_data(coord)
            loss_df = pd.DataFrame(records)

            # Management Summary Metrics Row
            sum_col1, sum_col2, sum_col3, sum_col4, sum_col5 = st.columns(5)
            sum_col1.metric("Machines at Risk", f"{summary['Total_Machines_at_Risk']} Assets")
            sum_col2.metric("Highest-Risk Asset", summary['Highest_Risk_Machine'].split(" (")[0])
            sum_col3.metric("Est. Downtime", f"{summary['Total_Estimated_Downtime']} Hrs")
            sum_col4.metric("Est. Financial Loss", f"₹{summary['Total_Estimated_Loss']:,}")
            sum_col5.metric("Potential Savings", f"₹{summary['Potential_Savings']:,}")

            # Loss Exposure Bar Chart
            fig_loss = px.bar(
                loss_df,
                x='Machine_Name',
                y='Total_Estimated_Loss',
                color='Risk_Level',
                title="Estimated Financial Exposure per Asset (₹)",
                labels={'Total_Estimated_Loss': 'Total Loss (₹)', 'Machine_Name': 'Machine'},
                color_discrete_map={'Critical': '#ef4444', 'High': '#f97316', 'Medium': '#eab308', 'Low': '#10b981'},
                category_orders={'Risk_Level': ['Critical', 'High', 'Medium', 'Low']}
            )
            fig_loss.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#94a3b8"),
                margin=dict(l=20, r=20, t=40, b=20),
                height=320,
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Machine Name"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Total Estimated Loss (₹)")
            )
            st.plotly_chart(fig_loss, use_container_width=True)

            # Failure Risk DataFrame Table
            st.markdown("##### Detailed Risk & Financial Exposure Register")

            display_df = loss_df.copy()
            display_df['Estimated_Downtime'] = display_df['Estimated_Downtime'].apply(lambda x: f"{x:.1f} Hrs")
            display_df['Estimated_Repair_Cost'] = display_df['Estimated_Repair_Cost'].apply(lambda x: f"₹{x:,.0f}")
            display_df['Estimated_Production_Loss'] = display_df['Estimated_Production_Loss'].apply(lambda x: f"₹{x:,.0f}")
            display_df['Total_Estimated_Loss'] = display_df['Total_Estimated_Loss'].apply(lambda x: f"₹{x:,.0f}")

            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("Click the **Generate / Synthesize Operations Report** button above to generate a smart AI-powered executive report using Groq.")
