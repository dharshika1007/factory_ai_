# Smart Manufacturing AI Dashboard

A professional, hackathon-ready **Agentic AI Smart Manufacturing System** built with Python and Streamlit. This application integrates machine learning (Random Forest) for predictive maintenance and an LLM-based agentic system (powered by Groq) to deliver actionable operational recommendations.

## Tech Stack
- **Dashboard & UI**: Streamlit, HTML/CSS
- **Data Analytics**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (Random Forest Classifier)
- **Visualizations**: Plotly
- **GenAI / Agents**: Groq API (Llama-3 model)
- **Config**: Python-dotenv

---

## Folder Structure
```text
SmartManufacturingAI/
│── app.py
│── .env
│── requirements.txt
│── README.md
```
*Note: As per project constraints, all code logic, machine learning pipelines, and AI agent architectures are defined solely within `app.py`.*

---

## Agent Architectures
1. **CoordinatorAgent**: Clean, preprocess, and route data inputs across the system.
2. **MachineHealthAgent**: Perform telemetry checks, calculate machine health metrics, and identify anomalies.
3. **FailurePredictionAgent**: Train a Random Forest model dynamically on uploaded data to predict critical failures and highlight feature importance.
4. **MaintenanceAgent**: Generate preventive maintenance schedules based on telemetry thresholds and failure prediction models.
5. **ProductionOptimizationAgent**: Calculate power efficiency, machine load distributions, and load balancing actions.
6. **ReportAgent**: Query the Groq API to compile comprehensive executive summaries and operations reports.

---

## Expected Dataset Columns (CSV)
The system expects a CSV file containing telemetry and status data for machines. The CSV file must contain the following columns:
*   `Machine_ID`: Unique identifier for the machine
*   `Machine_Name`: Name or model of the machine
*   `Start_Time`: Operation starting timestamp
*   `End_Time`: Operation ending timestamp
*   `Operating_Hours`: Cumulative run hours
*   `Temperature_C`: Internal temperature in Celsius
*   `Vibration_mm_s`: Mechanical vibration speed
*   `Pressure_bar`: Hydraulic/pneumatic pressure
*   `Power_Consumption_kW`: Current energy usage
*   `Load_Percentage`: Operational capacity percentage
*   `Oil_Level_Percentage`: Lubricant levels
*   `Humidity_Percentage`: Ambient humidity levels
*   `RPM`: Rotational speed (Revolutions Per Minute)
*   `Remaining_Useful_Life_Days`: Predicted days before structural component failure
*   `Machine_Health`: Operational rating (e.g. Good, Fair, Critical)
*   `Fault_Status`: Indicator of a fault event (e.g. 0 for normal, 1 for fault, or string labels)
*   `Maintenance_Required`: Flag indicating scheduled/needed checkup (0/1 or Yes/No)

---

## Installation & Running

1. Clone or copy files into your local directory.
2. Install Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your Groq API Key in `.env`:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```
4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
5. Open your browser to `http://localhost:8501`, upload your manufacturing telemetry CSV, and begin analyzing.
