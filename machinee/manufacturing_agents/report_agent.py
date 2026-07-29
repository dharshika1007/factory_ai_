"""
agents/report_agent.py
AURA Smart Manufacturing AI

Contains:
  - ReportAgent: Groq LLM integration for executive report generation
  - NumberedCanvas: ReportLab canvas with automatic page headers/footers
  - generate_matplotlib_charts(): Render operational charts as PNG images
  - generate_pdf_report(): Assemble and compile the full professional PDF report
"""

import os
import io
import re
import json
import datetime

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from groq import Groq

# ReportLab PDF imports
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas


# ---------------------------------------------------------------------------
# ReportAgent
# ---------------------------------------------------------------------------

class ReportAgent:
    """
    Coordinates with Groq LLM API to assemble dynamic executive summaries.
    Falls back gracefully if Groq API key is missing or fails.
    """
    def __init__(self, coordinator, api_key: str = None):
        self.coordinator = coordinator
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = None
        if self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception:
                self.client = None

    def generate_report(
        self,
        health_results: dict,
        prediction_results: dict,
        maintenance_results: dict,
        optimization_results: dict
    ) -> str:
        # Prepare context data block for the LLM
        total_machines = len(health_results['machine_summary'])
        crit_machines = health_results['critical_count']
        fair_machines = health_results['fair_count']
        good_machines = health_results['good_count']
        total_anoms = health_results['total_anomalies']

        rf_metrics = prediction_results['metrics']
        feat_imp = prediction_results['feature_importances'].head(4).to_dict(orient='records')

        maint_queue = maintenance_results['maintenance_queue'].head(5).to_dict(orient='records')
        opt_recs = optimization_results['opt_insights'].to_dict(orient='records')
        avg_efficiency = optimization_results['avg_factory_efficiency']

        context_data = {
            "factory_summary": {
                "total_machines": total_machines,
                "health_distribution": {
                    "Critical": crit_machines,
                    "Fair": fair_machines,
                    "Good": good_machines
                },
                "total_telemetry_anomalies_detected": total_anoms,
                "average_energy_load_efficiency": avg_efficiency
            },
            "failure_prediction_ml_model": {
                "classifier_name": "Random Forest Classifier",
                "accuracy": f"{rf_metrics.get('Accuracy', 0):.2%}",
                "precision": f"{rf_metrics.get('Precision', 0):.2%}",
                "recall": f"{rf_metrics.get('Recall', 0):.2%}",
                "f1_score": f"{rf_metrics.get('F1_Score', 0):.2%}",
                "top_predictive_features": feat_imp
            },
            "urgent_maintenance_schedule": maint_queue,
            "production_optimization_opportunities": opt_recs[:5]
        }

        prompt = f"""
You are an advanced Lead Manufacturing AI & Reliability Engineer.
Analyze the following factory telemetry analysis summary JSON and write a detailed, highly professional, executive operations report.

```json
{json.dumps(context_data, indent=2)}
```

Write a structured report containing:
1. **EXECUTIVE OVERVIEW**: High-level status of the factory operations, including health breakdown and key risk factors.
2. **PREDICTIVE MAINTENANCE & FAILURE RISKS**: Detailed discussion of the ML model results (accuracy, top features) and what they mean for the floor. Specifically comment on the top urgent machines and their failure probabilities.
3. **PRODUCTION & ENERGY EFFICIENCY ANALYSIS**: Highlight energy waste, load-balancing recommendations, and scheduling optimizations.
4. **ACTIONABLE RECOMMENDATIONS CHECKLIST**: A bulleted roadmap for the shifts/maintenance crews to address immediate failure risks first, followed by medium-term efficiency projects.

Write the report in clear, authoritative, and descriptive markdown. Include styling elements, quotes, and clean tables where useful. Do not include references to XML tags or code blocks outside the report itself.
"""

        # Call Groq API if client is available
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "You are a smart manufacturing AI assistant. Provide professional, deep, and context-aware reports for plant operators and managers."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=2500
                )
                return response.choices[0].message.content
            except Exception as e:
                return self._generate_rules_fallback(context_data, error_msg=str(e))
        else:
            return self._generate_rules_fallback(context_data, error_msg="Groq API key not provided or client not initialized.")

    def _generate_rules_fallback(self, context: dict, error_msg: str) -> str:
        # Fallback local rules-based markdown report generation
        fs = context['factory_summary']
        ml = context['failure_prediction_ml_model']
        queue = context['urgent_maintenance_schedule']
        opt = context['production_optimization_opportunities']

        maint_list = ""
        for item in queue:
            maint_list += f"- **{item['Machine_Name']} (ID: {item['Machine_ID']})**: Urgency Score {item['Urgency_Score']}/100 - *Priority: {item['Priority']}*. Actions: {item['Recommended_Actions']}\n"

        opt_list = ""
        for item in opt:
            opt_list += f"- **{item['Machine_Name']}**: Status: `{item['Status_Tag']}`. Suggestion: {item['Actionable_Insight']}\n"

        feat_list = ""
        for feat in ml['top_predictive_features']:
            feat_list += f"- {feat['Feature']}: {feat['Importance']:.2%}\n"

        fallback_report = f"""# FACTORY OPERATIONAL AI SUMMARY REPORT
*(Note: Generated via rules-based compiler. Groq API connection status: {error_msg})*

## Executive Overview
Our digital twin telemetry pipeline has analyzed **{fs['total_machines']} active machines**. The overall status of the factory operations is categorized as follows:
- **Good Condition**: {fs['health_distribution']['Good']} machines
- **Fair Condition**: {fs['health_distribution']['Fair']} machines
- **Critical Risk**: {fs['health_distribution']['Critical']} machines
- **Total Telemetry Anomalies Detected**: {fs['total_telemetry_anomalies_detected']}

The average factory load-to-power efficiency metric stands at **{fs['average_energy_load_efficiency']}**.

---

## Predictive Failure Risk Analysis
The dynamic Random Forest Classifier was trained to evaluate machine failure modes using key features.
- **Model Evaluation Metric Summary**:
  - Accuracy: `{ml['accuracy']}`
  - Precision: `{ml['precision']}`
  - Recall: `{ml['recall']}`
  - F1-Score: `{ml['f1_score']}`

- **Top Predictive Sensory Features**:
{feat_list}

---

## Scheduled Preventive Maintenance Plan
Immediate maintenance tasks have been prioritized by calculating combined risks from remaining life, sensor anomalies, and failure probabilities:
{maint_list}

---

## Production Scheduling & Energy Optimization Insights
The following efficiency modifications are recommended to balance factory thermal loads and optimize power consumption:
{opt_list}

---
*Report generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
        return fallback_report


# ---------------------------------------------------------------------------
# NumberedCanvas — ReportLab page helper
# ---------------------------------------------------------------------------

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_elements(num_pages)
            super().showPage()
        super().save()

    def draw_page_elements(self, page_count):
        if self._pageNumber == 1:
            # Skip headers/footers on cover page
            return

        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#0f172a"))

        # Header banner text and colored line
        self.drawString(54, 750, "FACTORY AI SMART MANUFACTURING SUITE | MULTI-AGENT TECHNICAL REPORT")
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(1)
        self.line(54, 742, 558, 742)

        # Footer page numbers and line
        self.line(54, 55, 558, 55)
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#475569"))
        self.drawString(54, 40, f"Generated: {datetime.date.today().strftime('%Y-%m-%d')} | Confidential")
        self.drawRightString(558, 40, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()



# ---------------------------------------------------------------------------
# compute_financial_loss_data
# ---------------------------------------------------------------------------

def compute_financial_loss_data(coord) -> tuple:
    pred_df = coord.prediction_results['df_with_predictions']
    failed_machines = pred_df[pred_df['Predicted_Fault_Class'] == 1].copy()
    
    if failed_machines.empty:
        failed_machines = pred_df.sort_values(by='Predicted_Fault_Probability', ascending=False).head(3).copy()
        
    records = []
    total_downtime = 0.0
    total_loss = 0.0
    total_repair = 0.0
    total_production_loss = 0.0
    
    for idx, row in failed_machines.iterrows():
        m_name = row['Machine_Name']
        m_id = row['Machine_ID']
        health = row['Machine_Health']
        prob = row['Predicted_Fault_Probability']
        rul = row['Remaining_Useful_Life_Days']
        load = row['Load_Percentage']
        power = row['Power_Consumption_kW']
        hours = row['Operating_Hours']
        pred_fail = "Yes" if row['Predicted_Fault_Class'] == 1 else "No"
        
        prob_pct = prob * 100
        if prob_pct > 80 or rul < 10:
            risk = "Critical"
        elif prob_pct > 60:
            risk = "High"
        elif prob_pct > 30:
            risk = "Medium"
        else:
            risk = "Low"
            
        downtime = round(24.0 * (1.0 + (load / 100.0)) * (1.5 if health == 'Critical' else (1.2 if health == 'Fair' else 1.0)), 1)
        repair_cost = round((power * 2500) + (hours * 20))
        production_loss = round(downtime * (power * (load / 100.0) * 1500))
        t_loss = repair_cost + production_loss
        
        if risk == "Critical":
            action = "Immediate Overhaul & Calibration"
        elif risk == "High":
            action = "Replace Components & Lubricate"
        elif risk == "Medium":
            action = "Scheduled Performance Tuning"
        else:
            action = "Routine Monitoring"
            
        records.append({
            "Machine_ID": m_id,
            "Machine_Name": m_name,
            "Machine_Health": health,
            "Failure_Prediction": pred_fail,
            "Failure_Probability": f"{prob_pct:.1f}%",
            "Remaining_Useful_Life_Days": f"{rul:.0f}",
            "Risk_Level": risk,
            "Estimated_Downtime": downtime,
            "Estimated_Repair_Cost": repair_cost,
            "Estimated_Production_Loss": production_loss,
            "Total_Estimated_Loss": t_loss,
            "Recommended_Action": action
        })
        
        total_downtime += downtime
        total_loss += t_loss
        total_repair += repair_cost
        total_production_loss += production_loss
        
    total_at_risk = len(pred_df[pred_df['Predicted_Fault_Class'] == 1])
    if not failed_machines.empty:
        highest_risk_row = pred_df.sort_values(by='Predicted_Fault_Probability', ascending=False).iloc[0]
        highest_risk_m = f"{highest_risk_row['Machine_Name']} ({highest_risk_row['Predicted_Fault_Probability']*100:.1f}%)"
    else:
        highest_risk_m = "N/A"
        
    potential_savings = total_production_loss + round(total_repair * 0.5)
    
    summary = {
        "Total_Machines_at_Risk": total_at_risk,
        "Highest_Risk_Machine": highest_risk_m,
        "Total_Estimated_Downtime": round(total_downtime, 1),
        "Total_Estimated_Loss": round(total_loss),
        "Potential_Savings": round(potential_savings)
    }
    
    return records, summary


# ---------------------------------------------------------------------------
# generate_matplotlib_charts
# ---------------------------------------------------------------------------

def generate_matplotlib_charts(coord, temp_dir: str) -> bool:
    """
    Renders 4 operational charts as PNG files into temp_dir using Matplotlib (Agg backend).
    Returns True on success, False on failure.
    """
    try:
        # Chart 1: Machine Health Distribution pie chart
        summary_df = coord.health_results['machine_summary']
        categories = summary_df['Health_Category'].value_counts()
 
        fig, ax = plt.subplots(figsize=(6, 3))
        colors_map = {'Good': '#10b981', 'Fair': '#fbbf24', 'Critical': '#ef4444'}
        colors_list = [colors_map.get(cat, '#cbd5e1') for cat in categories.index]
 
        ax.pie(categories, labels=categories.index, autopct='%1.1f%%', colors=colors_list, startangle=90)
        ax.axis('equal')
        plt.title("Machine Health Distribution", fontsize=10, fontweight='bold')
        plt.tight_layout()
        pie_path = os.path.join(temp_dir, "overview_trend.png")
        plt.savefig(pie_path, dpi=150)
        plt.close(fig)
 
        # Chart 2: Feature Importance Bar Chart
        feat_imp_df = coord.prediction_results['feature_importances'].head(5)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(feat_imp_df['Feature'].str.replace('_', ' '), feat_imp_df['Importance'], color='#8b5cf6')
        ax.invert_yaxis()
        plt.title("Top Predictive Features Importance", fontsize=10, fontweight='bold')
        plt.xlabel("Importance")
        plt.tight_layout()
        bar_path = os.path.join(temp_dir, "feature_importance.png")
        plt.savefig(bar_path, dpi=150)
        plt.close(fig)
 
        # Chart 3: Load vs Power Consumption Scatter
        stats_df = coord.optimization_results['machine_stats']
        fig, ax = plt.subplots(figsize=(6, 3))
        scatter = ax.scatter(
            stats_df['Avg_Load_Pct'],
            stats_df['Avg_Power_kW'],
            s=stats_df['Total_Hours'] / 10,
            c=stats_df['Avg_Load_Efficiency'],
            cmap='viridis',
            alpha=0.8
        )
        plt.colorbar(scatter, label='Efficiency Ratio')
        plt.title("Asset Grid Load Mapping", fontsize=10, fontweight='bold')
        plt.xlabel("Avg Load Pct (%)")
        plt.ylabel("Avg Power kW")
        plt.tight_layout()
        scatter_path = os.path.join(temp_dir, "efficiency_scatter.png")
        plt.savefig(scatter_path, dpi=150)
        plt.close(fig)

        # Chart 4: Financial Loss Bar Chart
        records, _ = compute_financial_loss_data(coord)
        loss_df = pd.DataFrame(records)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(loss_df['Machine_Name'], loss_df['Total_Estimated_Loss'] / 1000.0, color='#ef4444')
        plt.title("Estimated Financial Exposure per Asset (kINR)", fontsize=10, fontweight='bold')
        plt.ylabel("Loss in Thousands (kINR)")
        plt.xlabel("Machine")
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        loss_chart_path = os.path.join(temp_dir, "financial_losses.png")
        plt.savefig(loss_chart_path, dpi=150)
        plt.close(fig)
 
        return True
    except Exception as e:
        print("Matplotlib generation failed:", str(e))
        return False


# ---------------------------------------------------------------------------
# generate_pdf_report
# ---------------------------------------------------------------------------

def generate_pdf_report(coord, report_text: str, logo_img_path: str = None, charts_dir: str = None) -> bytes:
    """
    Assembles and compiles a professional executive PDF report.
    Returns the raw PDF bytes.
    """
    buffer = io.BytesIO()

    # 0.75 in margins (54 points)
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,
        bottomMargin=72
    )

    styles = getSampleStyleSheet()

    # Custom high contrast styles
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=15
    )

    subtitle_style = ParagraphStyle(
        'CoverSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#475569"),
        spaceAfter=30
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=15,
        spaceAfter=10,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#334155"),
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=8
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=5
    )

    meta_label_style = ParagraphStyle(
        'MetaLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#000000")
    )

    meta_val_style = ParagraphStyle(
        'MetaVal',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#334155")
    )

    story = []

    # ----------------------------------------------------
    # 1. COVER PAGE
    # ----------------------------------------------------
    story.append(Spacer(1, 40))
    if logo_img_path and os.path.exists(logo_img_path):
        try:
            story.append(Image(logo_img_path, width=80, height=80))
            story.append(Spacer(1, 20))
        except Exception:
            pass

    # Cover Tag/Banner
    story.append(Paragraph("<font color='#000000'><b>FACTORY AI SYSTEM REPORT</b></font>", body_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Smart Manufacturing AI<br/>Executive Performance Report", title_style))
    story.append(Paragraph(
        "Autonomously compiled by the FACTORY AI multi-agent cognitive hierarchy, evaluating asset telemetry anomalies, "
        "machine learning failure models, and dynamic production grid balances.",
        subtitle_style
    ))
    story.append(Spacer(1, 40))

    # System info metadata table
    meta_data = [
        [
            Paragraph("<b>Document ID:</b>", meta_label_style),
            Paragraph("FACTORY-AI-SR-2026-X1", meta_val_style),
            Paragraph("<b>Generation Date:</b>", meta_label_style),
            Paragraph(datetime.date.today().strftime('%Y-%m-%d'), meta_val_style)
        ],
        [
            Paragraph("<b>Target Assets:</b>", meta_label_style),
            Paragraph(f"{len(coord.df['Machine_Name'].unique())} Active Machines", meta_val_style),
            Paragraph("<b>Overall Risk Level:</b>", meta_label_style),
            Paragraph("CRITICAL" if coord.health_results['critical_count'] > 0 else "NOMINAL", meta_val_style)
        ],
        [
            Paragraph("<b>Groq Engine:</b>", meta_label_style),
            Paragraph("Llama-3.3-70b-versatile", meta_val_style),
            Paragraph("<b>Pipeline Status:</b>", meta_label_style),
            Paragraph("SUCCESS / COMPILED", meta_val_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[100, 150, 110, 144])
    meta_table.setStyle(TableStyle([
        ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(meta_table)

    story.append(Spacer(1, 140))
    story.append(Paragraph(
        "<b>CONFIDENTIALITY NOTICE:</b> The content contained within this technical document is proprietary "
        "to plant operations control. Any unauthorized distribution is prohibited.",
        ParagraphStyle('Notice', parent=body_style, fontSize=8, leading=10, textColor=colors.HexColor("#94a3b8"))
    ))
    story.append(PageBreak())

    # ----------------------------------------------------
    # 2. TABLE OF CONTENTS
    # ----------------------------------------------------
    story.append(Paragraph("Table of Contents", h1_style))
    story.append(Spacer(1, 10))

    toc_data = [
        [Paragraph("<b>Section</b>", meta_label_style), Paragraph("<b>Page</b>", ParagraphStyle('PageH', parent=meta_label_style, alignment=2))],
        [Paragraph("1. Executive Summary", body_style), Paragraph("3", ParagraphStyle('PageVal', parent=meta_val_style, alignment=2))],
        [Paragraph("2. Global Factory Performance KPIs", body_style), Paragraph("3", ParagraphStyle('PageVal', parent=meta_val_style, alignment=2))],
        [Paragraph("3. Machine Health Agent - Telemetry Outliers", body_style), Paragraph("4", ParagraphStyle('PageVal', parent=meta_val_style, alignment=2))],
        [Paragraph("4. Failure Prediction Agent - ML Diagnosis", body_style), Paragraph("4", ParagraphStyle('PageVal', parent=meta_val_style, alignment=2))],
        [Paragraph("5. Maintenance Agent - Preventive Scheduling", body_style), Paragraph("5", ParagraphStyle('PageVal', parent=meta_val_style, alignment=2))],
        [Paragraph("6. Production Optimization Agent - Balancing Commands", body_style), Paragraph("5", ParagraphStyle('PageVal', parent=meta_val_style, alignment=2))],
        [Paragraph("7. Predicted Machine Failures & Financial Loss Analysis", body_style), Paragraph("6", ParagraphStyle('PageVal', parent=meta_val_style, alignment=2))],
        [Paragraph("8. AI Agent Executive Narrative Details", body_style), Paragraph("6", ParagraphStyle('PageVal', parent=meta_val_style, alignment=2))],
        [Paragraph("9. Approval & Sign-Off Registry", body_style), Paragraph("7", ParagraphStyle('PageVal', parent=meta_val_style, alignment=2))]
    ]
    toc_table = Table(toc_data, colWidths=[400, 104])
    toc_table.setStyle(TableStyle([
        ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor("#f1f5f9")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(toc_table)
    story.append(PageBreak())

    # ----------------------------------------------------
    # 3. EXECUTIVE SUMMARY & KPIS
    # ----------------------------------------------------
    story.append(Paragraph("1. Executive Summary", h1_style))
    exec_summary_text = (
        "This analytical report presents a comprehensive operational assessment of the manufacturing floor assets. "
        "Through data pipelines, the coordinator agent has processed raw sensor telemetry, trained "
        "and executed random forest classifiers, sorted preventive maintenance order registers, "
        "and formulated grid energy balance redirections. Detailed agent analysis sections follow, providing "
        "actionable insight blocks required to optimize equipment uptime and grid consumption profiles."
    )
    story.append(Paragraph(exec_summary_text, body_style))
    story.append(Spacer(1, 15))

    story.append(Paragraph("2. Global Factory Performance KPIs", h1_style))

    total_m = len(coord.health_results['machine_summary'])
    avg_health = coord.health_results['machine_summary']['Health_Score'].mean()
    tot_anom = coord.health_results['total_anomalies']
    critical_m = coord.health_results['critical_count']

    kpi_table_data = [
        [
            Paragraph("<b>Monitored Assets</b>", meta_label_style),
            Paragraph("<b>Mean Health Index</b>", meta_label_style),
            Paragraph("<b>Total Sensor Outliers</b>", meta_label_style),
            Paragraph("<b>Critical Asset Alert</b>", meta_label_style)
        ],
        [
            Paragraph(str(total_m), body_style),
            Paragraph(f"{avg_health:.1f}%", body_style),
            Paragraph(str(tot_anom), body_style),
            Paragraph(f"<font color='red'><b>{critical_m}</b></font>" if critical_m > 0 else "0", body_style)
        ]
    ]
    kpi_table = Table(kpi_table_data, colWidths=[126, 126, 126, 126])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f8fafc")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(kpi_table)

    # Overview trend chart
    if charts_dir:
        chart1_path = os.path.join(charts_dir, "overview_trend.png")
        if os.path.exists(chart1_path):
            story.append(Spacer(1, 15))
            story.append(Paragraph(
                "<i>Figure 1: Core Telemetry Trend Visualization</i>",
                ParagraphStyle('FigStyle', parent=body_style, fontName='Helvetica-Oblique', fontSize=8, alignment=1)
            ))
            story.append(Image(chart1_path, width=480, height=200))

    story.append(PageBreak())

    # ----------------------------------------------------
    # 4. MACHINE HEALTH AGENT & ML PREDICTIONS
    # ----------------------------------------------------
    story.append(Paragraph("3. Machine Health Agent - Telemetry Outliers", h1_style))
    story.append(Paragraph(
        "The Machine Health Agent continuously monitors telemetry sensors against dynamic threshold boundaries to tag outlier deviations:",
        body_style
    ))

    health_data = [[
        Paragraph("<b>Machine Asset</b>", meta_label_style),
        Paragraph("<b>Health Score</b>", meta_label_style),
        Paragraph("<b>Risk Status</b>", meta_label_style),
        Paragraph("<b>Total Outliers</b>", meta_label_style),
        Paragraph("<b>Avg RUL</b>", meta_label_style)
    ]]
    for idx, row in coord.health_results['machine_summary'].iterrows():
        if row['Health_Category'] == 'Critical':
            status_str = "<font color='red'><b>CRITICAL</b></font>"
        elif row['Health_Category'] == 'Fair':
            status_str = "<font color='orange'><b>WARNING</b></font>"
        else:
            status_str = "<font color='green'><b>HEALTHY</b></font>"

        health_data.append([
            Paragraph(row['Machine_Name'], body_style),
            Paragraph(f"{row['Health_Score']:.1f}%", body_style),
            Paragraph(status_str, body_style),
            Paragraph(str(row['Total_Anomalies']), body_style),
            Paragraph(f"{row['Avg_RUL_Days']:.1f} Days", body_style)
        ])

    health_table = Table(health_data, colWidths=[120, 90, 100, 100, 94])
    health_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f8fafc")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(health_table)

    story.append(Spacer(1, 20))
    story.append(Paragraph("4. Failure Prediction Agent - ML Diagnosis", h1_style))
    story.append(Paragraph(
        "The Failure Prediction Agent leverages a supervised Random Forest Classifier model to compute failure probabilities. "
        "The current diagnostic metrics evaluate to:",
        body_style
    ))

    metrics = coord.prediction_results['metrics']
    ml_metrics_data = [
        [
            Paragraph("<b>Model Metric</b>", meta_label_style),
            Paragraph("<b>Accuracy</b>", meta_label_style),
            Paragraph("<b>Precision</b>", meta_label_style),
            Paragraph("<b>Recall</b>", meta_label_style),
            Paragraph("<b>F1 Score</b>", meta_label_style)
        ],
        [
            Paragraph("Performance", body_style),
            Paragraph(f"{metrics['Accuracy']:.2%}", body_style),
            Paragraph(f"{metrics['Precision']:.2%}", body_style),
            Paragraph(f"{metrics['Recall']:.2%}", body_style),
            Paragraph(f"{metrics['F1_Score']:.2%}", body_style)
        ]
    ]
    ml_table = Table(ml_metrics_data, colWidths=[120, 96, 96, 96, 96])
    ml_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f8fafc")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(ml_table)

    # Feature importances chart
    if charts_dir:
        chart2_path = os.path.join(charts_dir, "feature_importance.png")
        if os.path.exists(chart2_path):
            story.append(Spacer(1, 15))
            story.append(Paragraph(
                "<i>Figure 2: ML Feature Importance Analysis</i>",
                ParagraphStyle('FigStyle2', parent=body_style, fontName='Helvetica-Oblique', fontSize=8, alignment=1)
            ))
            story.append(Image(chart2_path, width=480, height=200))

    story.append(PageBreak())

    # ----------------------------------------------------
    # 5. MAINTENANCE & PRODUCTION OPTIMIZATION
    # ----------------------------------------------------
    story.append(Paragraph("5. Maintenance Agent - Preventive Scheduling", h1_style))
    story.append(Paragraph(
        "The Maintenance Agent processes anomalous health records to construct a prioritized preventive queue. "
        "Critical actions are scheduled immediately:",
        body_style
    ))

    m_queue = coord.maintenance_results['maintenance_queue']
    maint_data = [[
        Paragraph("<b>Machine Asset</b>", meta_label_style),
        Paragraph("<b>Urgency</b>", meta_label_style),
        Paragraph("<b>Priority Level</b>", meta_label_style),
        Paragraph("<b>Required Actions</b>", meta_label_style)
    ]]
    for idx, row in m_queue.iterrows():
        p_str = row['Priority']
        if "HIGH" in p_str or "CRITICAL" in p_str:
            p_formatted = f"<font color='red'><b>{p_str}</b></font>"
        else:
            p_formatted = f"<font color='green'><b>{p_str}</b></font>"

        maint_data.append([
            Paragraph(row['Machine_Name'], body_style),
            Paragraph(f"{row['Urgency_Score']:.1f}", body_style),
            Paragraph(p_formatted, body_style),
            Paragraph(row['Recommended_Actions'], ParagraphStyle('MaintActions', parent=body_style, fontSize=8, leading=10))
        ])

    maint_table = Table(maint_data, colWidths=[110, 60, 114, 220])
    maint_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f8fafc")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(maint_table)

    story.append(Spacer(1, 20))
    story.append(Paragraph("6. Production Optimization Agent - Balancing Commands", h1_style))
    story.append(Paragraph(
        "The Production Optimization Agent balances operational loads to mitigate degradation risks and maximize energy efficiency:",
        body_style
    ))

    opt_res = coord.optimization_results
    opt_data = [[
        Paragraph("<b>Machine Asset</b>", meta_label_style),
        Paragraph("<b>Operational Status</b>", meta_label_style),
        Paragraph("<b>Actionable Instruction</b>", meta_label_style)
    ]]
    for idx, row in opt_res['opt_insights'].iterrows():
        opt_data.append([
            Paragraph(row['Machine_Name'], body_style),
            Paragraph(row['Status_Tag'], body_style),
            Paragraph(row['Actionable_Insight'], body_style)
        ])

    opt_table = Table(opt_data, colWidths=[130, 110, 264])
    opt_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f8fafc")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(opt_table)

    # Efficiency scatter chart
    if charts_dir:
        chart3_path = os.path.join(charts_dir, "efficiency_scatter.png")
        if os.path.exists(chart3_path):
            story.append(Spacer(1, 15))
            story.append(Paragraph(
                "<i>Figure 3: Load vs Power Consumption Optimization Scatter</i>",
                ParagraphStyle('FigStyle3', parent=body_style, fontName='Helvetica-Oblique', fontSize=8, alignment=1)
            ))
            story.append(Image(chart3_path, width=480, height=200))

    story.append(PageBreak())

    # ----------------------------------------------------
    # 7. FINANCIAL LOSS ANALYSIS SECTION
    # ----------------------------------------------------
    story.append(Paragraph("7. Predicted Machine Failures & Financial Loss Analysis", h1_style))
    story.append(Paragraph(
        "This section evaluates plant-floor financial exposure based on predictive failure models. "
        "Estimated downtime, repair costs, and production loss rates are calculated dynamically from power usage, load profiles, "
        "and mechanical health indices:",
        body_style
    ))
    story.append(Spacer(1, 10))

    records, summary = compute_financial_loss_data(coord)

    # Management summary block
    summary_data = [
        [
            Paragraph("<b>Total Machines at Risk:</b>", meta_label_style),
            Paragraph(f"{summary['Total_Machines_at_Risk']} Assets", meta_val_style),
            Paragraph("<b>Highest-Risk Asset:</b>", meta_label_style),
            Paragraph(summary['Highest_Risk_Machine'], meta_val_style)
        ],
        [
            Paragraph("<b>Total Est. Downtime:</b>", meta_label_style),
            Paragraph(f"{summary['Total_Estimated_Downtime']} Hours", meta_val_style),
            Paragraph("<b>Total Est. Loss (₹):</b>", meta_label_style),
            Paragraph(f"INR {summary['Total_Estimated_Loss']:,}", meta_val_style)
        ],
        [
            Paragraph("<b>Potential Savings (₹):</b>", meta_label_style),
            Paragraph(f"<font color='green'><b>INR {summary['Potential_Savings']:,}</b></font>", meta_val_style),
            Paragraph("", meta_label_style),
            Paragraph("", meta_val_style)
        ]
    ]
    summary_table = Table(summary_data, colWidths=[130, 120, 120, 134])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 15))

    # Detailed table of records
    loss_headers = [
        Paragraph("<b>Machine ID</b>", meta_label_style),
        Paragraph("<b>Machine Name</b>", meta_label_style),
        Paragraph("<b>Health</b>", meta_label_style),
        Paragraph("<b>Prob</b>", meta_label_style),
        Paragraph("<b>RUL (d)</b>", meta_label_style),
        Paragraph("<b>Risk</b>", meta_label_style),
        Paragraph("<b>Downtime (h)</b>", meta_label_style),
        Paragraph("<b>Loss (₹)</b>", meta_label_style),
        Paragraph("<b>Action</b>", meta_label_style)
    ]
    loss_table_data = [loss_headers]
    for r in records:
        risk_color = "red" if r['Risk_Level'] in ["Critical", "High"] else ("orange" if r['Risk_Level'] == "Medium" else "green")
        loss_table_data.append([
            Paragraph(r['Machine_ID'], body_style),
            Paragraph(r['Machine_Name'], body_style),
            Paragraph(r['Machine_Health'], body_style),
            Paragraph(r['Failure_Probability'], body_style),
            Paragraph(r['Remaining_Useful_Life_Days'], body_style),
            Paragraph(f"<font color='{risk_color}'><b>{r['Risk_Level']}</b></font>", body_style),
            Paragraph(f"{r['Estimated_Downtime']:.1f}", body_style),
            Paragraph(f"{r['Total_Estimated_Loss']:,}", body_style),
            Paragraph(r['Recommended_Action'], ParagraphStyle('RecAction', parent=body_style, fontSize=7, leading=8))
        ])

    loss_table = Table(loss_table_data, colWidths=[55, 60, 45, 45, 35, 45, 45, 64, 110])
    loss_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f8fafc")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(loss_table)
    story.append(Spacer(1, 15))

    # Matplotlib loss bar chart in PDF
    if charts_dir:
        chart4_path = os.path.join(charts_dir, "financial_losses.png")
        if os.path.exists(chart4_path):
            story.append(Paragraph(
                "<i>Figure 4: Predicted Financial Exposure by Machine Asset</i>",
                ParagraphStyle('FigStyle4', parent=body_style, fontName='Helvetica-Oblique', fontSize=8, alignment=1)
            ))
            story.append(Spacer(1, 5))
            story.append(Image(chart4_path, width=480, height=200))

    story.append(PageBreak())

    # ----------------------------------------------------
    # 8. NARRATIVE AI EXECUTIVE REPORT
    # ----------------------------------------------------
    story.append(Paragraph("8. AI Agent Executive Narrative Details", h1_style))

    # Clean the markdown formatting from the report so ReportLab Paragraphs render it properly.
    raw_lines = report_text.split("\n")
    for line in raw_lines:
        line_stripped = line.strip()
        if not line_stripped:
            story.append(Spacer(1, 5))
            continue

        # Strip dividers
        if line_stripped == "---" or line_stripped.startswith("==="):
            continue

        # Parse headings
        if line_stripped.startswith("###"):
            text = line_stripped.replace("###", "").strip()
            story.append(Paragraph(text, h2_style))
        elif line_stripped.startswith("##"):
            text = line_stripped.replace("##", "").strip()
            story.append(Paragraph(text, h2_style))
        elif line_stripped.startswith("#"):
            text = line_stripped.replace("#", "").strip()
            story.append(Paragraph(text, h1_style))
        elif line_stripped.startswith("-") or line_stripped.startswith("*"):
            text = line_stripped[1:].strip()
            text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
            story.append(Paragraph(f"&bull; {text}", bullet_style))
        else:
            text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line_stripped)
            story.append(Paragraph(text, body_style))

    story.append(Spacer(1, 30))

    # ----------------------------------------------------
    # 7. SIGN-OFF BLOCK
    # ----------------------------------------------------
    story.append(Paragraph("9. Approval & Sign-Off Registry", h1_style))
    story.append(Spacer(1, 10))

    sign_data = [
        [
            Paragraph("<b>Prepared By:</b>", meta_label_style),
            Paragraph("<b>Reviewed By:</b>", meta_label_style),
            Paragraph("<b>Approved By:</b>", meta_label_style)
        ],
        [Spacer(1, 30), Spacer(1, 30), Spacer(1, 30)],
        [
            Paragraph("FACTORY AI Report Agent<br/>AI Operations Specialist", body_style),
            Paragraph("Maintenance Director<br/>Factory Control Systems", body_style),
            Paragraph("Chief Operations Officer (COO)<br/>Manufacturing Operations", body_style)
        ]
    ]
    sign_table = Table(sign_data, colWidths=[168, 168, 168])
    sign_table.setStyle(TableStyle([
        ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(sign_table)

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)

    buffer.seek(0)
    return buffer.getvalue()
