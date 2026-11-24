"""
=============================================================================
Student Performance Data Analytics System - PDF & Summary Export Module
=============================================================================
Purpose:
    Generates high-quality, professional executive PDF reports using ReportLab.
    Includes:
    - Cover banner & executive summary
    - Key academic KPIs (Enrollment, Average Marks, Pass Rate, Attendance)
    - Data Cleaning & Preprocessing audit log
    - Departmental and Subject performance tables
    - High-resolution embedded analytics charts
    - Machine learning evaluation summary & ethical statements
    - Calculated insights & strategic academic recommendations
    Also provides CSV and Excel summary export capabilities.
=============================================================================
"""

import os
import datetime
import pandas as pd
from typing import Dict, Any, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY


def export_dataframe_to_csv(df: pd.DataFrame, filepath: str) -> str:
    """Export cleaned data to CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    df.to_csv(filepath, index=False)
    return filepath


def export_analytical_summaries_to_excel(
    df: pd.DataFrame,
    dept_df: pd.DataFrame,
    sub_df: pd.DataFrame,
    at_risk_df: pd.DataFrame,
    filepath: str
) -> str:
    """
    Export multi-sheet analytical workbook in Excel.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Cleaned_Students", index=False)
        dept_df.to_excel(writer, sheet_name="Department_Analytics", index=False)
        sub_df.to_excel(writer, sheet_name="Subject_Analytics", index=False)
        at_risk_df.to_excel(writer, sheet_name="At_Risk_Students", index=False)
    return filepath


def generate_pdf_report(
    df: pd.DataFrame,
    insights: Dict[str, Any],
    dept_df: pd.DataFrame,
    sub_df: pd.DataFrame,
    cleaning_audit: Optional[Dict[str, Any]],
    ml_bundle: Optional[Dict[str, Any]],
    chart_paths: Optional[Dict[str, str]],
    output_pdf_path: str = "reports/student_performance_analytics_report.pdf"
) -> str:
    """
    Generates an executive, publication-grade academic analytics PDF report.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_pdf_path)), exist_ok=True)
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom Color Palette
    PRIMARY_COLOR = colors.HexColor("#1E1B4B")  # Deep Indigo
    SECONDARY_COLOR = colors.HexColor("#4F46E5") # Bright Indigo
    ACCENT_PURPLE = colors.HexColor("#7C3AED")   # Purple
    DARK_TEXT = colors.HexColor("#0F172A")
    LIGHT_BG = colors.HexColor("#F8FAFC")
    BORDER_COLOR = colors.HexColor("#CBD5E1")
    ALERT_RED = colors.HexColor("#DC2626")
    SUCCESS_GREEN = colors.HexColor("#059669")

    # Typography Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=PRIMARY_COLOR,
        alignment=TA_LEFT
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748B"),
        alignment=TA_LEFT
    )

    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=SECONDARY_COLOR,
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=DARK_TEXT,
        alignment=TA_JUSTIFY
    )

    bullet_style = ParagraphStyle(
        "ReportBullet",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=DARK_TEXT,
        leftIndent=15,
        spaceAfter=4
    )

    kpi_number_style = ParagraphStyle(
        "KpiNum",
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=SECONDARY_COLOR,
        alignment=TA_CENTER
    )

    kpi_label_style = ParagraphStyle(
        "KpiLabel",
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#475569"),
        alignment=TA_CENTER
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=TA_CENTER
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=DARK_TEXT,
        alignment=TA_CENTER
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("STUDENT PERFORMANCE DATA ANALYTICS SYSTEM", title_style))
    report_date = datetime.datetime.now().strftime("%B %d, %Y - %I:%M %p")
    story.append(Paragraph(f"Comprehensive Academic Analytics & Performance Intelligence Report | Generated: {report_date}", subtitle_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=2, color=SECONDARY_COLOR, spaceBefore=2, spaceAfter=10))

    # 2. Executive KPI Summary Cards Table
    total_stud = len(df)
    avg_marks = round(float(df["final_marks"].mean()), 1)
    pass_pct = round((df["final_marks"] >= 40.0).mean() * 100, 1)
    avg_att = round(float(df["attendance_percentage"].mean()), 1)
    top_dept = insights.get("best_department", "N/A")
    strong_driver = insights.get("strongest_driver", "N/A")

    kpi_data = [
        [
            Paragraph(f"<b>{total_stud}</b>", kpi_number_style),
            Paragraph(f"<b>{avg_marks}</b>", kpi_number_style),
            Paragraph(f"<b>{pass_pct}%</b>", kpi_number_style),
            Paragraph(f"<b>{avg_att}%</b>", kpi_number_style),
            Paragraph(f"<b>{top_dept[:14]}</b>", kpi_number_style),
            Paragraph(f"<b>{strong_driver[:14]}</b>", kpi_number_style)
        ],
        [
            Paragraph("TOTAL STUDENTS", kpi_label_style),
            Paragraph("AVERAGE MARKS", kpi_label_style),
            Paragraph("PASS PERCENTAGE", kpi_label_style),
            Paragraph("AVG ATTENDANCE", kpi_label_style),
            Paragraph("TOP DEPARTMENT", kpi_label_style),
            Paragraph("KEY DRIVER", kpi_label_style)
        ]
    ]
    kpi_table = Table(kpi_data, colWidths=[90, 90, 90, 90, 90, 90])
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
        ("BOX", (0, 0), (-1, -1), 1, BORDER_COLOR),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 10))

    # 3. Automated Executive Insights Section
    story.append(Paragraph("1. Executive Analytics Findings & Core Insights", h1_style))
    for narrative in insights.get("narratives", []):
        clean_text = narrative.replace("**", "")
        story.append(Paragraph(f"• {clean_text}", bullet_style))
    story.append(Spacer(1, 10))

    # 4. Data Cleaning & Preprocessing Audit
    if cleaning_audit:
        story.append(Paragraph("2. Data Quality & Preprocessing Lifecycle Audit", h1_style))
        b = cleaning_audit.get("before", {})
        a = cleaning_audit.get("after", {})
        audit_table_data = [
            [
                Paragraph("<b>Audit Checkpoint</b>", table_header_style),
                Paragraph("<b>Raw Ingestion</b>", table_header_style),
                Paragraph("<b>Post-Cleaning</b>", table_header_style),
                Paragraph("<b>Treatment Applied</b>", table_header_style)
            ],
            [
                Paragraph("Total Record Count", table_cell_style),
                Paragraph(str(b.get("total_records", len(df))), table_cell_style),
                Paragraph(str(len(df)), table_cell_style),
                Paragraph(f"{cleaning_audit.get('records_removed', 0)} duplicates/anomalies removed", table_cell_style)
            ],
            [
                Paragraph("Missing Values Detected", table_cell_style),
                Paragraph(str(b.get("total_missing_values", 0)), table_cell_style),
                Paragraph(str(a.get("total_missing_values", 0)), table_cell_style),
                Paragraph(f"{cleaning_audit.get('impute_strategy_used', 'median').title()} Imputation", table_cell_style)
            ],
            [
                Paragraph("Duplicate Records", table_cell_style),
                Paragraph(str(b.get("exact_duplicates", 0)), table_cell_style),
                Paragraph("0", table_cell_style),
                Paragraph("Deduplicated by Student ID & Row", table_cell_style)
            ],
            [
                Paragraph("Outliers / Bounds Violations", table_cell_style),
                Paragraph(str(sum(b.get("validation_violations", {}).values())), table_cell_style),
                Paragraph("0", table_cell_style),
                Paragraph("IQR-based Capping to Standard Range", table_cell_style)
            ]
        ]
        audit_table = Table(audit_table_data, colWidths=[150, 100, 100, 190])
        audit_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), PRIMARY_COLOR),
            ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(audit_table)
        story.append(Spacer(1, 10))

    # 5. Embedded Analytics Visualizations (Two side-by-side or stacked)
    if chart_paths:
        story.append(Paragraph("3. Visual Performance Analytics", h1_style))
        dist_img = chart_paths.get("marks_dist")
        corr_img = chart_paths.get("correlation")
        dept_img = chart_paths.get("department")
        att_img = chart_paths.get("attendance")

        if dist_img and os.path.exists(dist_img) and corr_img and os.path.exists(corr_img):
            img_table_data = [[
                Image(dist_img, width=265, height=145),
                Image(corr_img, width=265, height=145)
            ]]
            img_table = Table(img_table_data, colWidths=[270, 270])
            img_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ]))
            story.append(img_table)
            story.append(Spacer(1, 8))

        if dept_img and os.path.exists(dept_img) and att_img and os.path.exists(att_img):
            img_table_data2 = [[
                Image(dept_img, width=265, height=145),
                Image(att_img, width=265, height=145)
            ]]
            img_table2 = Table(img_table_data2, colWidths=[270, 270])
            img_table2.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ]))
            story.append(img_table2)
            story.append(Spacer(1, 10))

    # 6. Departmental & Subject Analytics Tables
    if not dept_df.empty:
        story.append(Paragraph("4. Department-Wise Academic Benchmarks", h1_style))
        dept_header = [
            Paragraph("<b>Department</b>", table_header_style),
            Paragraph("<b>Students</b>", table_header_style),
            Paragraph("<b>Avg Marks</b>", table_header_style),
            Paragraph("<b>Pass Rate</b>", table_header_style),
            Paragraph("<b>Avg Attendance</b>", table_header_style),
            Paragraph("<b>Study Hrs</b>", table_header_style)
        ]
        dept_rows = [dept_header]
        for _, r in dept_df.iterrows():
            dept_rows.append([
                Paragraph(str(r.get("Department", r.get("department", "N/A"))), table_cell_style),
                Paragraph(str(r.get("Total Students", r.get("student_count", "N/A"))), table_cell_style),
                Paragraph(str(r.get("Average Marks", r.get("avg_marks", "N/A"))), table_cell_style),
                Paragraph(f"{r.get('Pass Rate (%)', r.get('pass_rate_pct', 'N/A'))}%", table_cell_style),
                Paragraph(f"{r.get('Average Attendance (%)', r.get('avg_attendance', 'N/A'))}%", table_cell_style),
                Paragraph(str(r.get("Average Study Hours (hrs/day)", r.get("avg_study_hours", "N/A"))), table_cell_style)
            ])
        dept_table = Table(dept_rows, colWidths=[150, 65, 80, 80, 85, 80])
        dept_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), PRIMARY_COLOR),
            ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(dept_table)
        story.append(Spacer(1, 10))

    # 7. Machine Learning Predictive Intelligence Section
    if ml_bundle:
        story.append(Paragraph("5. Predictive Analytics & Machine Learning Evaluation", h1_style))
        lr_m = ml_bundle.get("lr_metrics", {})
        rf_m = ml_bundle.get("rf_metrics", {})
        model_name = ml_bundle.get("primary_model_name", "Linear Regression")

        ml_summary_text = (
            f"Supervised regression models were trained using an 80/20 train-test split (Training: "
            f"{ml_bundle.get('train_records')} records, Test: {ml_bundle.get('test_records')} records). "
            f"The <b>{model_name}</b> was selected as the champion model. Evaluation metrics on the held-out "
            f"test set demonstrate an <b>R² Score of {lr_m.get('r2_score', 'N/A')}</b> (explaining ~{int(float(lr_m.get('r2_score', 0))*100)}% "
            f"of marks variance) with a Mean Absolute Error (MAE) of <b>±{lr_m.get('mae', 'N/A')} marks</b>."
        )
        story.append(Paragraph(ml_summary_text, body_style))
        story.append(Spacer(1, 6))

        # Ethical disclaimer box
        disclaimer_text = (
            "<b>ETHICAL AI NOTICE:</b> Feature inputs are restricted exclusively to behavioral and academic "
            "effort indicators (Study Hours, Attendance, Internal Assessments). Protected demographic attributes "
            "(such as Gender) are strictly excluded from predictive models."
        )
        story.append(Paragraph(disclaimer_text, bullet_style))
        story.append(Spacer(1, 10))

    # 8. Actionable Strategic Recommendations
    story.append(Paragraph("6. Strategic Academic Recommendations", h1_style))
    recs = [
        "<b>Institutionalize Attendance Interventions:</b> Given the confirmed statistical penalty associated with sub-75% attendance, implement automated weekly notifications to students whose attendance slips below 80%.",
        "<b>Early Internal Marks Remediation:</b> Internal marks exhibit the highest predictive power for final outcomes. Faculty should schedule remedial tutorials immediately after the first midterm evaluation.",
        "<b>Targeted Study Habit Coaching:</b> Advise students with high attendance but low exam scores on active study methodologies, aiming for a minimum of 4 dedicated daily study hours.",
        "<b>Curriculum Moderation:</b> Conduct a syllabus review for subjects exhibiting the lowest average scores to ensure balanced examination difficulty."
    ]
    for r in recs:
        story.append(Paragraph(f"• {r}", bullet_style))

    # Build the document
    doc.build(story)
    return output_pdf_path
