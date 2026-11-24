"""
=============================================================================
Student Performance Data Analytics System - Visualization Module
=============================================================================
Purpose:
    Generates rich, interactive Plotly visualizations for the Streamlit dashboard
    using a cohesive Dark Navy and Purple theme (#0B0F19, #1E1B4B, #6366F1, #8B5CF6).
    Also generates static Matplotlib / Seaborn figures for embedding into
    executive PDF reports.
=============================================================================
"""

import os
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless and thread safety
import matplotlib.pyplot as plt
import seaborn as sns

# Design Palette: Dark Navy & Purple Premium Theme
THEME_BG = "#0E131F"
THEME_CARD_BG = "#161D2F"
THEME_TEXT = "#E2E8F0"
THEME_GRID = "#28344E"
COLOR_PRIMARY = "#6366F1"    # Indigo
COLOR_SECONDARY = "#8B5CF6"  # Purple
COLOR_ACCENT = "#EC4899"     # Pink / Magenta
COLOR_CYAN = "#06B6D4"       # Cyan
COLOR_SUCCESS = "#10B981"    # Emerald
COLOR_WARNING = "#F59E0B"    # Amber
COLOR_DANGER = "#EF4444"     # Rose

PALETTE = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_CYAN, COLOR_ACCENT, COLOR_SUCCESS, COLOR_WARNING]


def _apply_dark_theme(fig: go.Figure, title: str = "") -> go.Figure:
    """
    Applies consistent Dark Navy & Purple styling to Plotly figures.
    """
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "font": {"size": 16, "color": "#F8FAFC", "family": "Inter, Roboto, sans-serif"},
            "x": 0.02,
            "xanchor": "left"
        },
        paper_bgcolor=THEME_CARD_BG,
        plot_bgcolor=THEME_CARD_BG,
        font={"color": THEME_TEXT, "family": "Inter, Roboto, sans-serif"},
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(
            bgcolor="rgba(22, 29, 47, 0.8)",
            bordercolor=THEME_GRID,
            borderwidth=1,
            font=dict(color=THEME_TEXT, size=11)
        ),
        xaxis=dict(gridcolor=THEME_GRID, zerolinecolor=THEME_GRID),
        yaxis=dict(gridcolor=THEME_GRID, zerolinecolor=THEME_GRID)
    )
    return fig


# =============================================================================
# 1. Plotly Interactive Figures
# =============================================================================

def plot_marks_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Histogram and KDE curve of final marks with pass benchmark line.
    """
    fig = px.histogram(
        df,
        x="final_marks",
        nbins=25,
        marginal="box",
        color="result_status",
        color_discrete_map={"Pass": COLOR_PRIMARY, "Fail": COLOR_DANGER},
        opacity=0.85,
        labels={"final_marks": "Final Marks (out of 100)", "result_status": "Status"}
    )
    fig.add_vline(
        x=40.0, line_dash="dash", line_color=COLOR_WARNING,
        annotation_text="Passing Threshold (40 Marks)",
        annotation_position="top left",
        annotation_font_color=COLOR_WARNING
    )
    return _apply_dark_theme(fig, "Final Marks Distribution & Box Plot")


def plot_department_comparison(df: pd.DataFrame) -> go.Figure:
    """
    Bar chart comparing average marks and attendance across departments.
    """
    dept_summary = df.groupby("department").agg(
        avg_marks=("final_marks", "mean"),
        avg_att=("attendance_percentage", "mean"),
        count=("final_marks", "count")
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dept_summary["department"],
        y=dept_summary["avg_marks"].round(2),
        name="Average Marks",
        marker_color=COLOR_PRIMARY,
        text=dept_summary["avg_marks"].round(1),
        textposition="auto"
    ))
    fig.add_trace(go.Bar(
        x=dept_summary["department"],
        y=dept_summary["avg_att"].round(2),
        name="Average Attendance (%)",
        marker_color=COLOR_CYAN,
        text=dept_summary["avg_att"].round(1),
        textposition="auto"
    ))
    fig.update_layout(barmode="group", yaxis_title="Score / Percentage")
    return _apply_dark_theme(fig, "Department Performance & Attendance Comparison")


def plot_subject_performance(df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart of subject-wise average marks with pass rates.
    """
    sub_summary = df.groupby("subject").agg(
        avg_marks=("final_marks", "mean"),
        pass_rate=("final_marks", lambda x: (x >= 40.0).mean() * 100)
    ).reset_index().sort_values(by="avg_marks", ascending=True)

    fig = go.Figure(go.Bar(
        x=sub_summary["avg_marks"].round(2),
        y=sub_summary["subject"],
        orientation="h",
        marker=dict(
            color=sub_summary["avg_marks"],
            colorscale=[[0, COLOR_DANGER], [0.5, COLOR_SECONDARY], [1.0, COLOR_SUCCESS]],
            showscale=True,
            colorbar=dict(title="Avg Marks")
        ),
        text=sub_summary["avg_marks"].round(1).astype(str) + " marks (" + sub_summary["pass_rate"].round(0).astype(int).astype(str) + "% pass)",
        textposition="inside"
    ))
    fig.update_layout(xaxis_title="Average Marks", yaxis_title="Subject")
    return _apply_dark_theme(fig, "Subject-Wise Academic Performance Benchmark")


def plot_study_hours_vs_marks(df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot showing study hours vs final marks with trendline.
    """
    fig = px.scatter(
        df,
        x="study_hours",
        y="final_marks",
        color="department",
        size="internal_marks",
        hover_data=["student_id", "name", "attendance_percentage"],
        trendline="ols",
        trendline_color_override="#F8FAFC",
        color_discrete_sequence=PALETTE,
        labels={"study_hours": "Daily Study Hours", "final_marks": "Final Marks"}
    )
    return _apply_dark_theme(fig, "Daily Study Hours vs Final Marks (Size: Internal Marks)")


def plot_attendance_vs_marks(df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot of attendance percentage vs final marks with 75% statutory boundary.
    """
    fig = px.scatter(
        df,
        x="attendance_percentage",
        y="final_marks",
        color="result_status",
        color_discrete_map={"Pass": COLOR_CYAN, "Fail": COLOR_DANGER},
        hover_data=["student_id", "name", "department", "subject"],
        trendline="ols",
        trendline_color_override="#A855F7",
        labels={"attendance_percentage": "Attendance Percentage (%)", "final_marks": "Final Marks"}
    )
    fig.add_vline(
        x=75.0, line_dash="dash", line_color=COLOR_WARNING,
        annotation_text="75% Mandatory Attendance Threshold",
        annotation_position="bottom right",
        annotation_font_color=COLOR_WARNING
    )
    return _apply_dark_theme(fig, "Attendance Percentage vs Final Marks")


def plot_assignments_vs_final(df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot of assignment scores vs final marks.
    """
    fig = px.scatter(
        df,
        x="assignment_scores",
        y="final_marks",
        color="grade",
        hover_data=["student_id", "name", "department"],
        color_discrete_sequence=PALETTE,
        labels={"assignment_scores": "Assignment Scores (out of 100)", "final_marks": "Final Marks"}
    )
    return _apply_dark_theme(fig, "Assignment Scores vs Final Examination Marks")


def plot_pass_fail_donut(df: pd.DataFrame) -> go.Figure:
    """
    Donut chart of Pass vs Fail ratio and Grade distribution.
    """
    status_counts = df["result_status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]

    fig = px.pie(
        status_counts,
        values="Count",
        names="Status",
        hole=0.55,
        color="Status",
        color_discrete_map={"Pass": COLOR_SUCCESS, "Fail": COLOR_DANGER}
    )
    fig.update_traces(textinfo="label+percent", textfont_size=13)
    return _apply_dark_theme(fig, "Academic Result Status (Pass vs Fail Ratio)")


def plot_grade_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Bar chart showing breakdown across letter grades.
    """
    if "grade" not in df.columns:
        return go.Figure()

    grade_counts = df["grade"].value_counts().reset_index()
    grade_counts.columns = ["Grade", "Count"]

    fig = px.bar(
        grade_counts,
        x="Grade",
        y="Count",
        color="Grade",
        color_discrete_sequence=PALETTE,
        text="Count"
    )
    fig.update_traces(textposition="outside")
    return _apply_dark_theme(fig, "Grade Distribution Analysis")


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Annotated correlation heatmap for numerical features.
    """
    num_cols = ["study_hours", "attendance_percentage", "assignment_scores", "internal_marks", "final_marks"]
    valid_cols = [c for c in num_cols if c in df.columns]
    corr = df[valid_cols].corr().round(2)

    labels = [c.replace("_", " ").title() for c in valid_cols]

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=labels,
        y=labels,
        colorscale=[[0, "#1E1B4B"], [0.5, "#6366F1"], [1.0, "#06B6D4"]],
        text=corr.values,
        texttemplate="%{text}",
        textfont={"size": 13, "color": "#FFFFFF"},
        colorbar=dict(title="r")
    ))
    return _apply_dark_theme(fig, "Correlation Matrix of Academic Variables")


def plot_box_marks_by_department(df: pd.DataFrame) -> go.Figure:
    """
    Box plot of final marks by department to reveal variance and outliers.
    """
    fig = px.box(
        df,
        x="department",
        y="final_marks",
        color="department",
        color_discrete_sequence=PALETTE,
        points="all",
        labels={"department": "Department", "final_marks": "Final Marks"}
    )
    return _apply_dark_theme(fig, "Marks Dispersion Across Departments (Box Plot)")


def plot_box_marks_by_subject(df: pd.DataFrame) -> go.Figure:
    """
    Box plot of marks by subject.
    """
    fig = px.box(
        df,
        x="subject",
        y="final_marks",
        color="subject",
        color_discrete_sequence=PALETTE,
        points="outliers",
        labels={"subject": "Subject", "final_marks": "Final Marks"}
    )
    return _apply_dark_theme(fig, "Subject Marks Dispersion & Outlier Analysis")


def plot_attendance_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Histogram of student attendance distribution.
    """
    fig = px.histogram(
        df,
        x="attendance_percentage",
        nbins=20,
        color_discrete_sequence=[COLOR_CYAN],
        labels={"attendance_percentage": "Attendance Percentage (%)"}
    )
    fig.add_vline(
        x=75.0, line_dash="dash", line_color=COLOR_WARNING,
        annotation_text="75% Threshold",
        annotation_position="top left",
        annotation_font_color=COLOR_WARNING
    )
    return _apply_dark_theme(fig, "Attendance Percentage Distribution")


# =============================================================================
# 2. Static Matplotlib Figures for PDF Report
# =============================================================================

def generate_static_charts_for_report(df: pd.DataFrame, output_dir: str = "reports") -> Dict[str, str]:
    """
    Generates high-resolution PNG charts using Matplotlib and Seaborn
    specifically tailored for embedding into executive PDF reports.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Chart 1: Final Marks Distribution
    fig, ax = plt.subplots(figsize=(7, 3.8), dpi=200)
    sns.histplot(df["final_marks"], kde=True, color="#4F46E5", bins=22, ax=ax)
    ax.axvline(40.0, color="#EF4444", linestyle="--", linewidth=1.5, label="Passing Mark (40)")
    ax.set_title("Distribution of Student Final Marks", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Final Marks", fontsize=10)
    ax.set_ylabel("Number of Students", fontsize=10)
    ax.legend()
    plt.tight_layout()
    p1 = os.path.join(output_dir, "report_chart_dist.png")
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    paths["marks_dist"] = p1

    # Chart 2: Correlation Heatmap
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=200)
    num_cols = ["study_hours", "attendance_percentage", "assignment_scores", "internal_marks", "final_marks"]
    valid_cols = [c for c in num_cols if c in df.columns]
    corr = df[valid_cols].corr()
    clean_labels = [c.replace("_", " ").title() for c in valid_cols]
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Purples", xticklabels=clean_labels, yticklabels=clean_labels, ax=ax, cbar=True)
    ax.set_title("Academic Variables Correlation Matrix", fontsize=12, fontweight="bold", pad=10)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    p2 = os.path.join(output_dir, "report_chart_corr.png")
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    paths["correlation"] = p2

    # Chart 3: Department Comparison
    fig, ax = plt.subplots(figsize=(7, 3.8), dpi=200)
    dept_df = df.groupby("department")["final_marks"].mean().reset_index().sort_values(by="final_marks", ascending=False)
    bars = ax.bar(dept_df["department"], dept_df["final_marks"], color="#7C3AED", edgecolor="#4C1D95", width=0.55)
    ax.set_title("Department-Wise Average Final Marks", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("Average Marks (100)", fontsize=10)
    ax.set_ylim(0, 100)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.xticks(rotation=20, ha="right", fontsize=9)
    plt.tight_layout()
    p3 = os.path.join(output_dir, "report_chart_dept.png")
    fig.savefig(p3, bbox_inches="tight")
    plt.close(fig)
    paths["department"] = p3

    # Chart 4: Attendance vs Final Marks Scatter
    fig, ax = plt.subplots(figsize=(7, 3.8), dpi=200)
    sns.regplot(data=df, x="attendance_percentage", y="final_marks", scatter_kws={"alpha": 0.55, "color": "#2563EB"}, line_kws={"color": "#DC2626"}, ax=ax)
    ax.axvline(75.0, color="#F59E0B", linestyle="--", label="75% Attendance Threshold")
    ax.set_title("Attendance Percentage vs Final Marks (Linear Trend)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Attendance (%)", fontsize=10)
    ax.set_ylabel("Final Marks", fontsize=10)
    ax.legend()
    plt.tight_layout()
    p4 = os.path.join(output_dir, "report_chart_attendance.png")
    fig.savefig(p4, bbox_inches="tight")
    plt.close(fig)
    paths["attendance"] = p4

    return paths
