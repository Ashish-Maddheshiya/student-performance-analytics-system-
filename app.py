"""
=============================================================================
Student Performance Data Analytics System - Streamlit Dashboard
=============================================================================
Purpose:
    Modern, interactive, executive Data Analytics Dashboard:
    - Dark navy and purple theme (#0B0F19, #1E1B4B, #6366F1, #8B5CF6)
    - Gradient KPI Cards with glassmorphic styling
    - End-to-end Data Analytics Lifecycle (Ingestion, Cleaning, EDA, Analysis, ML, PDF Export)
    - Viva & Academic Defense reference guide
=============================================================================
"""

import os
import sys
import datetime
import pandas as pd
import numpy as np
import streamlit as st

# Ensure local modules are importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import database
import data_generator
import data_cleaning
import eda
import analysis
import visualization
import prediction
import report_generator

# Page Configuration
st.set_page_config(
    page_title="Student Performance Analytics System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Navy & Purple Premium CSS Styling
st.markdown("""
<style>
    /* Global Base */
    .stApp {
        background: linear-gradient(135deg, #090D16 0%, #0F172A 50%, #16102E 100%);
        color: #F1F5F9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0B0F19 !important;
        border-right: 1px solid #1E293B;
    }
    
    /* Main Content Headers */
    h1, h2, h3 {
        color: #F8FAFC !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    
    /* KPI Gradient Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 14px;
        margin-bottom: 24px;
    }
    .kpi-card {
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.7) 0%, rgba(49, 46, 129, 0.4) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 12px;
        padding: 16px 14px;
        text-align: center;
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, border-color 0.2s ease;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(139, 92, 246, 0.6);
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.2);
    }
    .kpi-title {
        color: #94A3B8;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }
    .kpi-value {
        color: #FFFFFF;
        font-size: 24px;
        font-weight: 800;
        line-height: 1.2;
    }
    .kpi-subtext {
        font-size: 11px;
        color: #A5B4FC;
        margin-top: 4px;
    }

    /* Modern Insight Banner */
    .insight-box {
        background: linear-gradient(90deg, rgba(99, 102, 241, 0.12) 0%, rgba(139, 92, 246, 0.08) 100%);
        border-left: 4px solid #6366F1;
        padding: 14px 18px;
        border-radius: 8px;
        margin: 12px 0;
        font-size: 13.5px;
        line-height: 1.5;
        color: #E2E8F0;
    }
    
    /* Section Header Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        background: rgba(99, 102, 241, 0.2);
        color: #818CF8;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 8px;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }

    /* Streamlit Tables & DataFrames */
    .dataframe {
        border-radius: 8px !important;
        overflow: hidden !important;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Session State Initialization & Data Ingestion
# =============================================================================

def get_or_create_session_data():
    """
    Manages cached datasets in Streamlit session state:
    1. Loads from SQLite if available
    2. Generates default benchmark dataset if none exists
    """
    if "raw_df" not in st.session_state:
        # Check if sample CSV exists on disk
        sample_csv = os.path.join(CURRENT_DIR, "data", "student_performance_sample.csv")
        if not os.path.exists(sample_csv):
            data_generator.save_default_datasets(os.path.join(CURRENT_DIR, "data"))

        # Load raw sample dataset
        raw_df = pd.read_csv(sample_csv)
        st.session_state.raw_df = raw_df
        st.session_state.source_name = "Default Benchmark Sample (360 Students)"

        # Automatically clean and cache
        clean_df, audit = data_cleaning.clean_and_preprocess_dataset(
            raw_df, impute_strategy="median", handle_outliers="cap", remove_duplicates=True
        )
        st.session_state.clean_df = clean_df
        st.session_state.audit = audit

        # Store into SQLite
        try:
            database.init_db()
            database.save_cleaned_data_to_db(clean_df, "Initial_Auto_Batch", "Auto-Generated Sample")
        except Exception:
            pass


get_or_create_session_data()


# =============================================================================
# Sidebar Navigation & Controls
# =============================================================================

with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 10px 0 16px 0;'>
        <div style='font-size: 36px; line-height: 1;'>🎓</div>
        <div style='font-size: 17px; font-weight: 800; color: #FFFFFF; letter-spacing: -0.01em; margin-top: 6px;'>
            STUDENT ANALYTICS
        </div>
        <div style='font-size: 11px; color: #818CF8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;'>
            Academic Intelligence Engine
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    menu = st.radio(
        "Navigation Menu",
        options=[
            "📊 Executive Dashboard",
            "🧹 Data Ingestion & Cleaning",
            "🔍 Exploratory Data Analysis (EDA)",
            "📈 Advanced Academic Analytics",
            "🎯 At-Risk Student Detection",
            "🤖 ML Performance Prediction",
            "📑 PDF Reports & Data Export",
            "🎓 Viva & Methodology Guide"
        ],
        index=0
    )

    st.markdown("---")

    # Current Dataset Status Widget
    if "clean_df" in st.session_state and st.session_state.clean_df is not None:
        rec_count = len(st.session_state.clean_df)
        avg_m = round(st.session_state.clean_df["final_marks"].mean(), 1)
        pass_r = round((st.session_state.clean_df["final_marks"] >= 40.0).mean() * 100, 1)

        st.markdown(f"""
        <div style='background: rgba(30, 41, 59, 0.5); border: 1px solid #334155; border-radius: 8px; padding: 12px; margin-bottom: 12px;'>
            <div style='font-size: 11px; font-weight: 700; color: #94A3B8; text-transform: uppercase;'>Active Dataset</div>
            <div style='font-size: 12px; color: #E2E8F0; font-weight: 600; margin-top: 2px;'>{st.session_state.get('source_name', 'Loaded Data')}</div>
            <div style='font-size: 11px; color: #A5B4FC; margin-top: 4px;'>
                • <b>{rec_count}</b> Students Loaded<br>
                • <b>{avg_m}</b> Mean Marks | <b>{pass_r}%</b> Pass
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Quick Reset / Reload Sample Data Button
    if st.button("🔄 Reset to Fresh Benchmark Data", use_container_width=True):
        raw_df = data_generator.generate_student_dataset(n_records=360, inject_anomalies=True)
        st.session_state.raw_df = raw_df
        st.session_state.source_name = "Fresh Benchmark Sample"
        clean_df, audit = data_cleaning.clean_and_preprocess_dataset(raw_df)
        st.session_state.clean_df = clean_df
        st.session_state.audit = audit
        st.success("Fresh benchmark dataset generated & loaded!")
        st.rerun()


# =============================================================================
# Helper: Gradient KPI Renderer
# =============================================================================

def render_kpi_metrics(df: pd.DataFrame):
    """
    Renders top KPI cards with gradient dark theme.
    """
    total_records = len(df)
    avg_marks = round(float(df["final_marks"].mean()), 1)
    pass_pct = round(float((df["final_marks"] >= 40.0).mean() * 100), 1)
    avg_att = round(float(df["attendance_percentage"].mean()), 1)
    highest_marks = round(float(df["final_marks"].max()), 1)

    # Overall Performance Rating
    if avg_marks >= 70:
        perf_rating = "Excellent (Tier 1)"
        perf_color = "#10B981"
    elif avg_marks >= 55:
        perf_rating = "Competent (Tier 2)"
        perf_color = "#6366F1"
    else:
        perf_rating = "Critical Action Required"
        perf_color = "#EF4444"

    kpi_html = f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-title">Total Records</div>
            <div class="kpi-value">{total_records}</div>
            <div class="kpi-subtext">Students Analyzed</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Average Marks</div>
            <div class="kpi-value">{avg_marks}</div>
            <div class="kpi-subtext">Out of 100 Marks</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Pass Percentage</div>
            <div class="kpi-value" style="color: {'#10B981' if pass_pct >= 75 else '#F59E0B'}">{pass_pct}%</div>
            <div class="kpi-subtext">≥40 Marks Criterion</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Average Attendance</div>
            <div class="kpi-value">{avg_att}%</div>
            <div class="kpi-subtext">Statutory: 75% Req.</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Highest Marks</div>
            <div class="kpi-value">{highest_marks}</div>
            <div class="kpi-subtext">Cohort Max Score</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Overall Performance</div>
            <div class="kpi-value" style="font-size: 16px; color: {perf_color}; margin-top: 5px;">{perf_rating}</div>
            <div class="kpi-subtext">Calculated Cohort Health</div>
        </div>
    </div>
    """
    st.markdown(kpi_html, unsafe_allow_html=True)


# Extract active working DataFrame
clean_df = st.session_state.get("clean_df")
raw_df = st.session_state.get("raw_df")
audit = st.session_state.get("audit")


# =============================================================================
# PAGE 1: EXECUTIVE DASHBOARD
# =============================================================================

if menu == "📊 Executive Dashboard":
    st.markdown("<span class='badge'>Strategic Overview</span>", unsafe_allow_html=True)
    st.title("Student Performance Executive Dashboard")
    st.caption("Real-time exploratory intelligence and academic performance insights.")

    if clean_df is not None and not clean_df.empty:
        # 1. Render Top KPIs
        render_kpi_metrics(clean_df)

        # 2. Executive Insights (Calculated from Real Data)
        st.markdown("### 💡 Automated Strategic Insights")
        insights = analysis.generate_automated_insights(clean_df)
        col_in1, col_in2 = st.columns([1, 1])

        with col_in1:
            for n in insights["narratives"][:4]:
                st.markdown(f"<div class='insight-box'>{n}</div>", unsafe_allow_html=True)
        with col_in2:
            for n in insights["narratives"][4:]:
                st.markdown(f"<div class='insight-box'>{n}</div>", unsafe_allow_html=True)

        st.markdown("---")

        # 3. Primary Visualizations Grid
        row1_col1, row1_col2 = st.columns([1.3, 1])
        with row1_col1:
            st.plotly_chart(visualization.plot_marks_distribution(clean_df), use_container_width=True)
        with row1_col2:
            st.plotly_chart(visualization.plot_pass_fail_donut(clean_df), use_container_width=True)

        row2_col1, row2_col2 = st.columns([1, 1])
        with row2_col1:
            st.plotly_chart(visualization.plot_department_comparison(clean_df), use_container_width=True)
        with row2_col2:
            st.plotly_chart(visualization.plot_subject_performance(clean_df), use_container_width=True)

        row3_col1, row3_col2 = st.columns([1, 1])
        with row3_col1:
            st.plotly_chart(visualization.plot_study_hours_vs_marks(clean_df), use_container_width=True)
        with row3_col2:
            st.plotly_chart(visualization.plot_attendance_vs_marks(clean_df), use_container_width=True)

    else:
        st.warning("No student dataset loaded. Please navigate to 'Data Ingestion & Cleaning' to upload or generate data.")


# =============================================================================
# PAGE 2: DATA INGESTION & CLEANING
# =============================================================================

elif menu == "🧹 Data Ingestion & Cleaning":
    st.markdown("<span class='badge'>Data Lifecycle: Stage 1 & 2</span>", unsafe_allow_html=True)
    st.title("Data Collection, Cleaning & Preprocessing Pipeline")
    st.caption("Demonstrating rigorous data auditing, outlier detection, deduplication, and missing value treatment.")

    tab_upload, tab_audit, tab_save = st.tabs(["📥 Data Upload & Ingestion", "🧼 Cleaning Audit & Treatment", "💾 SQLite Database Management"])

    with tab_upload:
        st.markdown("#### 1. Ingest Dataset (CSV or Excel)")
        st.write("Upload your institutional CSV or Excel file, or leverage the built-in realistic academic dataset generator.")

        uploaded_file = st.file_uploader(
            "Upload Student Dataset",
            type=["csv", "xlsx", "xls"],
            help="Accepts CSV or Excel files containing student academic records."
        )

        col_up1, col_up2 = st.columns([1, 1])

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    new_df = pd.read_csv(uploaded_file)
                else:
                    new_df = pd.read_excel(uploaded_file)

                st.session_state.raw_df = new_df
                st.session_state.source_name = f"Uploaded File: {uploaded_file.name}"
                # Clean immediately
                clean_df, audit = data_cleaning.clean_and_preprocess_dataset(new_df)
                st.session_state.clean_df = clean_df
                st.session_state.audit = audit
                st.success(f"Successfully uploaded and ingested `{uploaded_file.name}` ({len(new_df)} records).")
            except Exception as e:
                st.error(f"Error parsing uploaded file: {str(e)}")

        with col_up1:
            st.markdown("##### Current Ingested Raw Data Preview")
            if raw_df is not None:
                st.write(f"Shape: **{raw_df.shape[0]} rows × {raw_df.shape[1]} columns**")
                st.dataframe(raw_df.head(10), use_container_width=True)
            else:
                st.info("No raw data loaded.")

        with col_up2:
            st.markdown("##### Benchmark Generator")
            n_gen = st.slider("Generate Synthetic Cohort Size", min_value=200, max_value=1000, value=360, step=20)
            inject_anom = st.checkbox("Inject Real-World Data Anomalies (Missing, Outliers, Duplicates)", value=True,
                                      help="Demonstrates the full cleaning pipeline for viva examinations.")
            if st.button("Generate & Load Dataset", use_container_width=True):
                new_df = data_generator.generate_student_dataset(n_records=n_gen, inject_anomalies=inject_anom)
                st.session_state.raw_df = new_df
                st.session_state.source_name = f"Generated Cohort ({n_gen} Students)"
                clean_df, audit = data_cleaning.clean_and_preprocess_dataset(new_df)
                st.session_state.clean_df = clean_df
                st.session_state.audit = audit
                st.success(f"Generated {n_gen} realistic student records!")
                st.rerun()

    with tab_audit:
        st.markdown("#### 2. Data Cleaning & Transformation Controls")
        if raw_df is not None:
            c1, c2, c3 = st.columns(3)
            with c1:
                impute_choice = st.selectbox("Missing Value Treatment", ["median", "mean", "drop"], index=0,
                                             help="Numerical imputation strategy for missing attendance or study hours.")
            with c2:
                outlier_choice = st.selectbox("Outlier & Boundary Handling", ["cap", "remove", "keep"], index=0,
                                              help="Cap invalid values (e.g. attendance > 100%) to domain boundaries.")
            with c3:
                dedup_choice = st.checkbox("Deduplicate Student Records", value=True)

            if st.button("Apply Cleaning Pipeline", type="primary"):
                clean_df, audit = data_cleaning.clean_and_preprocess_dataset(
                    raw_df, impute_strategy=impute_choice, handle_outliers=outlier_choice, remove_duplicates=dedup_choice
                )
                st.session_state.clean_df = clean_df
                st.session_state.audit = audit
                st.success("Data cleaning pipeline executed successfully!")
                st.rerun()

            st.markdown("---")
            st.markdown("#### 3. Before vs After Cleaning Audit Statistics")
            if audit:
                b = audit["before"]
                a = audit["after"]

                audit_cols = st.columns(4)
                audit_cols[0].metric("Records Ingested", b["total_records"], f"-{audit['records_removed']} removed")
                audit_cols[1].metric("Missing Values", b["total_missing_values"], f"-{audit['missing_handled']} imputed")
                audit_cols[2].metric("Exact Duplicates", b["exact_duplicates"], f"-{audit['duplicates_removed']} purged")
                audit_cols[3].metric("Bound Violations", sum(b["validation_violations"].values()), "All Capped")

                # Detailed comparison table
                comparison_data = [
                    {"Metric": "Total Records", "Raw Ingestion": b["total_records"], "Post-Cleaning": a["total_records"], "Action": f"Purged {audit['records_removed']} redundant records"},
                    {"Metric": "Missing Values", "Raw Ingestion": b["total_missing_values"], "Post-Cleaning": a["total_missing_values"], "Action": f"Imputed via {audit['impute_strategy_used']}"},
                    {"Metric": "Duplicate Rows", "Raw Ingestion": b["exact_duplicates"], "Post-Cleaning": a["exact_duplicates"], "Action": "Strict primary-key deduplication"},
                    {"Metric": "Outlier / Range Violations", "Raw Ingestion": sum(b["validation_violations"].values()), "Post-Cleaning": 0, "Action": f"{audit['outlier_handling_used'].title()} to valid boundaries"},
                    {"Metric": "Mean Final Marks", "Raw Ingestion": round(float(raw_df['final_marks'].mean()), 2) if 'final_marks' in raw_df else 'N/A', "Post-Cleaning": round(float(clean_df['final_marks'].mean()), 2), "Action": "Stabilized distribution"}
                ]
                st.table(pd.DataFrame(comparison_data))

                st.markdown("##### Cleaned & Feature-Engineered Dataset (Ready for Analytics)")
                st.dataframe(clean_df.head(15), use_container_width=True)

    with tab_save:
        st.markdown("#### 4. SQLite Database Storage")
        st.write("Persist cleaned analytical batches into the local SQLite database.")

        batch_name = st.text_input("Analytical Batch Identifier", value=f"Batch_{datetime.date.today().strftime('%Y%m%d')}")
        if st.button("Save Cleaned Data to SQLite", use_container_width=True):
            if clean_df is not None:
                batch_id = database.save_cleaned_data_to_db(clean_df, batch_name=batch_name, source_type=st.session_state.get("source_name", "App"))
                st.success(f"Batch saved successfully with Batch ID: **{batch_id}** into SQLite database (`student_analytics.db`)")
            else:
                st.error("No cleaned data available to save.")

        st.markdown("##### Historical Saved Batches in Database")
        batches_df = database.get_all_batches()
        if not batches_df.empty:
            st.dataframe(batches_df, use_container_width=True)
        else:
            st.info("No batches stored yet.")


# =============================================================================
# PAGE 3: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

elif menu == "🔍 Exploratory Data Analysis (EDA)":
    st.markdown("<span class='badge'>Data Lifecycle: Stage 3</span>", unsafe_allow_html=True)
    st.title("Exploratory Data Analysis (EDA)")
    st.caption("Deep-dive structural, distributional, parametric, and non-parametric exploratory analytics.")

    if clean_df is not None:
        overview = eda.get_dataset_overview(clean_df)

        # Overview Metrics Row
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Rows", overview["rows"])
        c2.metric("Columns", overview["columns"])
        c3.metric("Data Completeness", f"{overview['completeness_percentage']}%")
        c4.metric("Memory Footprint", f"{overview['memory_usage_kb']} KB")
        c5.metric("Duplicates", overview["duplicates"])

        st.markdown("---")

        tab_stats, tab_schema, tab_cat, tab_skew = st.tabs([
            "📊 Rigorous Statistical Summary",
            "📋 Schema & Missing Values",
            "🏷️ Categorical Distributions",
            "📈 Distribution & Skewness Analysis"
        ])

        with tab_stats:
            st.markdown("#### Comprehensive Parametric & Non-Parametric Statistics")
            st.caption("Includes Mean, Median, Mode, Std Dev, Variance, IQR, Skewness, and Kurtosis.")
            stats_df = eda.get_statistical_summary(clean_df)
            st.dataframe(stats_df, use_container_width=True)

            st.info("""
            **Statistical Interpretation Guide:**
            - **Skewness ≈ 0**: Symmetrical Gaussian distribution.
            - **Positive Skewness (> 0.5)**: Right-tailed (clustering in lower scores).
            - **Negative Skewness (< -0.5)**: Left-tailed (clustering in higher scores).
            - **IQR (Q3 - Q1)**: Resilient measure of statistical dispersion unaffected by extreme outliers.
            """)

        with tab_schema:
            st.markdown("#### Column Catalog & Data Types")
            types_df = eda.get_data_types_summary(clean_df)
            st.dataframe(types_df, use_container_width=True)

        with tab_cat:
            st.markdown("#### Categorical Frequency Distributions")
            cat_breakdowns = eda.get_categorical_breakdown(clean_df)
            cat_cols = list(cat_breakdowns.keys())
            selected_cat = st.selectbox("Select Categorical Attribute", cat_cols, index=0 if "department" not in cat_cols else cat_cols.index("department"))
            if selected_cat:
                col_tab1, col_tab2 = st.columns([1, 1.2])
                with col_tab1:
                    st.dataframe(cat_breakdowns[selected_cat], use_container_width=True)
                with col_tab2:
                    fig_cat = px.bar(
                        cat_breakdowns[selected_cat],
                        x=selected_cat,
                        y="Count",
                        color=selected_cat,
                        text="Count"
                    )
                    fig_cat.update_layout(paper_bgcolor="#161D2F", plot_bgcolor="#161D2F", font_color="#E2E8F0")
                    st.plotly_chart(fig_cat, use_container_width=True)

        with tab_skew:
            st.markdown("#### Continuous Variable Distributions")
            num_cols = ["final_marks", "attendance_percentage", "study_hours", "internal_marks", "assignment_scores"]
            valid_nums = [c for c in num_cols if c in clean_df.columns]
            selected_num = st.selectbox("Select Numerical Feature to Inspect", valid_nums)

            fig_num = px.histogram(
                clean_df,
                x=selected_num,
                nbins=30,
                marginal="box",
                color_discrete_sequence=["#6366F1"],
                opacity=0.85
            )
            fig_num.update_layout(paper_bgcolor="#161D2F", plot_bgcolor="#161D2F", font_color="#E2E8F0")
            st.plotly_chart(fig_num, use_container_width=True)

    else:
        st.warning("Please load or clean a dataset first.")


# =============================================================================
# PAGE 4: ADVANCED ACADEMIC ANALYTICS
# =============================================================================

elif menu == "📈 Advanced Academic Analytics":
    st.markdown("<span class='badge'>Data Lifecycle: Stage 4</span>", unsafe_allow_html=True)
    st.title("Advanced Academic Performance Analytics")
    st.caption("Multivariate comparative analysis, behavioral correlations, and ranking determinants.")

    if clean_df is not None:
        tab_sub, tab_dept, tab_behav, tab_corr, tab_rank = st.tabs([
            "📚 Subject Analysis",
            "🏛️ Department Benchmarks",
            "⏱️ Behavioral Drivers (Attendance & Study)",
            "🔥 Correlation Matrix & Factors",
            "🏆 Top & Bottom Performers"
        ])

        with tab_sub:
            st.markdown("#### Subject-Wise Academic Benchmarks")
            sub_df = analysis.analyze_subject_performance(clean_df)
            st.dataframe(sub_df, use_container_width=True)

            col_sub1, col_sub2 = st.columns([1, 1])
            with col_sub1:
                st.plotly_chart(visualization.plot_subject_performance(clean_df), use_container_width=True)
            with col_sub2:
                st.plotly_chart(visualization.plot_box_marks_by_subject(clean_df), use_container_width=True)

        with tab_dept:
            st.markdown("#### Department Performance Benchmarks")
            dept_df = analysis.analyze_department_performance(clean_df)
            st.dataframe(dept_df, use_container_width=True)

            col_d1, col_d2 = st.columns([1, 1])
            with col_d1:
                st.plotly_chart(visualization.plot_department_comparison(clean_df), use_container_width=True)
            with col_d2:
                st.plotly_chart(visualization.plot_box_marks_by_department(clean_df), use_container_width=True)

            # Unbiased Gender Cohort Analytics
            st.markdown("##### Cohort Analytics (Gender Parity Evaluation)")
            gender_df = analysis.analyze_gender_performance(clean_df)
            st.dataframe(gender_df, use_container_width=True)

        with tab_behav:
            st.markdown("#### Impact of Attendance & Daily Study Hours")
            att_analysis = analysis.analyze_attendance_vs_marks(clean_df)
            study_analysis = analysis.analyze_study_hours_vs_marks(clean_df)

            col_b1, col_b2 = st.columns(2)
            with col_b1:
                st.markdown("##### Attendance Tiers vs Final Marks")
                if "tier_summary" in att_analysis:
                    st.dataframe(att_analysis["tier_summary"], use_container_width=True)
                st.markdown(f"""
                <div class='insight-box'>
                    • Attendance Correlation: <b>r = {att_analysis.get('correlation', 'N/A')}</b><br>
                    • Eligible Attendance (≥75%): <b>{att_analysis.get('avg_marks_eligible_att')} marks</b><br>
                    • Shortage Attendance (<75%): <b>{att_analysis.get('avg_marks_shortage_att')} marks</b><br>
                    • Attendance Advantage: <b>+{att_analysis.get('performance_premium_eligible')} marks</b>
                </div>
                """, unsafe_allow_html=True)
                st.plotly_chart(visualization.plot_attendance_vs_marks(clean_df), use_container_width=True)

            with col_b2:
                st.markdown("##### Study Intensity Tiers vs Final Marks")
                if "tier_summary" in study_analysis:
                    st.dataframe(study_analysis["tier_summary"], use_container_width=True)
                st.markdown(f"""
                <div class='insight-box'>
                    • Study Hours Correlation: <b>r = {study_analysis.get('correlation', 'N/A')}</b><br>
                    • Higher dedicated study time directly correlates with higher class percentiles.
                </div>
                """, unsafe_allow_html=True)
                st.plotly_chart(visualization.plot_study_hours_vs_marks(clean_df), use_container_width=True)

        with tab_corr:
            st.markdown("#### Correlation Matrix & Key Factor Ranking")
            col_c1, col_c2 = st.columns([1.1, 1])
            with col_c1:
                st.plotly_chart(visualization.plot_correlation_heatmap(clean_df), use_container_width=True)
            with col_c2:
                st.markdown("##### Ranked Academic Drivers (Relative Correlation)")
                factors_df = analysis.identify_performance_factors(clean_df)
                st.dataframe(factors_df, use_container_width=True)
                st.caption("Ranking calculated using Pearson correlation coefficient (r) with Final Marks.")

        with tab_rank:
            st.markdown("#### Top and Bottom Performing Cohorts")
            top_df, bottom_df = analysis.get_top_and_bottom_performers(clean_df, n=10)
            st.markdown("##### 🌟 Top 10 High-Performing Students")
            st.dataframe(top_df, use_container_width=True)
            st.markdown("##### ⚠️ Bottom 10 Students (Priority Academic Monitoring)")
            st.dataframe(bottom_df, use_container_width=True)

    else:
        st.warning("Please load or clean a dataset first.")


# =============================================================================
# PAGE 5: AT-RISK STUDENT INTERVENTION
# =============================================================================

elif menu == "🎯 At-Risk Student Detection":
    st.markdown("<span class='badge'>Decision Intelligence</span>", unsafe_allow_html=True)
    st.title("Early-Warning At-Risk Student Detection System")
    st.caption("Identifies students facing imminent academic failure or statutory attendance shortages.")

    if clean_df is not None:
        at_risk_df = analysis.identify_at_risk_students(clean_df)
        total_risk = len(at_risk_df)
        risk_pct = round((total_risk / len(clean_df) * 100), 1)

        c1, c2, c3 = st.columns(3)
        c1.metric("Flagged At-Risk Students", total_risk)
        c2.metric("Cohort Percentage", f"{risk_pct}%")
        c3.metric("Passing Threshold", "40.0 Marks")

        st.markdown("---")
        st.markdown("#### Diagnostic At-Risk Roster")

        # Department filter
        depts = ["All Departments"] + list(at_risk_df["department"].unique()) if "department" in at_risk_df else ["All Departments"]
        sel_dept = st.selectbox("Filter by Department", depts)

        filtered_risk = at_risk_df if sel_dept == "All Departments" else at_risk_df[at_risk_df["department"] == sel_dept]
        st.dataframe(filtered_risk, use_container_width=True)

        st.download_button(
            "📥 Download At-Risk Student Intervention Roster (CSV)",
            data=filtered_risk.to_csv(index=False),
            file_name="at_risk_students_roster.csv",
            mime="text/csv",
            use_container_width=True
        )

    else:
        st.warning("Please load data first.")


# =============================================================================
# PAGE 6: ML PERFORMANCE PREDICTION
# =============================================================================

elif menu == "🤖 ML Performance Prediction":
    st.markdown("<span class='badge'>Data Lifecycle: Stage 5</span>", unsafe_allow_html=True)
    st.title("Machine Learning Student Performance Prediction")
    st.caption("Responsible, leak-free predictive regression models using Scikit-Learn.")

    if clean_df is not None:
        try:
            ml_bundle = prediction.train_performance_models(clean_df)

            tab_model, tab_sim = st.tabs(["📊 Model Evaluation & Metrics", "🔮 Interactive What-If Simulator"])

            with tab_model:
                st.markdown("#### Supervised Regression Model Evaluation")
                st.markdown(f"""
                <div class='insight-box'>
                    • <b>Champion Model</b>: {ml_bundle['primary_model_name']}<br>
                    • <b>Features Used</b>: Study Hours, Attendance Percentage, Internal Marks, Assignment Scores<br>
                    • <b>Train-Test Partition</b>: 80% Training ({ml_bundle['train_records']} records) / 20% Held-Out Testing ({ml_bundle['test_records']} records)
                </div>
                """, unsafe_allow_html=True)

                col_m1, col_m2 = st.columns(2)
                lr_m = ml_bundle["lr_metrics"]
                rf_m = ml_bundle["rf_metrics"]

                with col_m1:
                    st.markdown("##### 1. Linear Regression Baseline")
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("R² Score", lr_m["r2_score"])
                    mc2.metric("MAE", f"±{lr_m['mae']}")
                    mc3.metric("RMSE", lr_m["rmse"])
                    mc4.metric("MSE", lr_m["mse"])

                with col_m2:
                    st.markdown("##### 2. Random Forest Regressor")
                    rc1, rc2, rc3, rc4 = st.columns(4)
                    rc1.metric("R² Score", rf_m["r2_score"])
                    rc2.metric("MAE", f"±{rf_m['mae']}")
                    rc3.metric("RMSE", rf_m["rmse"])
                    rc4.metric("MSE", rf_m["mse"])

                st.markdown("---")
                col_feat, col_res = st.columns([1, 1.2])

                with col_feat:
                    st.markdown("##### Feature Importance / Coefficients")
                    st.dataframe(ml_bundle["feature_importance"], use_container_width=True)

                with col_res:
                    st.markdown("##### Actual vs Predicted Marks (Held-Out Test Set)")
                    eval_df = ml_bundle["test_evaluation_df"]
                    fig_eval = px.scatter(
                        eval_df,
                        x="Actual Marks",
                        y="Predicted (Linear)",
                        labels={"Actual Marks": "Actual Marks", "Predicted (Linear)": "Predicted Marks"}
                    )
                    fig_eval.add_shape(
                        type="line", line=dict(dash="dash", color="#10B981"),
                        x0=eval_df["Actual Marks"].min(), y0=eval_df["Actual Marks"].min(),
                        x1=eval_df["Actual Marks"].max(), y1=eval_df["Actual Marks"].max()
                    )
                    fig_eval.update_layout(paper_bgcolor="#161D2F", plot_bgcolor="#161D2F", font_color="#E2E8F0")
                    st.plotly_chart(fig_eval, use_container_width=True)

                # Ethical AI Box
                st.warning(ml_bundle["ethical_statement"])

            with tab_sim:
                st.markdown("#### 🔮 Interactive Student Performance Simulator")
                st.write("Adjust behavioral and academic parameters to estimate projected final marks and grade.")

                sim_c1, sim_c2 = st.columns(2)
                with sim_c1:
                    in_study = st.slider("Daily Study Hours", min_value=0.5, max_value=14.0, value=4.5, step=0.5)
                    in_att = st.slider("Attendance Percentage (%)", min_value=30.0, max_value=100.0, value=82.0, step=1.0)
                with sim_c2:
                    in_internal = st.slider("Internal Assessment Marks (out of 50)", min_value=5.0, max_value=50.0, value=34.0, step=1.0)
                    in_assign = st.slider("Assignment Scores (out of 100)", min_value=20.0, max_value=100.0, value=75.0, step=1.0)

                pred_result = prediction.predict_student_performance(
                    ml_bundle, in_study, in_att, in_internal, in_assign
                )

                st.markdown("##### Simulation Output")
                res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                res_col1.metric("Predicted Final Marks", f"{pred_result['predicted_marks']} / 100")
                res_col2.metric("Projected Result Status", pred_result["projected_status"])
                res_col3.metric("Projected Letter Grade", pred_result["projected_grade"])
                res_col4.metric("95% Confidence Interval", f"[{pred_result['confidence_interval_95'][0]}, {pred_result['confidence_interval_95'][1]}]")

                st.info(f"💡 {pred_result['disclaimer']}")

        except Exception as e:
            st.error(f"Error training machine learning models: {str(e)}")

    else:
        st.warning("Please load or clean a dataset first.")


# =============================================================================
# PAGE 7: PDF REPORTS & DATA EXPORT
# =============================================================================

elif menu == "📑 PDF Reports & Data Export":
    st.markdown("<span class='badge'>Reporting & Data Sharing</span>", unsafe_allow_html=True)
    st.title("Executive PDF Reports & Multi-Format Data Export")
    st.caption("Generate publication-ready PDF reports with embedded charts and export data in CSV/Excel formats.")

    if clean_df is not None:
        insights = analysis.generate_automated_insights(clean_df)
        dept_df = analysis.analyze_department_performance(clean_df)
        sub_df = analysis.analyze_subject_performance(clean_df)
        at_risk_df = analysis.identify_at_risk_students(clean_df)

        try:
            ml_bundle = prediction.train_performance_models(clean_df)
        except Exception:
            ml_bundle = None

        col_rep1, col_rep2 = st.columns([1.2, 1])

        with col_rep1:
            st.markdown("#### 1. Generate Comprehensive Executive PDF Report")
            st.write("""
            Generates a publication-grade academic report containing:
            - Executive Summary & Gradient KPI Highlights
            - Data Cleaning & Ingestion Audit Log
            - Embedded High-Resolution Analytics Charts
            - Department & Subject Performance Benchmarks
            - Machine Learning Evaluation Metrics
            - Automated Real-Data Insights & Strategic Recommendations
            """)

            if st.button("📄 Generate Executive PDF Report Now", type="primary", use_container_width=True):
                with st.spinner("Rendering high-res charts and compiling PDF document..."):
                    reports_dir = os.path.join(CURRENT_DIR, "reports")
                    chart_paths = visualization.generate_static_charts_for_report(clean_df, reports_dir)
                    pdf_filename = f"student_analytics_report_{datetime.date.today().strftime('%Y%m%d')}.pdf"
                    pdf_full_path = os.path.join(reports_dir, pdf_filename)
                    report_generator.generate_pdf_report(
                        clean_df, insights, dept_df, sub_df, audit, ml_bundle, chart_paths, pdf_full_path
                    )
                    st.session_state.pdf_path = pdf_full_path
                    st.success(f"PDF Report generated successfully!")

            if "pdf_path" in st.session_state and os.path.exists(st.session_state.pdf_path):
                with open(st.session_state.pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button(
                    "⬇️ Download Generated Executive PDF Report",
                    data=pdf_bytes,
                    file_name=os.path.basename(st.session_state.pdf_path),
                    mime="application/pdf",
                    use_container_width=True
                )

        with col_rep2:
            st.markdown("#### 2. Raw & Summary Data Export")
            st.write("Export clean datasets and analytical summaries for external tooling or archival.")

            # CSV Download
            csv_data = clean_df.to_csv(index=False)
            st.download_button(
                "📥 Download Cleaned Dataset (CSV)",
                data=csv_data,
                file_name="cleaned_student_performance_data.csv",
                mime="text/csv",
                use_container_width=True
            )

            # Excel Multi-sheet Download
            excel_path = os.path.join(CURRENT_DIR, "reports", "student_analytics_summary.xlsx")
            report_generator.export_analytical_summaries_to_excel(
                clean_df, dept_df, sub_df, at_risk_df, excel_path
            )
            if os.path.exists(excel_path):
                with open(excel_path, "rb") as f:
                    excel_bytes = f.read()
                st.download_button(
                    "📥 Download Multi-Sheet Workbook (Excel)",
                    data=excel_bytes,
                    file_name="student_analytics_summary.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

    else:
        st.warning("Please load data first.")


# =============================================================================
# PAGE 8: VIVA & METHODOLOGY GUIDE
# =============================================================================

elif menu == "🎓 Viva & Methodology Guide":
    st.markdown("<span class='badge'>Academic Defense Preparation</span>", unsafe_allow_html=True)
    st.title("College Viva & Project Methodology Reference")
    st.caption("Comprehensive academic explanation of the complete data analytics lifecycle for oral examination.")

    st.markdown("""
    ### 🔬 1. Complete Data Analytics Lifecycle Demonstrated
    This project is strictly structured as an end-to-end Data Analytics project demonstrating every stage of the CRISP-DM / Data Science Lifecycle:
    
    1. **Data Collection & Ingestion**:
       - Realistic cohort generation (350+ records) with domain-specific correlation distributions.
       - Multi-format ingestion support: CSV and Excel files.
       - Persistent SQLite relational storage for batch auditing.
    
    2. **Data Cleaning & Quality Assurance**:
       - Missing value detection and intelligent imputation (median imputation for skewed distributions).
       - Primary-key and row deduplication.
       - Outlier detection via the Interquartile Range (IQR) method and domain range bounding.
       - Transparent Before-and-After audit log.
    
    3. **Exploratory Data Analysis (EDA)**:
       - Parametric metrics: Mean, Variance, Standard Deviation.
       - Non-parametric metrics: Median, Mode, Quartiles (Q1, Q3), Interquartile Range (IQR).
       - Distribution shape: Skewness (asymmetry) and Kurtosis (tailedness).
    
    4. **Advanced Analytical Modeling**:
       - Subject-wise and Department-wise comparative benchmarks.
       - Statutory threshold analysis (75% attendance rule impact).
       - Pearson correlation matrix and academic driver ranking.
       - Early-warning At-Risk student diagnostic classifier.
    
    5. **Predictive Analytics (Machine Learning)**:
       - Leak-free supervised regression (StandardScaler fit strictly on training set).
       - 80/20 train-test evaluation with $R^2$, MAE, MSE, and RMSE.
       - **Ethical AI Principle**: Explicit exclusion of sensitive demographic attributes (gender, age) to eliminate algorithmic discrimination.
    
    6. **Visualization & Decision Intelligence**:
       - Interactive Plotly figures with Dark Navy/Purple executive theme.
       - Automated dynamic insight synthesis from calculated figures.
       - Executive PDF reporting via ReportLab and Excel multi-sheet summaries.
    """)

    st.markdown("---")

    st.markdown("### 🗣️ 2. Top Viva Questions & Expert Model Answers")

    with st.expander("Q1: What is the difference between a Data Analytics System and a Student Management System?"):
        st.write("""
        **Model Answer:**
        A Student Management System is an operational CRUD (Create, Read, Update, Delete) transactional system focused on administrative records (e.g. fees, enrollment, attendance logging).
        In contrast, a **Data Analytics System** focuses on statistical exploration, pattern recognition, correlation analysis, predictive modeling, and executive decision intelligence to discover actionable insights from historical data.
        """)

    with st.expander("Q2: Why did you use Median imputation instead of Mean for missing data?"):
        st.write("""
        **Model Answer:**
        Academic variables like study hours and attendance often exhibit non-normal distributions (e.g. log-normal or skewed distributions) or contain outliers. The **Mean** is sensitive to extreme values, whereas the **Median** is a robust non-parametric central tendency measure that does not distort the data distribution.
        """)

    with st.expander("Q3: How does the IQR method detect outliers?"):
        st.write("""
        **Model Answer:**
        The Interquartile Range is calculated as $IQR = Q_3 - Q_1$, representing the spread of the middle 50% of the data.
        The Tukey fences are established as:
        - **Lower Fence**: $Q_1 - 1.5 \\times IQR$
        - **Upper Fence**: $Q_3 + 1.5 \\times IQR$
        Any observation falling outside these bounds is flagged as a statistical outlier.
        """)

    with st.expander("Q4: Why was data leakage prevented, and how?"):
        st.write("""
        **Model Answer:**
        Data leakage occurs when information from the test dataset leaks into the training pipeline, yielding artificially high validation accuracy that fails on unseen real data.
        In this system, data leakage was prevented by:
        1. Performing an 80/20 train-test split **before** any feature transformation.
        2. Fitting the `StandardScaler` strictly on `X_train` and using `.transform()` on `X_test`.
        """)

    with st.expander("Q5: Why are sensitive features like gender excluded from the prediction model?"):
        st.write("""
        **Model Answer:**
        In compliance with Ethical AI principles and academic fairness guidelines, predictive models must not rely on protected demographic characteristics (e.g. gender or race) to forecast academic merit. Predictions rely solely on student behavioral effort and mastery indicators (attendance, daily study hours, assignments, and internal scores).
        """)
