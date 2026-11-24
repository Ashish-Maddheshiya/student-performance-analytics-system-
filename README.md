# 🎓 Student Performance Data Analytics System

An enterprise-grade, comprehensive Data Analytics System built with Python, Streamlit, Pandas, NumPy, Scikit-learn, Plotly, SQLite, and ReportLab.

> **IMPORTANT ARCHITECTURAL NOTE**: This is a **pure Data Analytics System**, NOT an operational CRUD student management system. Its purpose is to ingest, audit, clean, preprocess, explore, analyze, model, and visualize academic performance data to generate actionable decision intelligence.

---

## 🌟 Key Highlights & Features

1. **Data Collection & Ingestion**:
   - Generates realistic, domain-correlated academic datasets ($350+$ student records).
   - Ingests institutional CSV and Excel spreadsheets with validation.
   - Fallback auto-loading of benchmark datasets.
   - Multi-format ingestion support.

2. **Data Cleaning & Preprocessing Pipeline**:
   - Automated missing value detection and configurable imputation (Median, Mean, Mode).
   - Exact duplicate and primary-key (`student_id`) deduplication.
   - Outlier detection via the Interquartile Range (IQR) Tukey rule.
   - Domain range bounding (marks capped between 0–100, attendance between 0–100%).
   - Transparent Before vs After cleaning audit statistics table.

3. **Exploratory Data Analysis (EDA)**:
   - Complete dataset overview, memory footprint, and completeness index.
   - Full parametric & non-parametric statistical summary: Mean, Median, Mode, Standard Deviation, Variance, Min, Max, Range, IQR, Skewness, Kurtosis.
   - Categorical breakdown tables with percentages.

4. **Advanced Performance Analytics**:
   - Subject-wise benchmarks (average marks, pass rates, standard deviations).
   - Department-wise performance comparisons and box plot variance analysis.
   - Gender parity evaluation (unbiased comparative descriptive metrics).
   - Statutory attendance threshold analysis ($75\%$ attendance impact vs shortage).
   - Daily study hours vs final marks regression trendline.
   - Pearson correlation matrix and ranked impact drivers.
   - Top 10 and Bottom 10 performers.

5. **Automated Real-Data Insight Engine**:
   - Dynamic executive insights generated from real calculated metrics (never fabricated).
   - Highlights best/toughest subjects, highest-performing departments, and attendance advantages.

6. **Early-Warning At-Risk Student Intervention**:
   - Flags students with attendance shortages or failing internal scores.
   - Categorizes risk diagnoses for targeted academic counseling.
   - One-click export of the At-Risk student roster.

7. **Basic Machine Learning (Scikit-Learn)**:
   - Supervised regression models (Linear Regression baseline + Random Forest Regressor).
   - Proper $80/20$ train-test split with zero data leakage (`StandardScaler` fitted strictly on training data).
   - Evaluation metrics: $R^2$ Score, MAE, RMSE, MSE.
   - **Ethical AI Commitment**: Protected demographic attributes (e.g., gender, age) are strictly excluded from predictive features.
   - Interactive "What-If" simulator with $95\%$ confidence interval and clear academic disclaimers.

8. **Executive PDF Reports & Multi-Format Export**:
   - Publication-grade PDF report generated via ReportLab with embedded high-res charts and KPI tables.
   - Download cleaned dataset as CSV.
   - Download multi-sheet analytical summaries in Excel (`.xlsx`).

9. **Premium Dark Navy & Purple UI/UX**:
   - Dark theme (`#0B0F19`, `#1E1B4B`, `#6366F1`, `#8B5CF6`).
   - Glassmorphic gradient KPI cards.
   - Responsive multi-page sidebar navigation.

10. **Persistent SQLite Database**:
    - Automatically initialized `student_analytics.db`.
    - Tracks analytical batches, cleaned student records, and historical KPIs without requiring external database servers (no MySQL or XAMPP needed).

---

## 📁 Project Directory Structure

```text
student_performance_analytics/
├── app.py                     # Main interactive Streamlit multi-page dashboard
├── data_generator.py          # Realistic student cohort data generator
├── database.py                # SQLite database management and batch tracking
├── data_cleaning.py           # Cleaning, imputation, deduplication & IQR outliers
├── eda.py                     # Exploratory Data Analysis & statistical metrics
├── analysis.py                # Advanced academic analytics & dynamic insights
├── visualization.py           # Plotly interactive figures & Matplotlib PDF charts
├── prediction.py              # Scikit-learn predictive models & What-If simulator
├── report_generator.py        # ReportLab executive PDF & Excel exporter
├── requirements.txt           # Project dependencies
├── README.md                  # Comprehensive technical documentation & viva guide
├── data/
│   ├── student_performance_sample.csv
│   ├── student_performance_sample.xlsx
│   ├── student_performance_benchmark_clean.csv
│   └── student_analytics.db
├── reports/
│   ├── student_performance_analytics_report.pdf
│   └── student_analytics_summary.xlsx
└── assets/
```

---

## 🚀 Installation & Running the Application

### 1. Prerequisites
- Python 3.9+ (Python 3.12 recommended)
- Pip package manager

### 2. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application
From inside the project directory:
```bash
streamlit run app.py
```
Or directly specifying Python:
```bash
python -m streamlit run app.py
```

The application will launch automatically in your browser at `http://localhost:8501`.

---

## 🎓 College Viva & Methodology Reference Guide

### 1. Data Analytics Lifecycle
This project illustrates all stages of the CRISP-DM methodology:
1. **Business/Academic Understanding**: Identifying factors that drive student success and diagnosing at-risk students before final exams.
2. **Data Ingestion**: Standardizing multi-source CSV/Excel feeds.
3. **Data Cleaning**: Handling missingness via median imputation, eliminating duplicates, and capping outliers.
4. **Exploratory Data Analysis**: Characterizing central tendency (mean, median, mode) and dispersion (IQR, standard deviation).
5. **Advanced Statistical Modeling**: Correlating attendance, study hours, and internal marks with outcomes.
6. **Machine Learning**: Supervised evaluation using $R^2$ and MAE.
7. **Decision Support & Reporting**: Providing stakeholders with actionable PDF reports and interactive dashboards.

### 2. Key Statistical Concepts Explained
- **Why Median over Mean for Imputation?** The mean is vulnerable to extreme values, whereas the median provides a robust, distribution-preserving estimate for skewed distributions.
- **Tukey IQR Outlier Fences**: Outliers are identified as values $< Q_1 - 1.5 \times IQR$ or $> Q_3 + 1.5 \times IQR$.
- **Data Leakage Prevention**: Standardization parameters ($\mu, \sigma$) are computed strictly on `X_train`, preventing information from `X_test` from contaminating model training.
- **Ethical AI**: Demographic features are omitted to avoid perpetuating systemic bias, focusing exclusively on behavioral effort metrics.
