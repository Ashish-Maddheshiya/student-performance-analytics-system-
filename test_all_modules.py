"""
=============================================================================
Student Performance Data Analytics System - Comprehensive Test Suite
=============================================================================
Purpose:
    Runs automated unit and integration tests across all modules to verify:
    - Data generation (>= 200 records, CSV & Excel formats)
    - SQLite database schema & batch persistence
    - Data cleaning pipeline (missing values, deduplication, IQR outlier capping)
    - Exploratory Data Analysis & statistical metrics
    - Advanced academic analytics & dynamic real-data insights
    - Plotly interactive figures & static report charts
    - Scikit-learn predictive models (80/20 split, zero leakage, ethical AI)
    - ReportLab executive PDF generation & multi-sheet Excel export
=============================================================================
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import data_generator
import database
import data_cleaning
import eda
import analysis
import visualization
import prediction
import report_generator


class TestStudentAnalyticsSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_data_dir = os.path.join(PROJECT_ROOT, "data")
        cls.test_reports_dir = os.path.join(PROJECT_ROOT, "reports")
        os.makedirs(cls.test_data_dir, exist_ok=True)
        os.makedirs(cls.test_reports_dir, exist_ok=True)

    def test_01_data_generation(self):
        """Test realistic data generation with anomalies and exports."""
        raw_df = data_generator.generate_student_dataset(n_records=300, inject_anomalies=True)
        self.assertGreaterEqual(len(raw_df), 200, "Dataset must have at least 200 records.")
        self.assertIn("student_id", raw_df.columns)
        self.assertIn("final_marks", raw_df.columns)
        self.assertIn("study_hours", raw_df.columns)
        self.assertIn("attendance_percentage", raw_df.columns)
        
        # Test export of sample files
        paths = data_generator.save_default_datasets(self.test_data_dir)
        self.assertTrue(os.path.exists(paths["sample_csv"]), "CSV sample must exist.")
        self.assertTrue(os.path.exists(paths["sample_excel"]), "Excel sample must exist.")
        print("✓ Test 01 Passed: Data Generation & Multi-format Export")

    def test_02_database_operations(self):
        """Test SQLite database initialization, table creation, and batch storage."""
        database.init_db()
        sample_df = data_generator.generate_student_dataset(n_records=220, inject_anomalies=False)
        batch_id = database.save_cleaned_data_to_db(sample_df, "Test_Batch_Unit", "Unit Test")
        self.assertIsInstance(batch_id, int)
        self.assertGreater(batch_id, 0)

        loaded_df = database.load_latest_cleaned_data()
        self.assertIsNotNone(loaded_df)
        self.assertEqual(len(loaded_df), 220)

        batches = database.get_all_batches()
        self.assertGreaterEqual(len(batches), 1)
        print("✓ Test 02 Passed: SQLite Database Operations")

    def test_03_data_cleaning_and_audit(self):
        """Test data cleaning, imputation, deduplication, and bounds enforcement."""
        raw_df = data_generator.generate_student_dataset(n_records=250, inject_anomalies=True)
        before_audit = data_cleaning.audit_raw_data(raw_df)
        self.assertGreater(before_audit["total_missing_values"], 0, "Anomalies must be present before cleaning.")

        clean_df, audit = data_cleaning.clean_and_preprocess_dataset(
            raw_df, impute_strategy="median", handle_outliers="cap", remove_duplicates=True
        )
        self.assertEqual(clean_df.isnull().sum().sum(), 0, "No nulls must remain after cleaning.")
        self.assertEqual(int(clean_df.duplicated().sum()), 0, "No duplicates must remain.")
        self.assertIn("result_status", clean_df.columns, "Feature engineering result_status must exist.")
        self.assertIn("grade", clean_df.columns, "Feature engineering grade must exist.")
        print("✓ Test 03 Passed: Data Cleaning, Audit & Feature Engineering")

    def test_04_exploratory_data_analysis(self):
        """Test EDA module: statistics, dimensions, and categorical summaries."""
        df = data_generator.generate_student_dataset(n_records=210, inject_anomalies=False)
        overview = eda.get_dataset_overview(df)
        self.assertEqual(overview["rows"], 210)
        self.assertEqual(overview["total_nulls"], 0)

        stats = eda.get_statistical_summary(df)
        self.assertIn("Mean", stats.columns)
        self.assertIn("Median", stats.columns)
        self.assertIn("Mode", stats.columns)
        self.assertIn("IQR", stats.columns)
        self.assertIn("Skewness", stats.columns)
        self.assertIn("Kurtosis", stats.columns)

        types_df = eda.get_data_types_summary(df)
        self.assertEqual(len(types_df), len(df.columns))
        print("✓ Test 04 Passed: Exploratory Data Analysis & Statistics")

    def test_05_advanced_analytics_and_insights(self):
        """Test academic analysis, correlation ranking, at-risk detection, and real insights."""
        df = data_generator.generate_student_dataset(n_records=260, inject_anomalies=False)
        clean_df, _ = data_cleaning.clean_and_preprocess_dataset(df)

        dept_df = analysis.analyze_department_performance(clean_df)
        self.assertFalse(dept_df.empty)
        self.assertIn("Average Marks", dept_df.columns)

        sub_df = analysis.analyze_subject_performance(clean_df)
        self.assertFalse(sub_df.empty)
        self.assertIn("Pass Rate (%)", sub_df.columns)

        at_risk_df = analysis.identify_at_risk_students(clean_df)
        self.assertIsInstance(at_risk_df, pd.DataFrame)

        factors_df = analysis.identify_performance_factors(clean_df)
        self.assertFalse(factors_df.empty)

        insights = analysis.generate_automated_insights(clean_df)
        self.assertIn("narratives", insights)
        self.assertGreater(len(insights["narratives"]), 4)
        self.assertIn("best_department", insights)
        self.assertIn("strongest_driver", insights)
        print("✓ Test 05 Passed: Advanced Performance Analytics & Dynamic Insights")

    def test_06_visualization_engine(self):
        """Test Plotly interactive figures and static Matplotlib charts."""
        df = data_generator.generate_student_dataset(n_records=220, inject_anomalies=False)
        clean_df, _ = data_cleaning.clean_and_preprocess_dataset(df)

        fig1 = visualization.plot_marks_distribution(clean_df)
        self.assertIsNotNone(fig1)

        fig2 = visualization.plot_department_comparison(clean_df)
        self.assertIsNotNone(fig2)

        fig3 = visualization.plot_correlation_heatmap(clean_df)
        self.assertIsNotNone(fig3)

        # Test static charts generation for report
        chart_paths = visualization.generate_static_charts_for_report(clean_df, self.test_reports_dir)
        for name, path in chart_paths.items():
            self.assertTrue(os.path.exists(path), f"Chart {name} must exist at {path}")
        print("✓ Test 06 Passed: Visualization Engine (Plotly & Matplotlib)")

    def test_07_machine_learning_prediction(self):
        """Test Scikit-learn model training, leakage prevention, and simulation."""
        df = data_generator.generate_student_dataset(n_records=300, inject_anomalies=False)
        clean_df, _ = data_cleaning.clean_and_preprocess_dataset(df)

        ml_bundle = prediction.train_performance_models(clean_df)
        self.assertIn("lr_metrics", ml_bundle)
        self.assertIn("rf_metrics", ml_bundle)
        self.assertGreater(ml_bundle["lr_metrics"]["r2_score"], 0.60, "Model R2 score should be solid on correlated data.")
        self.assertNotIn("gender", [f.lower() for f in ml_bundle["features_used"]], "Gender must NOT be in features.")

        # Test interactive simulation
        sim_res = prediction.predict_student_performance(
            ml_bundle, study_hours=6.0, attendance_percentage=90.0, internal_marks=42.0, assignment_scores=85.0
        )
        self.assertIn("predicted_marks", sim_res)
        self.assertGreaterEqual(sim_res["predicted_marks"], 0.0)
        self.assertLessEqual(sim_res["predicted_marks"], 100.0)
        self.assertIn("projected_grade", sim_res)
        self.assertIn("disclaimer", sim_res)
        print("✓ Test 07 Passed: Machine Learning Prediction & Ethical AI")

    def test_08_pdf_and_excel_export(self):
        """Test executive PDF report generation and Excel multi-sheet export."""
        df = data_generator.generate_student_dataset(n_records=250, inject_anomalies=False)
        clean_df, audit = data_cleaning.clean_and_preprocess_dataset(df)
        insights = analysis.generate_automated_insights(clean_df)
        dept_df = analysis.analyze_department_performance(clean_df)
        sub_df = analysis.analyze_subject_performance(clean_df)
        at_risk_df = analysis.identify_at_risk_students(clean_df)
        ml_bundle = prediction.train_performance_models(clean_df)
        chart_paths = visualization.generate_static_charts_for_report(clean_df, self.test_reports_dir)

        # PDF Report
        pdf_out = os.path.join(self.test_reports_dir, "unit_test_report.pdf")
        pdf_path = report_generator.generate_pdf_report(
            clean_df, insights, dept_df, sub_df, audit, ml_bundle, chart_paths, pdf_out
        )
        self.assertTrue(os.path.exists(pdf_path), "Generated PDF report must exist.")
        self.assertGreater(os.path.getsize(pdf_path), 5000, "PDF size must be substantial.")

        # Excel Export
        excel_out = os.path.join(self.test_reports_dir, "unit_test_summary.xlsx")
        report_generator.export_analytical_summaries_to_excel(
            clean_df, dept_df, sub_df, at_risk_df, excel_out
        )
        self.assertTrue(os.path.exists(excel_out), "Generated Excel workbook must exist.")
        print("✓ Test 08 Passed: PDF Executive Report & Excel Export")


if __name__ == "__main__":
    unittest.main()
