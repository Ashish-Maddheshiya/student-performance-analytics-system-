"""
=============================================================================
Student Performance Data Analytics System - Machine Learning Prediction
=============================================================================
Purpose:
    Implements a responsible, leak-free Scikit-learn predictive model:
    - Features: study_hours, attendance_percentage, internal_marks, assignment_scores
    - Target: final_marks (continuous) and Pass/Fail (binary)
    - Sensitive attributes (gender, age, demographic identity) are STRICTLY EXCLUDED
    - Proper Train-Test Split (80/20) with standard scaling fit ONLY on training set
    - Rigorous regression metrics: R-squared, MAE, MSE, RMSE
    - Interactive "What-If" student performance simulator
    - Clear academic ethics and viva disclaimer
=============================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# Ethical feature selection: purely academic & behavioral drivers
FEATURE_COLUMNS = [
    "study_hours",
    "attendance_percentage",
    "internal_marks",
    "assignment_scores"
]
TARGET_COLUMN = "final_marks"


def train_performance_models(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Train Linear and Random Forest regression models to predict final marks.
    Guarantees zero data leakage and ethical feature selection.
    """
    # 1. Validation & Pre-filtering
    available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    if len(available_features) < 2 or TARGET_COLUMN not in df.columns:
        raise ValueError("Insufficient feature columns for predictive modeling.")

    clean_subset = df[available_features + [TARGET_COLUMN]].dropna()
    if len(clean_subset) < 30:
        raise ValueError("Dataset too small to train a reliable model (minimum 30 records required).")

    X = clean_subset[available_features]
    y = clean_subset[TARGET_COLUMN]

    # 2. Train-Test Split (80/20) with fixed seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # 3. Standardization (fit ONLY on training partition to prevent data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train Models
    # A) Linear Regression (interpretable baseline)
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)

    # B) Random Forest Regressor (non-linear ensemble)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=6)
    rf_model.fit(X_train, y_train)  # Tree-based doesn't require scaling
    y_pred_rf = rf_model.predict(X_test)

    # 5. Evaluate Metrics on Unseen Test Set
    r2_lr = r2_score(y_test, y_pred_lr)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mse_lr)

    r2_rf = r2_score(y_test, y_pred_rf)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mse_rf)

    # Choose best model based on R2
    primary_model = lr_model if r2_lr >= r2_rf else rf_model
    primary_name = "Linear Regression" if primary_model == lr_model else "Random Forest Regressor"

    # Feature Importance / Coefficients
    importance_df = pd.DataFrame({
        "Feature": [f.replace("_", " ").title() for f in available_features],
        "Linear Coefficient (Standardized)": lr_model.coef_.round(3),
        "Random Forest Importance": rf_model.feature_importances_.round(3)
    }).sort_values(by="Random Forest Importance", ascending=False).reset_index(drop=True)

    # Predictions DataFrame for visualization
    test_eval_df = pd.DataFrame({
        "Actual Marks": y_test.values.round(1),
        "Predicted (Linear)": y_pred_lr.round(1),
        "Predicted (Random Forest)": y_pred_rf.round(1),
        "Residual (Linear)": (y_test.values - y_pred_lr).round(1)
    })

    return {
        "primary_model": primary_model,
        "primary_model_name": primary_name,
        "scaler": scaler,
        "features_used": available_features,
        "train_records": len(X_train),
        "test_records": len(X_test),
        "lr_metrics": {
            "r2_score": round(r2_lr, 3),
            "mae": round(mae_lr, 2),
            "mse": round(mse_lr, 2),
            "rmse": round(rmse_lr, 2)
        },
        "rf_metrics": {
            "r2_score": round(r2_rf, 3),
            "mae": round(mae_rf, 2),
            "mse": round(mse_rf, 2),
            "rmse": round(rmse_rf, 2)
        },
        "feature_importance": importance_df,
        "test_evaluation_df": test_eval_df,
        "ethical_statement": (
            "Ethical Model Statement: This predictive model strictly uses behavioral and academic "
            "effort indicators (Study Hours, Attendance, Internal Marks, Assignments). "
            "Sensitive demographic attributes (such as Gender, Age, or Personal Identity) are intentionally "
            "excluded from the feature space to prevent algorithmic bias."
        )
    }


def predict_student_performance(
    model_bundle: Dict[str, Any],
    study_hours: float,
    attendance_percentage: float,
    internal_marks: float,
    assignment_scores: float
) -> Dict[str, Any]:
    """
    Simulate predicted academic performance given student behavioral inputs.
    """
    scaler = model_bundle["scaler"]
    model = model_bundle["primary_model"]
    is_linear = model_bundle["primary_model_name"] == "Linear Regression"

    input_df = pd.DataFrame([{
        "study_hours": float(study_hours),
        "attendance_percentage": float(attendance_percentage),
        "internal_marks": float(internal_marks),
        "assignment_scores": float(assignment_scores)
    }])

    if is_linear:
        input_scaled = scaler.transform(input_df)
        pred_raw = model.predict(input_scaled)[0]
    else:
        pred_raw = model.predict(input_df)[0]

    # Bound marks strictly within [0, 100]
    predicted_marks = round(float(np.clip(pred_raw, 0.0, 100.0)), 1)

    # Determine projected status and grade
    status = "Pass" if predicted_marks >= 40.0 else "Fail"

    if predicted_marks >= 85.0: grade = "A+ (Outstanding)"
    elif predicted_marks >= 75.0: grade = "A (Distinction)"
    elif predicted_marks >= 65.0: grade = "B (First Class)"
    elif predicted_marks >= 50.0: grade = "C (Second Class)"
    elif predicted_marks >= 40.0: grade = "D (Pass Class)"
    else: grade = "F (Fail)"

    # Confidence interval estimate based on model RMSE
    rmse = model_bundle["lr_metrics"]["rmse"] if is_linear else model_bundle["rf_metrics"]["rmse"]
    lower_bound = round(max(0.0, predicted_marks - 1.96 * rmse), 1)
    upper_bound = round(min(100.0, predicted_marks + 1.96 * rmse), 1)

    return {
        "predicted_marks": predicted_marks,
        "projected_status": status,
        "projected_grade": grade,
        "confidence_interval_95": (lower_bound, upper_bound),
        "model_used": model_bundle["primary_model_name"],
        "disclaimer": (
            "Academic Advisory Notice: Predictions generated by this statistical model are heuristic "
            "estimates based on historical aggregate trends. They serve solely as formative guidance and "
            "do not represent definitive examination results or academic determinations."
        )
    }
