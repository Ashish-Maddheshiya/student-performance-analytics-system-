"""
=============================================================================
Student Performance Data Analytics System - Exploratory Data Analysis (EDA)
=============================================================================
Purpose:
    Performs comprehensive statistical, distributional, and structural
    Exploratory Data Analysis:
    - Dataset dimensions, memory footprint, column catalog
    - Missingness profile and completeness index
    - Parametric and non-parametric central tendency: Mean, Median, Mode
    - Dispersion metrics: Standard Deviation, Variance, Min, Max, Range, IQR
    - Shape analytics: Skewness and Kurtosis
    - Categorical frequency distribution breakdowns
=============================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List


def get_dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Summarize dataset dimensions, memory consumption, and schema characteristics.
    """
    total_cells = df.shape[0] * df.shape[1]
    total_nulls = int(df.isnull().sum().sum())
    completeness = round(((total_cells - total_nulls) / total_cells * 100), 2) if total_cells > 0 else 0.0

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "total_nulls": total_nulls,
        "completeness_percentage": completeness,
        "memory_usage_kb": round(df.memory_usage(deep=True).sum() / 1024, 2),
        "duplicates": int(df.duplicated().sum())
    }


def get_data_types_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a tabular summary of columns, data types, non-null counts, and unique values.
    """
    summary_data = []
    for col in df.columns:
        non_null_count = int(df[col].notnull().sum())
        null_count = int(df[col].isnull().sum())
        null_pct = round((null_count / len(df) * 100), 2) if len(df) > 0 else 0.0
        unique_count = int(df[col].nunique())
        sample_val = str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "None"

        summary_data.append({
            "Column Name": col,
            "Data Type": str(df[col].dtype),
            "Non-Null Count": non_null_count,
            "Null Count": null_count,
            "Null (%)": null_pct,
            "Unique Values": unique_count,
            "Sample Value": sample_val
        })
    return pd.DataFrame(summary_data)


def get_statistical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rigorous statistical metrics across all numerical columns:
    Mean, Median, Mode, Std Dev, Variance, Min, Max, Range, Q1, Q3, IQR, Skewness, Kurtosis.
    """
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        return pd.DataFrame()

    stats_list = []
    for col in num_df.columns:
        s = num_df[col].dropna()
        if len(s) == 0:
            continue

        mean_val = float(s.mean())
        median_val = float(s.median())
        mode_series = s.mode()
        mode_val = float(mode_series.iloc[0]) if not mode_series.empty else mean_val
        std_val = float(s.std(ddof=1)) if len(s) > 1 else 0.0
        var_val = float(s.var(ddof=1)) if len(s) > 1 else 0.0
        min_val = float(s.min())
        max_val = float(s.max())
        val_range = max_val - min_val
        q1_val = float(s.quantile(0.25))
        q3_val = float(s.quantile(0.75))
        iqr_val = q3_val - q1_val
        skew_val = float(s.skew()) if len(s) > 2 else 0.0
        kurt_val = float(s.kurtosis()) if len(s) > 3 else 0.0

        stats_list.append({
            "Metric / Feature": col,
            "Mean": round(mean_val, 2),
            "Median": round(median_val, 2),
            "Mode": round(mode_val, 2),
            "Std Dev": round(std_val, 2),
            "Variance": round(var_val, 2),
            "Min": round(min_val, 2),
            "25% (Q1)": round(q1_val, 2),
            "50% (Median)": round(median_val, 2),
            "75% (Q3)": round(q3_val, 2),
            "Max": round(max_val, 2),
            "Range": round(val_range, 2),
            "IQR": round(iqr_val, 2),
            "Skewness": round(skew_val, 2),
            "Kurtosis": round(kurt_val, 2)
        })

    return pd.DataFrame(stats_list).set_index("Metric / Feature")


def get_categorical_breakdown(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Produce categorical frequency tables with counts and relative percentages.
    """
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    breakdowns = {}

    for col in cat_cols:
        counts = df[col].value_counts(dropna=False).reset_index()
        counts.columns = [col, "Count"]
        counts["Percentage (%)"] = (counts["Count"] / len(df) * 100).round(2)
        breakdowns[col] = counts

    return breakdowns


def get_missing_value_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detailed audit of missing values per column.
    """
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "Missing Count": null_counts,
        "Missing Percentage (%)": null_pct
    })
    return missing_df[missing_df["Missing Count"] > 0].sort_values(
        by="Missing Count", ascending=False
    )
