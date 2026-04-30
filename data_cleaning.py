# -*- coding: utf-8 -*-
"""
data_cleaning.py
----------------
Prepare raw intake data for Excel-replication MTI computation.
No MTI formula is applied here.
"""

import pandas as pd
import numpy as np


def safe_series(df, col, default=0):
    if isinstance(df, pd.DataFrame) and col in df.columns:
        return df[col]
    return pd.Series(default, index=df.index)


def clean_text(series):
    return (
        series.astype(str)
        .str.strip()
        .str.replace("-", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .replace({"nan": np.nan, "NaN": np.nan, "NAN": np.nan, "NONE": np.nan, "": np.nan})
    )


def clean_application_data(df, policy):
    df = df.copy()
    df.columns = df.columns.str.strip()

    for col in df.select_dtypes(include="object").columns:
        df[col] = clean_text(df[col])

    numeric_cols = [
        "KRAIncomeEmploymentCombined",
        "KRAIncomeBusinessCombined",
        "SelfDeclaredIncomeCombined",
        "Total_Income",
        "NumberofChildren",
        "SiblingsHigherEduc",
        "FamilySize",
        "PovertyIndex",
        "ProgramCost",
        "MTIScore",
        "MTIScore_2025_2026",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["KRAIncomeEmploymentCombined"] = pd.to_numeric(
        safe_series(df, "KRAIncomeEmploymentCombined", 0), errors="coerce"
    ).fillna(0)
    df["KRAIncomeBusinessCombined"] = pd.to_numeric(
        safe_series(df, "KRAIncomeBusinessCombined", 0), errors="coerce"
    ).fillna(0)

    if "Total_Income" in df.columns:
        df["VerifiedAnnualHouseholdIncome"] = pd.to_numeric(df["Total_Income"], errors="coerce").fillna(0)
    else:
        df["VerifiedAnnualHouseholdIncome"] = (
            df["KRAIncomeEmploymentCombined"] + df["KRAIncomeBusinessCombined"]
        )

    df["NumberofChildren"] = pd.to_numeric(safe_series(df, "NumberofChildren", 0), errors="coerce").fillna(0).clip(0, 20)
    df["SiblingsHigherEduc"] = pd.to_numeric(safe_series(df, "SiblingsHigherEduc", 0), errors="coerce").fillna(0).clip(0, 10)
    df["PovertyIndex"] = pd.to_numeric(safe_series(df, "PovertyIndex", 0), errors="coerce").fillna(0).clip(0, 100)

    if "ProgramCost" in df.columns:
        df["ProgramCost"] = pd.to_numeric(df["ProgramCost"], errors="coerce")
        df["ProgramCost"] = df["ProgramCost"].fillna(df["ProgramCost"].median())
    else:
        df["ProgramCost"] = policy["university_cap"]

    if "MTIScore_2025_2026" not in df.columns and "MTIScore" in df.columns:
        df["MTIScore_2025_2026"] = pd.to_numeric(df["MTIScore"], errors="coerce").clip(0, 100)
    elif "MTIScore_2025_2026" in df.columns:
        df["MTIScore_2025_2026"] = pd.to_numeric(df["MTIScore_2025_2026"], errors="coerce").clip(0, 100)

    study = safe_series(df, "StudyLevel", "").astype(str).str.upper()
    programme = safe_series(df, "ProgramDescription", "").astype(str).str.upper()
    df["StudyLevel_clean"] = study
    df["is_tvet"] = (
        study.str.contains("TVET|CERTIFICATE|DIPLOMA|TECHNICAL|VOCATIONAL", na=False)
        | programme.str.contains("TVET|CERTIFICATE|TECHNICAL|VOCATIONAL", na=False)
    ).astype(int)

    df["C_affordability"] = policy.get("fee_benchmark", policy["university_cap"])

    return df


def cleaning_diagnostics(df):
    return {
        "rows": len(df),
        "missing_program_cost": int(df["ProgramCost"].isna().sum()) if "ProgramCost" in df.columns else None,
        "missing_poverty_index": int(df["PovertyIndex"].isna().sum()) if "PovertyIndex" in df.columns else None,
        "zero_income_count": int((df["VerifiedAnnualHouseholdIncome"] == 0).sum()) if "VerifiedAnnualHouseholdIncome" in df.columns else None,
        "tvet_count": int(df["is_tvet"].sum()) if "is_tvet" in df.columns else None,
    }
