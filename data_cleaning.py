# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:31:01 2026

@author: hp
"""

"""
data_cleaning.py
----------------

Purpose:
    Prepare raw intake data for MTI computation.

This stage is purely structural:
    - Clean text fields
    - Convert numeric fields
    - Construct verified income
    - Define programme type (TVET vs University)
    - Define affordability constant (C)

NO MTI formulas are applied here.

Why this matters:
    If this step is wrong, the MTI engine will be wrong even if formulas are correct.
"""

import pandas as pd
import numpy as np

def safe_series(df, col, default=0):
    """
    Return df[col] if it exists, otherwise a Series filled with default.
    Prevents errors like: int object has no attribute fillna.
    """
    if isinstance(df, pd.DataFrame) and col in df.columns:
        return df[col]
    return pd.Series(default, index=df.index)




# =====================================================
# TEXT CLEANING
# =====================================================

def clean_text(series):
    """
    Standardize categorical variables.

    Ensures:
        - consistent uppercase
        - no trailing spaces
        - unified formatting for mapping (fee mapping depends on this)
    """
    return (
        series.astype(str)
        .str.upper()
        .str.strip()
        .str.replace("-", " ", regex=False)
        .str.replace("_", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})
    )


# =====================================================
# MAIN CLEANING FUNCTION
# =====================================================

def clean_application_data(df, policy):
    """
    Clean raw application dataset.

    INPUT:
        df: raw dataframe
        policy: config dictionary

    OUTPUT:
        cleaned dataframe ready for:
            -> fee mapping
            -> MTI scoring

    Core transformations:

    1. Numeric conversion
    2. Verified income construction
    3. Household structure preparation
    4. Programme classification
    5. Affordability constant (C)
    """

    df = df.copy()
    df.columns = df.columns.str.strip()

    # ---------------------------------------------
    # 1. CLEAN TEXT VARIABLES
    # ---------------------------------------------
    for col in df.select_dtypes(include="object").columns:
        df[col] = clean_text(df[col])

    # ---------------------------------------------
    # 2. NUMERIC VARIABLES
    # ---------------------------------------------
    numeric_cols = [
        "KRAIncomeEmploymentCombined",
        "KRAIncomeBusinessCombined",
        "SelfDeclaredIncomeCombined",
        "NumberofChildren",
        "SiblingsHigherEduc",
        "PovertyIndex",
        "ProgramCost",
        "MTIScore"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---------------------------------------------
    # 3. VERIFIED INCOME (CRITICAL)
    # ---------------------------------------------
    """
    VerifiedAnnualHouseholdIncome =
        KRA employment income + KRA business income

    This is the ONLY income used for MTI adjustment.
    Self-declared income is ignored (unreliable).
    """

    df["KRAIncomeEmploymentCombined"] = safe_series(df, "KRAIncomeEmploymentCombined", 0).fillna(0)
    df["KRAIncomeBusinessCombined"] = safe_series(df, "KRAIncomeBusinessCombined", 0).fillna(0)

    df["VerifiedAnnualHouseholdIncome"] = (
        df["KRAIncomeEmploymentCombined"]
        + df["KRAIncomeBusinessCombined"]
    )

    # ---------------------------------------------
    # 4. BOUND KEY VARIABLES
    # ---------------------------------------------
    """
    These bounds prevent unrealistic values from distorting MTI.
    """

    df["NumberofChildren"] = safe_series(df, "NumberofChildren", 0).fillna(0).clip(0, 10)
    df["SiblingsHigherEduc"] = safe_series(df, "SiblingsHigherEduc", 0).fillna(0).clip(0, 5)
    df["PovertyIndex"] = safe_series(df, "PovertyIndex", 0).fillna(0).clip(0, 100)

    # ---------------------------------------------
    # 5. PROGRAM COST
    # ---------------------------------------------
    if "ProgramCost" in df.columns:
        df["ProgramCost"] = df["ProgramCost"].fillna(df["ProgramCost"].median())
    else:
        df["ProgramCost"] = policy["university_cap"]

    # ---------------------------------------------
    # 6. EXISTING MTI (FOR COMPARISON ONLY)
    # ---------------------------------------------
    if "MTIScore" in df.columns:
        df["MTIScore"] = pd.to_numeric(df["MTIScore"], errors="coerce").clip(0, 100)

    # ---------------------------------------------
    # 7. STUDY LEVEL CLASSIFICATION
    # ---------------------------------------------
    """
    Identify TVET vs University.

    This determines:
        - affordability constant (C)
        - allocation formulas
    """

    df["StudyLevel_clean"] = df.get("StudyLevel", "").astype(str).str.upper()

    df["is_tvet"] = df["StudyLevel_clean"].str.contains(
        "TVET|CERTIFICATE|DIPLOMA", na=False
    ).astype(int)

    # ---------------------------------------------
    # 8. AFFORDABILITY CONSTANT (C)
    # ---------------------------------------------
    """
    C is used in MTI formulas:

        S_primary ∝ (C - PrimaryFees)/C
        S_secondary ∝ (C - SecondaryFees)/C

    University: 150,000
    TVET:       67,189
    """

    df["C_affordability"] = np.where(
        df["is_tvet"] == 1,
        policy["tvet_cost"],
        policy["university_cap"]
    )

    return df


# =====================================================
# DIAGNOSTICS
# =====================================================

def cleaning_diagnostics(df):
    """
    Quick checks before moving to MTI stage.
    """

    return {
        "rows": len(df),
        "missing_program_cost": int(df["ProgramCost"].isna().sum()),
        "missing_poverty_index": int(df["PovertyIndex"].isna().sum()),
        "zero_income_count": int((df["VerifiedAnnualHouseholdIncome"] == 0).sum()),
        "tvet_count": int(df["is_tvet"].sum()),
    }
