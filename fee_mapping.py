# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:35:53 2026

@author: hp
"""

"""
fee_mapping.py
--------------

Purpose:
    Map primary and secondary school categories to official Annex fee values.

This stage implements the fee inputs used in MTI scoring.

Primary formula input:
    PrimaryFees = Annex fee assigned from PrimaryType

Secondary formula input:
    SecondaryFees = Annex fee assigned from SecondaryType + SecondaryCategory

Sponsorship rule:
    If sponsored:
        NetFees = 0
    Else:
        NetFees = AnnexFees

Important:
    This file does not compute MTI.
    It only prepares official fee inputs for MTI formulas.
"""

import pandas as pd
import numpy as np


# =====================================================
# 1. OFFICIAL ANNEX PRIMARY SCHOOL FEES
# =====================================================

PRIMARY_FEE_MAP = {
    "PUBLIC DAY SCHOOL": 8333,
    "MISSION SCHOOL": 33333,
    "PUBLIC BOARDING SCHOOL": 41667,
    "PRIVATE DAY SCHOOL": 83333,
    "PRIVATE BOARDING SCHOOL": 250000,
    "PRIVATE DAY AND BOARDING": 166667,
    "PUBLIC DAY AND BOARDING": 8333,
}


# =====================================================
# 2. OFFICIAL ANNEX SECONDARY SCHOOL FEES
# =====================================================

SECONDARY_FEE_MAP = {
    ("PUBLIC DAY SCHOOL", "SUB COUNTY"): 8400,
    ("PUBLIC DAY SCHOOL", "COUNTY"): 21808,
    ("PUBLIC DAY SCHOOL", "EXTRA COUNTY"): 29077,
    ("PUBLIC DAY SCHOOL", "NATIONAL"): 31500,

    ("PUBLIC BOARDING SCHOOL", "SUB COUNTY"): 20192,
    ("PUBLIC BOARDING SCHOOL", "COUNTY"): 64672,
    ("PUBLIC BOARDING SCHOOL", "EXTRA COUNTY"): 72780,
    ("PUBLIC BOARDING SCHOOL", "NATIONAL"): 86510,

    ("MISSION SCHOOL", "SUB COUNTY"): 23528,
    ("MISSION SCHOOL", "COUNTY"): 61081,
    ("MISSION SCHOOL", "EXTRA COUNTY"): 81442,
    ("MISSION SCHOOL", "NATIONAL"): 88229,

    ("PRIVATE DAY SCHOOL", "SUB COUNTY"): 47055,
    ("PRIVATE DAY SCHOOL", "COUNTY"): 122163,
    ("PRIVATE DAY SCHOOL", "EXTRA COUNTY"): 162884,
    ("PRIVATE DAY SCHOOL", "NATIONAL"): 176457,

    ("PRIVATE BOARDING SCHOOL", "SUB COUNTY"): 113114,
    ("PRIVATE BOARDING SCHOOL", "COUNTY"): 362281,
    ("PRIVATE BOARDING SCHOOL", "EXTRA COUNTY"): 407698,
    ("PRIVATE BOARDING SCHOOL", "NATIONAL"): 484615,
}


# =====================================================
# 3. NORMALIZATION HELPERS
# =====================================================

def normalize_primary_type(value):
    """
    Convert raw PrimaryType into Annex-compatible category.

    Output categories must match PRIMARY_FEE_MAP keys.
    """

    x = str(value).upper().strip()

    if "PRIVATE DAY AND BOARDING" in x:
        return "PRIVATE DAY AND BOARDING"

    if "PUBLIC DAY AND BOARDING" in x:
        return "PUBLIC DAY AND BOARDING"

    if "PUBLIC DAY" in x:
        return "PUBLIC DAY SCHOOL"

    if "PUBLIC BOARDING" in x:
        return "PUBLIC BOARDING SCHOOL"

    if "PRIVATE DAY" in x:
        return "PRIVATE DAY SCHOOL"

    if "PRIVATE BOARDING" in x:
        return "PRIVATE BOARDING SCHOOL"

    if "MISSION" in x:
        return "MISSION SCHOOL"

    return np.nan


def normalize_secondary_type(value):
    """
    Convert raw SecondaryType into Annex-compatible school type.

    Output categories must match the first element of SECONDARY_FEE_MAP keys.
    """

    x = str(value).upper().strip()

    if "PUBLIC DAY" in x:
        return "PUBLIC DAY SCHOOL"

    if "PUBLIC BOARDING" in x:
        return "PUBLIC BOARDING SCHOOL"

    if "PRIVATE DAY" in x:
        return "PRIVATE DAY SCHOOL"

    if "PRIVATE BOARDING" in x:
        return "PRIVATE BOARDING SCHOOL"

    if "MISSION" in x:
        return "MISSION SCHOOL"

    return np.nan


def normalize_secondary_category(value):
    """
    Convert raw SecondaryCategory into Annex-compatible category.

    Output categories:
        SUB COUNTY
        COUNTY
        EXTRA COUNTY
        NATIONAL
    """

    x = str(value).upper().strip()

    if x in ["NAN", "NONE", "", "PRIMARY"]:
        return np.nan

    if "SUB" in x:
        return "SUB COUNTY"

    if "EXTRA" in x:
        return "EXTRA COUNTY"

    if "NATIONAL" in x:
        return "NATIONAL"

    if "COUNTY" in x:
        return "COUNTY"

    return np.nan


def yes_flag(series):
    """
    Convert sponsorship-like variables into boolean flags.

    Accepts:
        YES, Y, 1, TRUE, SPONSORED
    """

    return series.astype(str).str.upper().isin(
        ["YES", "Y", "1", "TRUE", "SPONSORED"]
    )


# =====================================================
# 4. APPLY FEE MAPPING
# =====================================================

def apply_fee_mapping(df):
    """
    Apply Annex fee mappings to the cleaned application dataset.

    Creates:
        PrimaryType_clean
        SecondaryType_clean
        SecondaryCategory_clean
        PrimaryFees
        SecondaryFees
        SponsoredPrimary_flag
        SponsoredSecondary_flag
        NetPrimaryFees
        NetSecondaryFees

    Secondary category issue:
        Some records have valid SecondaryType but missing SecondaryCategory.
        Since Annex fees require both fields, missing categories are filled using
        the dominant observed category within each school type.
    """

    df = df.copy()

    # ---------------------------------------------
    # 1. Normalize school labels
    # ---------------------------------------------
    df["PrimaryType_clean"] = df["PrimaryType"].apply(normalize_primary_type)
    df["SecondaryType_clean"] = df["SecondaryType"].apply(normalize_secondary_type)
    df["SecondaryCategory_clean"] = df["SecondaryCategory"].apply(
        normalize_secondary_category
    )

    # ---------------------------------------------
    # 2. Map primary fees directly
    # ---------------------------------------------
    df["PrimaryFees"] = df["PrimaryType_clean"].map(PRIMARY_FEE_MAP)

    # ---------------------------------------------
    # 3. Fill missing secondary categories structurally
    # ---------------------------------------------
    known_categories = df[df["SecondaryCategory_clean"].notna()].copy()

    if not known_categories.empty:
        dominant_category_by_type = (
            known_categories
            .groupby("SecondaryType_clean")["SecondaryCategory_clean"]
            .agg(lambda x: x.value_counts().idxmax())
        )

        missing_category = df["SecondaryCategory_clean"].isna()

        df.loc[missing_category, "SecondaryCategory_clean"] = (
            df.loc[missing_category, "SecondaryType_clean"]
            .map(dominant_category_by_type)
        )

    # ---------------------------------------------
    # 4. Map secondary fees strictly from Annex
    # ---------------------------------------------
    df["SecondaryFees"] = df.apply(
        lambda row: SECONDARY_FEE_MAP.get(
            (row["SecondaryType_clean"], row["SecondaryCategory_clean"]),
            np.nan
        ),
        axis=1
    )

    # ---------------------------------------------
    # 5. Sponsorship flags
    # ---------------------------------------------
    df["SponsoredPrimary_flag"] = yes_flag(df["SponsoredPrimary"])
    df["SponsoredSecondary_flag"] = yes_flag(df["SponsoredSecondary"])

    # ---------------------------------------------
    # 6. Net fees
    # ---------------------------------------------
    df["NetPrimaryFees"] = np.where(
        df["SponsoredPrimary_flag"],
        0,
        df["PrimaryFees"]
    )

    df["NetSecondaryFees"] = np.where(
        df["SponsoredSecondary_flag"],
        0,
        df["SecondaryFees"]
    )

    return df


# =====================================================
# 5. DIAGNOSTICS
# =====================================================

def fee_mapping_diagnostics(df):
    """
    Return key checks after fee mapping.

    Expected:
        primary_missing = 0
        secondary_missing = 0

    If either is not zero, MTI reconstruction should not proceed.
    """

    diagnostics = {
        "primary_missing": int(df["PrimaryFees"].isna().sum()),
        "secondary_missing": int(df["SecondaryFees"].isna().sum()),
        "primary_types": int(df["PrimaryType_clean"].nunique(dropna=True)),
        "secondary_types": int(df["SecondaryType_clean"].nunique(dropna=True)),
        "secondary_categories": int(df["SecondaryCategory_clean"].nunique(dropna=True)),
    }

    return diagnostics


def show_unmapped_fees(df, n=30):
    """
    Display unmapped fee categories for debugging.
    """

    primary_unmapped = (
        df[df["PrimaryFees"].isna()]
        [["PrimaryType", "PrimaryType_clean"]]
        .value_counts()
        .reset_index(name="count")
        .head(n)
    )

    secondary_unmapped = (
        df[df["SecondaryFees"].isna()]
        [
            "SecondaryType",
            "SecondaryCategory",
            "SecondaryType_clean",
            "SecondaryCategory_clean"
        ]
        .value_counts()
        .reset_index(name="count")
        .head(n)
    )

    return primary_unmapped, secondary_unmapped