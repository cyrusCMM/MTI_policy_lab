# -*- coding: utf-8 -*-
"""
fee_mapping_policy_aligned.py
-----------------------------
Policy-aligned school-fee mapping for MTI scoring.

Current data supports school-type/category mapping and sponsorship flags.
Optional declared-fee or arrears columns are used only when present, so the
engine remains accurate for the current dataset and ready for future data.
"""

import numpy as np
import pandas as pd


PRIMARY_FEE_MAP = {
    "PUBLIC DAY SCHOOL": 8333,
    "MISSION SCHOOL": 33333,
    "PUBLIC BOARDING SCHOOL": 41667,
    "PRIVATE DAY SCHOOL": 83333,
    "PRIVATE BOARDING SCHOOL": 250000,
    "PRIVATE DAY AND BOARDING": 166667,
    "PUBLIC DAY AND BOARDING": 8333,
}

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


def normalize_primary_type(value):
    x = str(value).upper().strip()
    if x in ["NAN", "NONE", ""]:
        return np.nan
    if "PRIVATE" in x and "DAY" in x and "BOARD" in x:
        return "PRIVATE DAY AND BOARDING"
    if "PUBLIC" in x and "DAY" in x and "BOARD" in x:
        return "PUBLIC DAY AND BOARDING"
    if "PUBLIC" in x and "DAY" in x:
        return "PUBLIC DAY SCHOOL"
    if "PUBLIC" in x and "BOARD" in x:
        return "PUBLIC BOARDING SCHOOL"
    if "PRIVATE" in x and "DAY" in x:
        return "PRIVATE DAY SCHOOL"
    if "PRIVATE" in x and "BOARD" in x:
        return "PRIVATE BOARDING SCHOOL"
    if "MISSION" in x:
        return "MISSION SCHOOL"
    return np.nan


def normalize_secondary_type(value):
    x = str(value).upper().strip()
    if x in ["NAN", "NONE", ""]:
        return np.nan
    if "PUBLIC" in x and "DAY" in x:
        return "PUBLIC DAY SCHOOL"
    if "PUBLIC" in x and "BOARD" in x:
        return "PUBLIC BOARDING SCHOOL"
    if "PRIVATE" in x and "DAY" in x:
        return "PRIVATE DAY SCHOOL"
    if "PRIVATE" in x and "BOARD" in x:
        return "PRIVATE BOARDING SCHOOL"
    if "MISSION" in x:
        return "MISSION SCHOOL"
    return np.nan


def normalize_secondary_category(value):
    x = str(value).upper().strip().replace("-", " ")
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
    if not isinstance(series, pd.Series):
        return pd.Series(False)
    return series.astype(str).str.upper().str.strip().isin(["YES", "Y", "1", "TRUE", "SPONSORED"])


def first_existing_numeric(df, candidates):
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def first_existing_flag(df, candidates):
    for col in candidates:
        if col in df.columns:
            return yes_flag(df[col])
    return pd.Series(False, index=df.index)


def apply_fee_mapping(df):
    df = df.copy()

    for col in ["PrimaryType", "SecondaryType", "SecondaryCategory", "SponsoredPrimary", "SponsoredSecondary"]:
        if col not in df.columns:
            df[col] = np.nan

    df["PrimaryType_clean"] = df["PrimaryType"].apply(normalize_primary_type)
    df["SecondaryType_clean"] = df["SecondaryType"].apply(normalize_secondary_type)
    df["SecondaryCategory_clean"] = df["SecondaryCategory"].apply(normalize_secondary_category)

    df["PrimaryFees_mapped"] = df["PrimaryType_clean"].map(PRIMARY_FEE_MAP)
    df["PrimaryFees_declared"] = first_existing_numeric(df, [
        "DeclaredPrimaryFees",
        "PrimaryDeclaredFees",
        "PrimarySchoolFeesDeclared",
        "PrimaryFeesDeclared",
    ])
    df["PrimaryFees"] = pd.concat([df["PrimaryFees_mapped"], df["PrimaryFees_declared"]], axis=1).max(axis=1, skipna=True)

    known = df[df["SecondaryCategory_clean"].notna()].copy()
    if not known.empty:
        dominant = known.groupby("SecondaryType_clean")["SecondaryCategory_clean"].agg(lambda x: x.value_counts().idxmax())
        missing = df["SecondaryCategory_clean"].isna()
        df.loc[missing, "SecondaryCategory_clean"] = df.loc[missing, "SecondaryType_clean"].map(dominant)

    df["SecondaryFees_mapped"] = df.apply(
        lambda row: SECONDARY_FEE_MAP.get((row["SecondaryType_clean"], row["SecondaryCategory_clean"]), np.nan),
        axis=1,
    )
    df["SecondaryFees_declared"] = first_existing_numeric(df, [
        "DeclaredSecondaryFees",
        "SecondaryDeclaredFees",
        "SecondarySchoolFeesDeclared",
        "SecondaryFeesDeclared",
    ])
    df["SecondaryFees"] = pd.concat([df["SecondaryFees_mapped"], df["SecondaryFees_declared"]], axis=1).max(axis=1, skipna=True)

    df["SponsoredPrimary_flag"] = yes_flag(df["SponsoredPrimary"])
    df["SponsoredSecondary_flag"] = yes_flag(df["SponsoredSecondary"])

    df["PrimaryArrears_flag"] = first_existing_flag(df, ["PrimaryArrears", "PrimaryFeesArrears", "HasPrimaryArrears"])
    df["SecondaryArrears_flag"] = first_existing_flag(df, ["SecondaryArrears", "SecondaryFeesArrears", "HasSecondaryArrears"])
    df["SchoolFeedingPrimary_flag"] = first_existing_flag(df, ["SchoolFeedingPrimary", "PrimarySchoolFeeding", "SchoolMealsProgramme"])
    df["SecondaryPlacementDowngrade_flag"] = first_existing_flag(df, ["SecondaryPlacementDowngrade", "PlacementDowngrade"])

    # Net fee rule: sponsored students get net fee = 0. Arrears do not reduce
    # fee directly in current formula; they trigger poverty-factor override.
    df["NetPrimaryFees"] = np.where(df["SponsoredPrimary_flag"], 0, df["PrimaryFees"])
    df["NetSecondaryFees"] = np.where(df["SponsoredSecondary_flag"], 0, df["SecondaryFees"])

    return df


def fee_mapping_diagnostics(df):
    return {
        "primary_missing": int(df["PrimaryFees"].isna().sum()),
        "secondary_missing": int(df["SecondaryFees"].isna().sum()),
        "primary_types": int(df["PrimaryType_clean"].nunique(dropna=True)),
        "secondary_types": int(df["SecondaryType_clean"].nunique(dropna=True)),
        "secondary_categories": int(df["SecondaryCategory_clean"].nunique(dropna=True)),
        "declared_primary_fee_column_used": bool(df["PrimaryFees_declared"].notna().any()) if "PrimaryFees_declared" in df.columns else False,
        "declared_secondary_fee_column_used": bool(df["SecondaryFees_declared"].notna().any()) if "SecondaryFees_declared" in df.columns else False,
        "primary_arrears_available": bool(df["PrimaryArrears_flag"].any()) if "PrimaryArrears_flag" in df.columns else False,
        "secondary_arrears_available": bool(df["SecondaryArrears_flag"].any()) if "SecondaryArrears_flag" in df.columns else False,
    }


def show_unmapped_fees(df, n=30):
    primary_unmapped = (
        df[df["PrimaryFees"].isna()][["PrimaryType", "PrimaryType_clean"]]
        .value_counts()
        .reset_index(name="count")
        .head(n)
    )
    secondary_unmapped = (
        df[df["SecondaryFees"].isna()][["SecondaryType", "SecondaryCategory", "SecondaryType_clean", "SecondaryCategory_clean"]]
        .value_counts()
        .reset_index(name="count")
        .head(n)
    )
    return primary_unmapped, secondary_unmapped
