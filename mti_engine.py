# -*- coding: utf-8 -*-
"""
mti_engine_policy_aligned.py
----------------------------
Policy-aligned MTI engine for FY2026/2027.

Strict sequence:
1. Household size
2. Primary score
3. Secondary score
4. Poverty score
5. Family score
6. Baseline MTI
7. Equity adjustment
8. Income adjustment

Important policy safeguards:
- Existing MTIScore is not used.
- Equity application is tracked per condition.
- Income adjustment can exclude equity-adjusted students, as required by the
  MTI document.
- Optional unavailable levers are only used if columns exist; otherwise the
  mapped-fee/sponsorship version is used transparently.
"""

import numpy as np
import pandas as pd


def _clean_str(series):
    return series.astype(str).str.upper().str.strip()


# =====================================================
# 1. HOUSEHOLD STRUCTURE
# =====================================================

def infer_parents(family_structure):
    """Infer number of parents from FamilyStructure."""
    fs = str(family_structure).upper()

    orphan_terms = ["ORPHAN", "NO PARENT", "NO PARENTS", "BOTH", "ABANDON"]
    one_parent_terms = ["SINGLE", "ONE", "WIDOW", "WIDOWER", "MOTHER ONLY", "FATHER ONLY"]

    if any(term in fs for term in orphan_terms):
        return 0
    if any(term in fs for term in one_parent_terms):
        return 1
    return 2


def is_orphan_series(series):
    fs = _clean_str(series)
    return (
        fs.str.contains("ORPHAN", na=False)
        | fs.str.contains("NO PARENT", na=False)
        | fs.str.contains("NO PARENTS", na=False)
        | fs.str.contains("BOTH", na=False)
        | fs.str.contains("ABANDON", na=False)
    )


def is_one_parent_series(series):
    fs = _clean_str(series)
    orphan = is_orphan_series(series)
    one_parent = (
        fs.str.contains("SINGLE", na=False)
        | fs.str.contains("ONE", na=False)
        | fs.str.contains("WIDOW", na=False)
        | fs.str.contains("WIDOWER", na=False)
        | fs.str.contains("MOTHER ONLY", na=False)
        | fs.str.contains("FATHER ONLY", na=False)
    )
    return one_parent & (~orphan)


def compute_household_size(df):
    """
    HouseholdSize = Parents + Applicant(1) + NumberofChildren + SiblingsHigherEduc.
    """
    df = df.copy()

    for col in ["FamilyStructure", "NumberofChildren", "SiblingsHigherEduc"]:
        if col not in df.columns:
            if col == "FamilyStructure":
                df[col] = ""
            else:
                df[col] = 0

    df["ParentsCount"] = df["FamilyStructure"].apply(infer_parents)
    df["NumberofChildren"] = pd.to_numeric(df["NumberofChildren"], errors="coerce").fillna(0).clip(0, 10)
    df["SiblingsHigherEduc"] = pd.to_numeric(df["SiblingsHigherEduc"], errors="coerce").fillna(0).clip(0, 5)

    df["HouseholdSize"] = (
        df["ParentsCount"]
        + 1
        + df["NumberofChildren"]
        + df["SiblingsHigherEduc"]
    )

    return df


# =====================================================
# 2. PRIMARY SCORE
# =====================================================

def compute_primary_score(df, policy):
    """
    S_primary = ((C - NetPrimaryFees) / C) * PovertyFactorPrimary * primary_weight

    PovertyFactorPrimary = min(1, PovertyProbability / primary_poverty_threshold)

    Available current data supports mapped fees and sponsorship. Additional
    overrides are applied only if corresponding columns exist.
    """
    df = df.copy()

    weights = policy["weights"]
    thresholds = policy["thresholds"]

    df["PovertyIndex"] = pd.to_numeric(df.get("PovertyIndex", 0), errors="coerce").fillna(0).clip(0, 100)
    df["PovertyProbability"] = df["PovertyIndex"] / 100

    primary_threshold = max(float(thresholds["primary_poverty"]), 1e-9)
    df["PovertyFactorPrimary"] = np.minimum(1, df["PovertyProbability"] / primary_threshold)

    sponsored_primary = df.get("SponsoredPrimary_flag", False)
    sponsored_secondary = df.get("SponsoredSecondary_flag", False)
    sponsored_primary = sponsored_primary if isinstance(sponsored_primary, pd.Series) else pd.Series(False, index=df.index)
    sponsored_secondary = sponsored_secondary if isinstance(sponsored_secondary, pd.Series) else pd.Series(False, index=df.index)

    # Optional policy overrides if future columns are present.
    override = sponsored_primary.fillna(False) | sponsored_secondary.fillna(False)
    for optional_col in [
        "PrimaryArrears_flag",
        "SchoolFeedingPrimary_flag",
        "SecondaryPlacementDowngrade_flag",
    ]:
        if optional_col in df.columns:
            override = override | df[optional_col].fillna(False).astype(bool)

    df.loc[override, "PovertyFactorPrimary"] = 1

    df["C_affordability"] = pd.to_numeric(df["C_affordability"], errors="coerce").replace(0, np.nan)
    df["NetPrimaryFees"] = pd.to_numeric(df["NetPrimaryFees"], errors="coerce")

    affordability_ratio = (df["C_affordability"] - df["NetPrimaryFees"]) / df["C_affordability"]

    df["S_primary"] = (
        affordability_ratio
        * df["PovertyFactorPrimary"]
        * weights["primary"]
    ).clip(0, weights["primary"])

    df["S_primary"] = df["S_primary"].fillna(0)
    return df


# =====================================================
# 3. SECONDARY SCORE
# =====================================================

def compute_secondary_score(df, policy):
    """S_secondary = ((C - NetSecondaryFees) / C) * secondary_weight."""
    df = df.copy()
    w = policy["weights"]["secondary"]

    df["C_affordability"] = pd.to_numeric(df["C_affordability"], errors="coerce").replace(0, np.nan)
    df["NetSecondaryFees"] = pd.to_numeric(df["NetSecondaryFees"], errors="coerce")

    affordability_ratio = (df["C_affordability"] - df["NetSecondaryFees"]) / df["C_affordability"]

    df["S_secondary"] = (affordability_ratio * w).clip(0, w)
    df["S_secondary"] = df["S_secondary"].fillna(0)
    return df


# =====================================================
# 4. POVERTY SCORE
# =====================================================

def compute_poverty_score(df, policy):
    """S_poverty = min(1, PovertyProbability / poverty_threshold) * poverty_weight."""
    df = df.copy()

    w = policy["weights"]["poverty"]
    threshold = max(float(policy["thresholds"]["poverty_score"]), 1e-9)

    if "PovertyProbability" not in df.columns:
        df["PovertyProbability"] = pd.to_numeric(df.get("PovertyIndex", 0), errors="coerce").fillna(0).clip(0, 100) / 100

    df["S_poverty"] = (np.minimum(1, df["PovertyProbability"] / threshold) * w).clip(0, w)
    df["S_poverty"] = df["S_poverty"].fillna(0)
    return df


# =====================================================
# 5. FAMILY SCORE
# =====================================================

def compute_family_score(df, policy):
    """
    Family score.

    Default policy:
        S_family = max_score * min(1, HouseholdSize / Fmax)

    This replaces the older step-band approach and gives a smooth,
    transparent marginal increase for each additional household member.
    """
    df = df.copy()
    fam = policy.get("family_scores", {})
    method = str(fam.get("method", "linear")).lower()

    household_size = pd.to_numeric(df["HouseholdSize"], errors="coerce").fillna(1).clip(lower=1)

    if method == "linear":
        max_score = float(fam.get("max_score", policy.get("weights", {}).get("family", 20.9)))
        fmax = max(float(fam.get("fmax", 7)), 1.0)
        df["S_family"] = max_score * np.minimum(1.0, household_size / fmax)
    elif method == "log":
        max_score = float(fam.get("max_score", policy.get("weights", {}).get("family", 20.9)))
        fmax = max(float(fam.get("fmax", 7)), 2.0)
        df["S_family"] = max_score * np.minimum(1.0, np.log(household_size) / np.log(fmax))
    else:
        df["S_family"] = np.select(
            [
                household_size <= 3,
                (household_size >= 4) & (household_size <= 6),
                household_size >= 7,
            ],
            [
                float(fam.get("small", 9.3)),
                float(fam.get("medium", 16.4)),
                float(fam.get("large", 20.9)),
            ],
            default=float(fam.get("small", 9.3)),
        )

    max_allowed = float(fam.get("max_score", policy.get("weights", {}).get("family", 20.9)))
    df["S_family"] = pd.to_numeric(df["S_family"], errors="coerce").fillna(0).clip(0, max_allowed)
    return df



# =====================================================
# 6. BASELINE MTI
# =====================================================

def compute_baseline_mti(df):
    """MTI_baseline = S_primary + S_secondary + S_poverty + S_family."""
    df = df.copy()
    df["MTI_baseline"] = (
        df["S_primary"] + df["S_secondary"] + df["S_poverty"] + df["S_family"]
    ).clip(0, 100)
    return df


# =====================================================
# 7. EQUITY ADJUSTMENT
# =====================================================

def _apply_gap_adjustment(df, mask, alpha, condition_name):
    """Apply M_new = M_old + alpha(100 - M_old) to masked rows."""
    if alpha <= 0:
        return df

    mask = mask.fillna(False).astype(bool)
    if not mask.any():
        return df

    before = df.loc[mask, "MTI_after_equity"]
    df.loc[mask, "MTI_after_equity"] = before + alpha * (100 - before)
    df.loc[mask, "equity_applied"] = 1
    df.loc[mask, f"equity_{condition_name}"] = 1
    return df


def apply_equity_adjustment(df, policy):
    """Apply verified-equity adjustments sequentially and track application."""
    df = df.copy()
    eq = policy["equity_adjustment"]

    df["MTI_after_equity"] = df["MTI_baseline"]
    df["equity_applied"] = 0

    for col in ["female", "one_parent", "ncpwd", "orphan"]:
        df[f"equity_{col}"] = 0

    if not eq.get("enabled", False):
        df["MTI_after_equity"] = df["MTI_after_equity"].clip(0, 100)
        return df

    gender = _clean_str(df.get("Gender", pd.Series("", index=df.index)))
    family_structure = df.get("FamilyStructure", pd.Series("", index=df.index))
    ncpwd = _clean_str(df.get("NCPWD", pd.Series("", index=df.index)))

    female_mask = gender.eq("FEMALE") | gender.eq("F")
    one_parent_mask = is_one_parent_series(family_structure)
    ncpwd_mask = ncpwd.isin(["YES", "Y", "TRUE", "1", "NCPWD"])
    orphan_mask = is_orphan_series(family_structure)

    # Sequential application: each adjustment acts on the remaining gap.
    df = _apply_gap_adjustment(df, female_mask, float(eq.get("female_alpha", 0)), "female")
    df = _apply_gap_adjustment(df, one_parent_mask, float(eq.get("one_parent_alpha", 0)), "one_parent")
    df = _apply_gap_adjustment(df, ncpwd_mask, float(eq.get("ncpwd_alpha", 0)), "ncpwd")
    df = _apply_gap_adjustment(df, orphan_mask, float(eq.get("orphan_alpha", 0)), "orphan")

    df["MTI_after_equity"] = df["MTI_after_equity"].clip(0, 100)
    return df


# =====================================================
# 8. INCOME ADJUSTMENT
# =====================================================

def apply_income_adjustment(df, policy):
    """
    Income adjustment.

    M = MTI after baseline and equity adjustment.
    T_IPH = income per household-member threshold.
    k = income scaling factor.
    U_IPH = k * T_IPH.
    z = min(1, max(0, (IPH - T_IPH) / ((k - 1) * T_IPH)))
    adjustment_ratio = 3z^2 - 2z^3
    MTI_final = M * [1 - lambda * adjustment_ratio]
    """
    df = df.copy()
    inc = policy["income_adjustment"]

    df["MTI_final"] = df["MTI_after_equity"]
    df["IncomeAdjustmentRatio"] = 0.0
    df["IncomeSmoothZ"] = 0.0
    df["income_adjustment_applied"] = 0

    if not inc.get("enabled", False):
        return df

    df["HouseholdSize"] = pd.to_numeric(df["HouseholdSize"], errors="coerce").replace(0, np.nan)
    df["VerifiedAnnualHouseholdIncome"] = pd.to_numeric(df.get("VerifiedAnnualHouseholdIncome", 0), errors="coerce").fillna(0)

    df["IPH"] = df["VerifiedAnnualHouseholdIncome"] / df["HouseholdSize"]

    T = float(inc["threshold"])
    k = float(inc["k"])
    lam = float(inc["lambda"])

    if T <= 0:
        raise ValueError("Income threshold T_IPH must be positive.")
    if k <= 1:
        raise ValueError("Income scaling factor k must be greater than 1.")
    if not (0 <= lam <= 1):
        raise ValueError("Income lambda must be between 0 and 1.")

    z = ((df["IPH"] - T) / ((k - 1) * T)).clip(0, 1).fillna(0)
    if str(inc.get("curve", "smoothstep")).lower() == "linear":
        adjustment_ratio = z
    else:
        adjustment_ratio = 3 * (z ** 2) - 2 * (z ** 3)

    df["IncomeSmoothZ"] = z
    df["IncomeAdjustmentRatio"] = adjustment_ratio

    mask = df["VerifiedAnnualHouseholdIncome"] > 0
    if inc.get("exclude_equity_adjusted", True):
        mask = mask & (df.get("equity_applied", 0).fillna(0).astype(int) == 0)

    df.loc[mask, "MTI_final"] = df.loc[mask, "MTI_after_equity"] * (1 - lam * df.loc[mask, "IncomeAdjustmentRatio"])
    df.loc[mask & (df["IncomeAdjustmentRatio"] > 0), "income_adjustment_applied"] = 1

    df["MTI_final"] = df["MTI_final"].clip(0, 100)
    return df



# =====================================================
# MASTER FUNCTION
# =====================================================

def compute_mti(df, policy):
    """Run the full MTI pipeline."""
    df = compute_household_size(df)
    df = compute_primary_score(df, policy)
    df = compute_secondary_score(df, policy)
    df = compute_poverty_score(df, policy)
    df = compute_family_score(df, policy)
    df = compute_baseline_mti(df)
    df = apply_equity_adjustment(df, policy)
    df = apply_income_adjustment(df, policy)
    return df
