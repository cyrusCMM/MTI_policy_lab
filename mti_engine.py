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


def _series(df, col, default=0):
    if isinstance(df, pd.DataFrame) and col in df.columns:
        return df[col]
    return pd.Series(default, index=df.index)


def _numeric_series(df, col, default=0):
    return pd.to_numeric(_series(df, col, default), errors="coerce").fillna(default)


def _bool_series(df, col, default=False):
    s = _series(df, col, default)
    if getattr(s, "dtype", None) == bool:
        return s.fillna(default).astype(bool)
    ss = s.astype(str).str.upper().str.strip()
    return ss.isin(["1", "TRUE", "YES", "Y", "T"])


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
    HouseholdSize = Parents + Applicant(1) + SchoolGoingSiblings + SiblingsHigherEduc.

    Current-data approximation:
    - Parents inferred from FamilyStructure.
    - SchoolGoingSiblings from NumberofChildren unless verified/self-declared sibling fields exist.
    - SiblingsHigherEduc from SiblingsHigherEduc.
    """
    df = df.copy()

    if "FamilyStructure" not in df.columns:
        df["FamilyStructure"] = ""

    df["ParentsCount"] = df["FamilyStructure"].apply(infer_parents)

    if "NEMISVerifiedSiblings" in df.columns and "SelfDeclaredSiblings" in df.columns:
        school_siblings = np.minimum(
            _numeric_series(df, "NEMISVerifiedSiblings", 0),
            _numeric_series(df, "SelfDeclaredSiblings", 0),
        )
    else:
        school_siblings = _numeric_series(df, "NumberofChildren", 0)

    df["SchoolGoingSiblings"] = pd.to_numeric(school_siblings, errors="coerce").fillna(0).clip(0, 20)
    df["SiblingsHigherEduc"] = _numeric_series(df, "SiblingsHigherEduc", 0).clip(0, 10)

    df["HouseholdSize"] = (
        df["ParentsCount"] + 1 + df["SchoolGoingSiblings"] + df["SiblingsHigherEduc"]
    ).clip(lower=1)

    return df


# =====================================================
# 2. PRIMARY SCORE
# =====================================================

def compute_primary_score(df, policy):
    """
    Primary score:
    NetPrimaryFees = max(0, PrimaryFees - PrimaryArrears)
    PovertyFactorPrimary = min(1, PovertyProbability / theta_p), with verified overrides set to 1.
    S_primary = ((C - NetPrimaryFees) / C) * PovertyFactorPrimary * 29.5
    C is programme-specific: 150,000 for university and 67,189 for TVET.
    """
    df = df.copy()
    weights = policy["weights"]
    thresholds = policy["thresholds"]

    if "PovertyProbability" not in df.columns:
        df["PovertyProbability"] = _numeric_series(df, "PovertyIndex", 0).clip(0, 100) / 100
    else:
        df["PovertyProbability"] = pd.to_numeric(df["PovertyProbability"], errors="coerce").fillna(0).clip(0, 1)

    theta = max(float(thresholds.get("primary_poverty", 0.40)), 1e-9)
    df["PovertyFactorPrimary"] = np.minimum(1.0, df["PovertyProbability"] / theta)

    sponsored_primary = _bool_series(df, "SponsoredPrimary_flag", False) | _bool_series(df, "SponsoredPrimary", False)
    sponsored_secondary = _bool_series(df, "SponsoredSecondary_flag", False) | _bool_series(df, "SponsoredSecondary", False)
    primary_arrears = _numeric_series(df, "PrimaryArrears", 0).clip(lower=0)
    placement_downgrade = _bool_series(df, "SecondaryPlacementDowngrade_flag", False) | _bool_series(df, "SecondaryPlacementDowngrade", False)
    school_feeding = _bool_series(df, "SchoolFeedingPrimary_flag", False) | _bool_series(df, "SchoolFeedingPrimary", False)

    override = (primary_arrears > 0) | sponsored_primary | sponsored_secondary | placement_downgrade | school_feeding
    df.loc[override, "PovertyFactorPrimary"] = 1.0

    if "PrimaryFees" in df.columns:
        primary_fees = np.maximum(_numeric_series(df, "PrimaryFees", 0), _numeric_series(df, "NetPrimaryFees", 0))
    else:
        primary_fees = _numeric_series(df, "NetPrimaryFees", 0)

    df["NetPrimaryFees"] = (primary_fees - primary_arrears).clip(lower=0)
    df.loc[sponsored_primary, "NetPrimaryFees"] = 0

    if "C_affordability" in df.columns:
        C = pd.to_numeric(df["C_affordability"], errors="coerce")
    else:
        is_tvet = _numeric_series(df, "is_tvet", 0).eq(1)
        C = pd.Series(np.where(is_tvet, policy.get("tvet_cost", 67189), policy.get("university_cap", 150000)), index=df.index)
    C = C.replace(0, np.nan).fillna(policy.get("university_cap", 150000))

    affordability_ratio = ((C - df["NetPrimaryFees"]) / C).clip(0, 1)
    w = float(weights.get("primary", 29.5))
    df["S_primary"] = (affordability_ratio * df["PovertyFactorPrimary"] * w).clip(0, w).fillna(0)
    return df


# =====================================================
# 3. SECONDARY SCORE
# =====================================================

def compute_secondary_score(df, policy):
    """
    Secondary score:
    NetSecondaryFees = max(0, SecondaryFees - SecondaryArrears); sponsored secondary => 0.
    S_secondary = max_score * exp(-decay_lambda * NetSecondaryFees)
    """
    df = df.copy()
    cfg = policy.get("secondary_score", {})
    max_score = float(cfg.get("max_score", policy.get("weights", {}).get("secondary", 24.8)))
    decay_lambda = float(cfg.get("decay_lambda", 3.0e-5))

    arrears = _numeric_series(df, "SecondaryArrears", 0).clip(lower=0)
    sponsored = _bool_series(df, "SponsoredSecondary_flag", False) | _bool_series(df, "SponsoredSecondary", False)

    if "SecondaryFees" in df.columns:
        fees = np.maximum(_numeric_series(df, "SecondaryFees", 0), _numeric_series(df, "NetSecondaryFees", 0))
    else:
        fees = _numeric_series(df, "NetSecondaryFees", 0)

    df["NetSecondaryFees"] = (fees - arrears).clip(lower=0)
    df.loc[sponsored, "NetSecondaryFees"] = 0

    df["S_secondary"] = max_score * np.exp(-decay_lambda * df["NetSecondaryFees"])
    df["S_secondary"] = pd.to_numeric(df["S_secondary"], errors="coerce").fillna(0).clip(0, max_score)
    return df


# =====================================================
# 4. POVERTY SCORE
# =====================================================

def compute_poverty_score(df, policy):
    """
    Poverty score:
    S_poverty = max_score / (1 + exp(-k * (P - P0)))
    """
    df = df.copy()
    if "PovertyProbability" not in df.columns:
        df["PovertyProbability"] = _numeric_series(df, "PovertyIndex", 0).clip(0, 100) / 100
    else:
        df["PovertyProbability"] = pd.to_numeric(df["PovertyProbability"], errors="coerce").fillna(0).clip(0, 1)

    cfg = policy.get("poverty_score", {})
    max_score = float(cfg.get("max_score", policy.get("weights", {}).get("poverty", 24.8)))
    midpoint = float(cfg.get("midpoint", policy.get("thresholds", {}).get("poverty_score", 0.40)))
    steepness = float(cfg.get("steepness", 10.0))

    P = df["PovertyProbability"].clip(0, 1)
    df["S_poverty"] = max_score / (1 + np.exp(-steepness * (P - midpoint)))
    df["S_poverty"] = pd.to_numeric(df["S_poverty"], errors="coerce").fillna(0).clip(0, max_score)
    return df


# =====================================================
# 5. FAMILY SCORE
# =====================================================

def compute_family_score(df, policy):
    """
    Family score:
    S_family = max_score * min(1, ln(F) / ln(Fmax))
    Default max_score = 20.9, Fmax = 10.
    """
    df = df.copy()
    fam = policy.get("family_scores", {})
    method = str(fam.get("method", "log")).lower()
    max_score = float(fam.get("max_score", policy.get("weights", {}).get("family", 20.9)))
    fmax = max(float(fam.get("fmax", 10)), 2.0)
    F = pd.to_numeric(df["HouseholdSize"], errors="coerce").fillna(1).clip(lower=1)

    if method == "linear":
        score = max_score * np.minimum(1.0, F / fmax)
    elif method == "band":
        score = np.select(
            [F <= 3, (F >= 4) & (F <= 6), F >= 7],
            [float(fam.get("small", 9.3)), float(fam.get("medium", 16.4)), float(fam.get("large", 20.9))],
            default=float(fam.get("small", 9.3)),
        )
    else:
        score = max_score * np.minimum(1.0, np.log(F) / np.log(fmax))

    df["S_family"] = pd.to_numeric(score, errors="coerce").fillna(0).clip(0, max_score)
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
    """
    Apply equity adjustments sequentially:
    orphan -> one parent -> student disability -> parent disability -> cash transfer -> female.
    """
    df = df.copy()
    eq = policy["equity_adjustment"]

    df["MTI_after_equity"] = df["MTI_baseline"]
    df["equity_applied"] = 0

    for col in ["orphan", "one_parent", "ncpwd", "parent_disability", "cash_transfer", "female"]:
        df[f"equity_{col}"] = 0

    if not eq.get("enabled", False):
        df["MTI_after_equity"] = df["MTI_after_equity"].clip(0, 100)
        return df

    gender = _clean_str(_series(df, "Gender", ""))
    family_structure = _series(df, "FamilyStructure", "")
    ncpwd = _clean_str(_series(df, "NCPWD", ""))

    female_mask = gender.eq("FEMALE") | gender.eq("F")
    one_parent_mask = is_one_parent_series(family_structure)
    ncpwd_mask = ncpwd.isin(["YES", "Y", "TRUE", "1", "NCPWD"])
    orphan_mask = is_orphan_series(family_structure)
    parent_disability = _bool_series(df, "ParentDisability", False) | _bool_series(df, "ParentDisability_flag", False)
    cash_transfer = _bool_series(df, "CashTransferBeneficiary", False) | _bool_series(df, "CashTransferBeneficiary_flag", False)

    df = _apply_gap_adjustment(df, orphan_mask, float(eq.get("orphan_alpha", 0)), "orphan")
    df = _apply_gap_adjustment(df, one_parent_mask, float(eq.get("one_parent_alpha", 0)), "one_parent")
    df = _apply_gap_adjustment(df, ncpwd_mask, float(eq.get("ncpwd_alpha", 0)), "ncpwd")
    df = _apply_gap_adjustment(df, parent_disability, float(eq.get("parent_disability_alpha", 0.50)), "parent_disability")
    df = _apply_gap_adjustment(df, cash_transfer, float(eq.get("cash_transfer_alpha", 0.50)), "cash_transfer")
    df = _apply_gap_adjustment(df, female_mask, float(eq.get("female_alpha", 0)), "female")

    df["MTI_after_equity"] = df["MTI_after_equity"].clip(0, 100)
    return df


# =====================================================
# 8. INCOME ADJUSTMENT
# =====================================================

def apply_income_adjustment(df, policy):
    """
    Income adjustment:
    IPH = VerifiedAnnualHouseholdIncome / HouseholdSize
    Z = min(1, max(0, (IPH - T_IPH) / ((k - 1) * T_IPH)))
    ratio = 3Z^2 - 2Z^3
    MTI_final = M * [1 - lambda * ratio]
    """
    df = df.copy()
    inc = policy["income_adjustment"]

    df["MTI_final"] = df["MTI_after_equity"]
    df["IncomeAdjustmentRatio"] = 0.0
    df["IncomeSmoothZ"] = 0.0
    df["income_adjustment_applied"] = 0

    if not inc.get("enabled", False):
        return df

    household_size = pd.to_numeric(df["HouseholdSize"], errors="coerce").replace(0, np.nan)
    income = _numeric_series(df, "VerifiedAnnualHouseholdIncome", 0)
    df["IPH"] = income / household_size

    T = float(inc.get("threshold", 399996))
    k = float(inc.get("k", 15))
    lam = float(inc.get("lambda", 0.20))

    if T <= 0:
        raise ValueError("Income threshold T_IPH must be positive.")
    if k <= 1:
        raise ValueError("Income scaling factor k must be greater than 1.")
    if not (0 <= lam <= 1):
        raise ValueError("Income lambda must be between 0 and 1.")

    Z = ((df["IPH"] - T) / ((k - 1) * T)).clip(0, 1).fillna(0)
    ratio = Z if str(inc.get("curve", "smoothstep")).lower() == "linear" else 3 * (Z ** 2) - 2 * (Z ** 3)

    df["IncomeSmoothZ"] = Z
    df["IncomeAdjustmentRatio"] = ratio

    mask = income > 0
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
