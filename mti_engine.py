"""mti_engine.py
Excel-replication MTI engine with the full policy flow:
core components -> MTI_baseline -> equity adjustment -> income adjustment -> MTI_final.
"""

import numpy as np
import pandas as pd


# =====================================================
# SAFETY HELPERS
# =====================================================

def _safe_section(policy, key):
    val = policy.get(key) if isinstance(policy, dict) else None
    return val if isinstance(val, dict) else {}


def _series(df, col, default=0):
    if isinstance(df, pd.DataFrame) and col in df.columns:
        return df[col]
    return pd.Series(default, index=df.index)


def _numeric_series(df, col, default=0):
    return pd.to_numeric(_series(df, col, default), errors="coerce").fillna(default)


def _clean_key(series):
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .str.replace("-", "_", regex=False)
        .str.replace(" ", "_", regex=False)
    )


def _yes(series):
    return series.astype(str).str.strip().str.upper().isin(
        ["YES", "Y", "TRUE", "1", "SPONSORED", "DISABLED", "FEMALE"]
    )


def _first_existing_flag(df, columns):
    flag = pd.Series(False, index=df.index)
    for col in columns:
        if col in df.columns:
            flag = flag | _yes(df[col])
    return flag


def _apply_gap_adjustment(score, alpha):
    return (score + alpha * (100.0 - score)).clip(0, 100)


# =====================================================
# HOUSEHOLD SIZE
# =====================================================

def compute_household_size(df):
    df = df.copy()

    if "FamilySize" in df.columns:
        df["HouseholdSize"] = pd.to_numeric(df["FamilySize"], errors="coerce").fillna(1).clip(lower=1)
        return df

    family_structure = _clean_key(_series(df, "FamilyStructure", ""))
    parents = pd.Series(2, index=df.index, dtype=float)
    parents.loc[family_structure.isin(["ORPHANED", "ORPHAN", "NO_PARENTS", "ABANDONED", "DECEASED_SINGLE"])] = 0
    parents.loc[family_structure.isin(["SINGLE_MOTHER", "SINGLE_FATHER", "ONE_PARENT_DECEASED"])] = 1

    children = _numeric_series(df, "NumberofChildren", 0).clip(0, 20)
    higher = _numeric_series(df, "SiblingsHigherEduc", 0).clip(0, 10)

    df["HouseholdSize"] = (parents + 1 + children + higher).clip(lower=1)
    return df


# =====================================================
# CORE SCORES
# =====================================================

def compute_primary_score(df, policy):
    df = df.copy()
    weights = _safe_section(policy, "weights")
    thresholds = _safe_section(policy, "thresholds")

    w = float(weights.get("primary", 29.5))
    benchmark = float(policy.get("fee_benchmark", 150000))
    threshold = float(thresholds.get("primary_poverty", 39.8))

    poverty = _numeric_series(df, "PovertyIndex", 0).clip(0, 100)
    sponsored = _yes(_series(df, "SponsoredPrimary", "No"))
    fees = _numeric_series(df, "NetPrimaryFees", np.nan)
    fees = fees.where(fees.notna(), _numeric_series(df, "PrimaryFees", 0)).clip(lower=0)

    ppi_factor = np.minimum(1.0, poverty / max(threshold, 1e-9))
    affordability = np.maximum(0.0, 1.0 - (fees / max(benchmark, 1e-9)))
    score = np.where(sponsored, w, affordability * w * ppi_factor)

    df["S_primary"] = pd.Series(score, index=df.index).clip(0, w)
    return df


def compute_secondary_score(df, policy):
    df = df.copy()
    cfg = _safe_section(policy, "secondary_score")
    weights = _safe_section(policy, "weights")

    method = str(cfg.get("method", "linear")).lower()
    w = float(cfg.get("max_score", weights.get("secondary", 24.8)))
    benchmark = float(cfg.get("fee_benchmark", policy.get("fee_benchmark", 150000)))

    fees = _numeric_series(df, "NetSecondaryFees", np.nan)
    fees = fees.where(fees.notna(), _numeric_series(df, "SecondaryFees", 0)).clip(lower=0)
    sponsored = _yes(_series(df, "SponsoredSecondary", "No"))

    if method == "exponential":
        lam = float(cfg.get("decay_lambda", 3e-5))
        score = w * np.exp(-lam * fees)
    else:
        score = np.maximum(0.0, 1.0 - (fees / max(benchmark, 1e-9))) * w

    score = np.where(sponsored, w, score)
    df["S_secondary"] = pd.Series(score, index=df.index).clip(0, w)
    return df


def compute_poverty_score(df, policy):
    df = df.copy()
    cfg = _safe_section(policy, "poverty_score")
    weights = _safe_section(policy, "weights")
    thresholds = _safe_section(policy, "thresholds")

    method = str(cfg.get("method", "linear")).lower()
    w = float(cfg.get("max_score", weights.get("poverty", 24.8)))
    poverty = _numeric_series(df, "PovertyIndex", 0).clip(0, 100)

    if method == "logistic":
        midpoint = float(cfg.get("midpoint", 0.40))
        steepness = float(cfg.get("steepness", 10.0))
        P = poverty / 100.0
        score = w / (1.0 + np.exp(-steepness * (P - midpoint)))
    else:
        threshold = float(cfg.get("threshold", thresholds.get("poverty_score", 60.0)))
        score = np.minimum(1.0, poverty / max(threshold, 1e-9)) * w

    df["S_poverty"] = pd.Series(score, index=df.index).clip(0, w)
    return df


def compute_family_score(df, policy):
    df = df.copy()
    fam = _safe_section(policy, "family_scores")
    weights = _safe_section(policy, "weights")

    method = str(fam.get("method", "log")).lower()
    w = float(fam.get("max_score", weights.get("family", 20.9)))
    fmax = max(float(fam.get("fmax", 10)), 2.0)
    F = pd.to_numeric(df["HouseholdSize"], errors="coerce").fillna(1).clip(lower=1)

    if method == "band":
        score = pd.Series(float(fam.get("small", 9.3)), index=df.index)
        score.loc[F.between(4, 6, inclusive="both")] = float(fam.get("medium", 16.4))
        score.loc[F >= 7] = float(fam.get("large", 20.9))
    elif method == "linear":
        score = w * np.minimum(1.0, F / fmax)
    else:
        score = w * np.minimum(1.0, np.log(F) / np.log(fmax))

    df["S_family"] = pd.Series(score, index=df.index).clip(0, w)
    return df


def compute_baseline_mti(df):
    df = df.copy()
    df["MTI_baseline"] = (
        df["S_primary"] + df["S_secondary"] + df["S_poverty"] + df["S_family"]
    ).clip(0, 100)
    return df


# =====================================================
# EQUITY ADJUSTMENT
# =====================================================

def apply_equity_adjustment(df, policy):
    df = df.copy()
    eq = _safe_section(policy, "equity_adjustment")

    df["MTI_before_equity"] = df["MTI_baseline"].clip(0, 100)
    df["MTI_equity"] = df["MTI_before_equity"]
    df["EquityAdjustmentApplied"] = False
    df["EquityAlphaMax"] = 0.0

    if not eq.get("enabled", True):
        return df

    score = df["MTI_equity"].copy()

    # 1. Family structure based adjustment. This captures orphan, one-parent and abandoned cases.
    family_key = _clean_key(_series(df, "FamilyStructure", ""))
    fs_map = eq.get("family_structure_alpha", {}) if isinstance(eq.get("family_structure_alpha", {}), dict) else {}
    normalized_map = {str(k).strip().upper().replace("-", "_").replace(" ", "_"): float(v) for k, v in fs_map.items()}
    alpha_fs = family_key.map(normalized_map).fillna(0.0).clip(0, 1)
    mask = alpha_fs > 0
    score.loc[mask] = _apply_gap_adjustment(score.loc[mask], alpha_fs.loc[mask])
    df.loc[mask, "EquityAdjustmentApplied"] = True
    df["EquityAlphaMax"] = np.maximum(df["EquityAlphaMax"], alpha_fs)

    # 2. Student disability / NCPWD.
    disability_flag = _first_existing_flag(df, ["NCPWD", "StudentDisability", "Disability", "Student_Disability"])
    alpha = float(eq.get("ncpwd_alpha", 0.50))
    mask = disability_flag & (alpha > 0)
    score.loc[mask] = _apply_gap_adjustment(score.loc[mask], alpha)
    df.loc[mask, "EquityAdjustmentApplied"] = True
    df.loc[mask, "EquityAlphaMax"] = np.maximum(df.loc[mask, "EquityAlphaMax"], alpha)

    # 3. Parent disability where such columns exist.
    parent_disability_flag = _first_existing_flag(df, ["ParentDisability", "ParentsDisability", "Parent_Disability"])
    alpha = float(eq.get("parent_disability_alpha", 0.50))
    mask = parent_disability_flag & (alpha > 0)
    score.loc[mask] = _apply_gap_adjustment(score.loc[mask], alpha)
    df.loc[mask, "EquityAdjustmentApplied"] = True
    df.loc[mask, "EquityAlphaMax"] = np.maximum(df.loc[mask, "EquityAlphaMax"], alpha)

    # 4. Government cash transfer where such columns exist.
    cash_transfer_flag = _first_existing_flag(df, ["CashTransfer", "GovernmentCashTransfer", "SocialProtectionBeneficiary"])
    alpha = float(eq.get("cash_transfer_alpha", 0.50))
    mask = cash_transfer_flag & (alpha > 0)
    score.loc[mask] = _apply_gap_adjustment(score.loc[mask], alpha)
    df.loc[mask, "EquityAdjustmentApplied"] = True
    df.loc[mask, "EquityAlphaMax"] = np.maximum(df.loc[mask, "EquityAlphaMax"], alpha)

    # 5. Female adjustment is deliberately applied last and modestly.
    gender = _series(df, "Gender", "").astype(str).str.strip().str.upper()
    female_flag = gender.isin(["F", "FEMALE", "WOMAN", "GIRL"])
    alpha = float(eq.get("female_alpha", 0.05))
    mask = female_flag & (alpha > 0)
    score.loc[mask] = _apply_gap_adjustment(score.loc[mask], alpha)
    df.loc[mask, "EquityAdjustmentApplied"] = True
    df.loc[mask, "EquityAlphaMax"] = np.maximum(df.loc[mask, "EquityAlphaMax"], alpha)

    df["MTI_equity"] = score.clip(0, 100)
    df["EquityAdjustmentValue"] = df["MTI_equity"] - df["MTI_before_equity"]
    return df


# =====================================================
# INCOME ADJUSTMENT
# =====================================================

def apply_income_adjustment(df, policy):
    df = df.copy()
    inc = _safe_section(policy, "income_adjustment")

    base_col = "MTI_equity" if "MTI_equity" in df.columns else "MTI_baseline"
    df["MTI_before_income"] = df[base_col].clip(0, 100)

    if not inc.get("enabled", True):
        df["Income_adjustment_ratio"] = 0.0
        df["Income_adjusted"] = df["MTI_before_income"]
        df["MTI_final"] = df["Income_adjusted"]
        if inc.get("round_final_mti", True):
            df["MTI_final"] = df["MTI_final"].round(0)
        return df

    T = float(inc.get("threshold", 399996))
    k = max(float(inc.get("k", 15)), 1.000001)
    lam = float(inc.get("lambda", 0.20))
    curve = str(inc.get("curve", "smoothstep")).lower()

    income = _numeric_series(df, "VerifiedAnnualHouseholdIncome", 0).clip(lower=0)
    size = pd.to_numeric(df["HouseholdSize"], errors="coerce").fillna(1).replace(0, 1)
    df["IncomePerHousehold"] = income / size

    z = ((df["IncomePerHousehold"] - T) / ((k - 1.0) * T)).clip(0, 1)
    if curve == "linear":
        ratio = z
    else:
        ratio = 3 * z**2 - 2 * z**3

    if inc.get("exclude_equity_adjusted", False):
        ratio = ratio.mask(df.get("EquityAdjustmentApplied", False), 0.0)

    df["Income_adjustment_ratio"] = ratio
    df["Income_adjusted"] = (df["MTI_before_income"] * (1.0 - lam * ratio)).clip(0, 100)
    df["MTI_final"] = df["Income_adjusted"]
    if inc.get("round_final_mti", True):
        df["MTI_final"] = df["MTI_final"].round(0)

    return df


# =====================================================
# MASTER FUNCTION
# =====================================================

def compute_mti(df, policy):
    if policy is None:
        raise ValueError("Policy is None")
    if df is None or len(df) == 0:
        raise ValueError("Empty dataframe passed to compute_mti")

    df = compute_household_size(df)
    df = compute_primary_score(df, policy)
    df = compute_secondary_score(df, policy)
    df = compute_poverty_score(df, policy)
    df = compute_family_score(df, policy)
    df = compute_baseline_mti(df)
    df = apply_equity_adjustment(df, policy)
    df = apply_income_adjustment(df, policy)
    return df
