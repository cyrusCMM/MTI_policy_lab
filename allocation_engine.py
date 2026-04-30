"""allocation_engine.py
MTI allocation engine for HH, SS, LL and upkeep.

University baseline defaults are Excel-replication defaults. The explicit
HH formula switch supports both programme-cost share mode and fixed-cap curve
mode without changing downstream reporting identities.
"""

import numpy as np
import pandas as pd


def _safe_section(policy, key):
    val = policy.get(key) if isinstance(policy, dict) else None
    return val if isinstance(val, dict) else {}


def _num(series, default=0):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _mti_series(df, mti_col):
    if mti_col in df.columns:
        return _num(df[mti_col], 0).clip(0, 100)
    return pd.Series(0.0, index=df.index)


def _program_cost(df, policy):
    if "ProgramCost" in df.columns:
        return _num(df["ProgramCost"], policy.get("university_cap", 150000)).clip(lower=0)
    return pd.Series(float(policy.get("university_cap", 150000)), index=df.index)


def compute_excel_common_allocation(df, policy, mti_col="MTI_final"):
    df = df.copy()
    p = _safe_section(policy, "university_allocation")

    mti = _mti_series(df, mti_col)
    x = (mti / 100.0).clip(0, 1)
    ability = 1.0 - x
    PC = _program_cost(df, policy)

    hh_formula_mode = str(p.get("hh_formula_mode", "program_cost_share")).lower()
    hh_base_share = float(p.get("hh_base_share", 0.10))
    hh_ability_share = float(p.get("hh_ability_share", 0.30))
    hh_cap = float(p.get("hh_cap", 150000))
    hh_discount = float(p.get("hh_discount", 0.90))

    df["PC_allocation"] = PC
    df["HH_formula_mode"] = hh_formula_mode

    if hh_formula_mode == "program_cost_share":
        hh_raw = (hh_base_share * PC) + (hh_ability_share * PC * ability)
        df["HH_raw"] = hh_raw
        df["HH"] = np.minimum(hh_cap, hh_raw)
    elif hh_formula_mode == "fixed_cap_curve":
        hh_raw = hh_cap * (1.0 - hh_discount * x)
        df["HH_raw"] = hh_raw
        df["HH"] = np.minimum(hh_raw, PC)
    else:
        raise ValueError(
            "hh_formula_mode must be 'program_cost_share' or 'fixed_cap_curve'. "
            f"Got: {hh_formula_mode}"
        )

    df["HH"] = pd.Series(df["HH"], index=df.index).clip(lower=0, upper=PC)
    df["FinancingGap"] = (PC - df["HH"]).clip(lower=0)

    ss_base = float(p.get("ss_base_share", 0.15))
    ss_need = float(p.get("ss_need_share", 0.30))
    df["SS_gap_share"] = (ss_base + ss_need * x).clip(0, 1)
    df["SS"] = (df["FinancingGap"] * df["SS_gap_share"]).clip(lower=0)

    # Residual loan enforces the tuition identity row-wise.
    df["LL"] = (PC - df["HH"] - df["SS"]).clip(lower=0)

    # Recompute SS after clipping LL edge cases, so HH + SS + LL = PC exactly.
    over = df["HH"] + df["SS"] > PC
    if over.any():
        df.loc[over, "SS"] = (PC.loc[over] - df.loc[over, "HH"]).clip(lower=0)
        df.loc[over, "LL"] = 0.0

    upkeep_intercept = float(p.get("upkeep_intercept", 0))
    upkeep_coefficient = float(p.get("upkeep_coefficient", 0))
    df["Upkeep"] = (upkeep_intercept + upkeep_coefficient * x).clip(lower=0)
    df["TotalLoan_with_Upkeep"] = df["LL"] + df["Upkeep"]

    return df


def compute_tvet_allocation(df, policy, mti_col="MTI_final"):
    df = df.copy()
    p = _safe_section(policy, "tvet_allocation")

    x = (_mti_series(df, mti_col) / 100.0).clip(0, 1)
    PC = _num(df["ProgramCost"], policy.get("tvet_cost", 67189)).clip(lower=0)

    df["PC_allocation"] = PC
    df["HH"] = PC * (float(p.get("hh_base", 0.10)) - float(p.get("hh_slope", 0.30)) * x)
    df["HH"] = df["HH"].clip(lower=0, upper=PC)
    df["FinancingGap"] = (PC - df["HH"]).clip(lower=0)

    df["SS_gap_share"] = (float(p.get("ss_base", 0.15)) + float(p.get("ss_slope", 0.30)) * x).clip(0, 1)
    df["SS"] = (df["FinancingGap"] * df["SS_gap_share"]).clip(lower=0)
    df["LL"] = (PC - df["HH"] - df["SS"]).clip(lower=0)

    df["Upkeep"] = (float(p.get("upkeep_base", 0)) + float(p.get("upkeep_slope", 0)) * x).clip(lower=0)
    df["TotalLoan_with_Upkeep"] = df["LL"] + df["Upkeep"]
    return df


def compute_allocations(df, policy, mti_col="MTI_final", hard_check=True):
    if policy is None:
        raise ValueError("Policy is None")
    if df is None or len(df) == 0:
        raise ValueError("Empty dataframe passed to compute_allocations")

    mode = policy.get("allocation_mode", "excel_common")
    if mode == "tvet":
        out = compute_tvet_allocation(df, policy, mti_col)
    else:
        out = compute_excel_common_allocation(df, policy, mti_col)

    out["TuitionIdentityCheck"] = out["HH"] + out["SS"] + out["LL"] - out["PC_allocation"]
    max_error = float(out["TuitionIdentityCheck"].abs().max())
    if hard_check and max_error > 1e-6:
        raise ValueError(f"Allocation identity broken: {max_error}")
    return out
