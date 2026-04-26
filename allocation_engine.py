"""
allocation_engine.py
--------------------

Purpose:
    Convert MTI scores into funding allocations while guaranteeing:

        HH + SS + LL = PC

Main rule:
    HH and SS are computed directly.
    LL is computed as the residual.

This makes the aggregate policy outputs reliable:
    Total HH = sum(HH)
    Total SS = sum(SS)
    Total LL = sum(LL)
    Total PC = sum(PC_allocation)
"""

import numpy as np
import pandas as pd


def compute_university_allocation(df, policy, mti_col="MTI_final"):
    """
    University allocation.

    x = MTI / 100

    HH = min{HH_intercept + HH_coefficient*x, PC}

    R = PC - HH

    SS = R * (SS_intercept + SS_coefficient*x)

    LL = R - SS

    Upkeep = Upkeep_intercept + Upkeep_coefficient*x
    """

    df = df.copy()
    p = policy["university_allocation"]

    x = df[mti_col].clip(0, 100) / 100
    PC = df["ProgramCost"]

    if p["hh_intercept_mode"] == "programme_cost":
        hh_intercept = PC
    else:
        hh_intercept = p["hh_intercept_amount"]

    df["HH"] = hh_intercept + p["hh_coefficient"] * x
    df["HH"] = np.minimum(df["HH"], PC)
    df["HH"] = df["HH"].clip(lower=0)

    df["FinancingGap"] = PC - df["HH"]

    df["SS_gap_share_raw"] = (
        p["ss_intercept"] + p["ss_coefficient"] * x
    )

    df["SS_gap_share"] = df["SS_gap_share_raw"].clip(0, 1)

    df["SS"] = df["FinancingGap"] * df["SS_gap_share"]

    # CRITICAL: loan is residual
    df["LL"] = df["FinancingGap"] - df["SS"]
    df["LL"] = df["LL"].clip(lower=0)

    df["LL_gap_share"] = np.where(
        df["FinancingGap"] > 0,
        df["LL"] / df["FinancingGap"],
        0
    )

    df["Upkeep"] = (
        p["upkeep_intercept"] + p["upkeep_coefficient"] * x
    ).clip(lower=0)

    df["PC_allocation"] = PC

    return df


def compute_tvet_allocation(df, policy, mti_col="MTI_final"):
    """
    TVET allocation.

    x = MTI / 100
    PC = 67,189

    HH = PC(0.40 - 0.30x)
    SS = PC(0.15 + 0.40x)
    LL = PC - HH - SS
    Upkeep = 13,600 + 5,000x
    """

    df = df.copy()
    p = policy["tvet_allocation"]

    x = df[mti_col].clip(0, 100) / 100
    PC = policy["tvet_cost"]

    df["PC_allocation"] = PC

    df["HH"] = PC * (p["hh_base"] - p["hh_slope"] * x)
    df["HH"] = df["HH"].clip(lower=0, upper=PC)

    df["SS_gap_share"] = (
        p["ss_base"] + p["ss_slope"] * x
    ).clip(0, 1)

    df["SS"] = PC * df["SS_gap_share"]

    # CRITICAL: loan is residual
    df["LL"] = PC - df["HH"] - df["SS"]
    df["LL"] = df["LL"].clip(lower=0)

    df["LL_gap_share"] = df["LL"] / PC

    df["FinancingGap"] = PC - df["HH"]

    df["Upkeep"] = (
        p["upkeep_base"] + p["upkeep_slope"] * x
    ).clip(lower=0)

    return df


def compute_allocations(df, policy, mti_col="MTI_final", hard_check=True):
    """
    Apply allocation logic to all students.

    Output columns:
        HH
        SS
        LL
        Upkeep
        FinancingGap
        PC_allocation
        TotalLoan_with_Upkeep
        TuitionIdentityCheck

    Identity:
        HH + SS + LL = PC_allocation
    """

    df = df.copy()

    output_cols = [
        "HH",
        "SS",
        "LL",
        "Upkeep",
        "FinancingGap",
        "PC_allocation",
        "SS_gap_share",
        "LL_gap_share",
    ]

    for col in output_cols:
        df[col] = np.nan

    university_mask = df["is_tvet"] == 0
    tvet_mask = df["is_tvet"] == 1

    if university_mask.any():
        uni_df = compute_university_allocation(
            df.loc[university_mask].copy(),
            policy,
            mti_col=mti_col
        )

        cols = [c for c in output_cols if c in uni_df.columns]
        df.loc[university_mask, cols] = uni_df[cols]

    if tvet_mask.any():
        tvet_df = compute_tvet_allocation(
            df.loc[tvet_mask].copy(),
            policy,
            mti_col=mti_col
        )

        cols = [c for c in output_cols if c in tvet_df.columns]
        df.loc[tvet_mask, cols] = tvet_df[cols]

    df["TotalLoan_with_Upkeep"] = df["LL"] + df["Upkeep"]

    df["TuitionIdentityCheck"] = (
        df["HH"] + df["SS"] + df["LL"] - df["PC_allocation"]
    )

    df["identity_check"] = df["TuitionIdentityCheck"]

    max_error = df["TuitionIdentityCheck"].abs().max()

    if hard_check and max_error > 1e-6:
        raise ValueError(
            f"Allocation identity broken: max error = {max_error}"
        )

    return df


def allocation_aggregate_check(df):
    """
    Confirm policy totals are summing the correct columns.
    """

    total_pc = df["PC_allocation"].sum()
    total_hh = df["HH"].sum()
    total_ss = df["SS"].sum()
    total_ll = df["LL"].sum()

    return pd.DataFrame([{
        "students": len(df),
        "PC_allocation": total_pc,
        "HH": total_hh,
        "SS": total_ss,
        "LL": total_ll,
        "Upkeep": df["Upkeep"].sum(),
        "TotalLoan_with_Upkeep": df["TotalLoan_with_Upkeep"].sum(),
        "HH_plus_SS_plus_LL": total_hh + total_ss + total_ll,
        "aggregate_identity_error": total_hh + total_ss + total_ll - total_pc,
        "max_student_identity_error": df["TuitionIdentityCheck"].abs().max(),
        "HH_share": total_hh / total_pc,
        "SS_share": total_ss / total_pc,
        "LL_share": total_ll / total_pc,
    }])


def allocation_diagnostics(df):
    return {
        "max_identity_error": df["TuitionIdentityCheck"].abs().max(),
        "aggregate_identity_error": (
            df["HH"].sum()
            + df["SS"].sum()
            + df["LL"].sum()
            - df["PC_allocation"].sum()
        ),
        "negative_HH": int((df["HH"] < 0).sum()),
        "negative_SS": int((df["SS"] < 0).sum()),
        "negative_LL": int((df["LL"] < 0).sum()),
        "negative_Upkeep": int((df["Upkeep"] < 0).sum()),
        "missing_HH": int(df["HH"].isna().sum()),
        "missing_SS": int(df["SS"].isna().sum()),
        "missing_LL": int(df["LL"].isna().sum()),
        "missing_Upkeep": int(df["Upkeep"].isna().sum()),
    }