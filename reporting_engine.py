# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:09:50 2026

@author: hp
"""

"""
reporting_engine.py
-------------------

Purpose:
    Summarize results after MTI + allocation.

Outputs must support:

1. Student-level (already in dataset)
2. Aggregate-level
3. Institution-level
4. Programme-level
5. County-level
6. MTI distribution diagnostics

This is what makes the model usable for policy decisions.
"""

import pandas as pd
import numpy as np


# =====================================================
# 1. AGGREGATE SUMMARY
# =====================================================

def aggregate_summary(df, label="scenario"):
    """
    Aggregate fiscal outcomes.

    Key identities:
        HH + SS + LL = Programme Cost
        Total Loan = LL + Upkeep
    """

    total_pc = df["PC_allocation"].sum()

    result = {
        "scenario": label,
        "students": len(df),

        # MTI stats
        "mean_mti": df["MTI_final"].mean(),
        "median_mti": df["MTI_final"].median(),

        # Totals
        "program_cost": total_pc,
        "HH": df["HH"].sum(),
        "SS": df["SS"].sum(),
        "LL": df["LL"].sum(),
        "Upkeep": df["Upkeep"].sum(),
        "TotalLoan_with_Upkeep": df["TotalLoan_with_Upkeep"].sum(),

        # Shares
        "HH_share": df["HH"].sum() / total_pc,
        "SS_share": df["SS"].sum() / total_pc,
        "LL_share": df["LL"].sum() / total_pc,

        # Identity check
        "max_identity_error": df["TuitionIdentityCheck"].abs().max(),
    }

    return pd.DataFrame([result])


# =====================================================
# 2. GROUP SUMMARIES
# =====================================================

def group_summary(df, group_col, label="scenario"):
    """
    Generic grouping function.

    Used for:
        - Institution
        - Programme
        - County
    """

    if group_col not in df.columns:
        return pd.DataFrame()

    grouped = (
        df.groupby(group_col, dropna=False)
        .agg(
            students=("MTI_final", "count"),
            mean_mti=("MTI_final", "mean"),
            median_mti=("MTI_final", "median"),

            program_cost=("PC_allocation", "sum"),
            HH=("HH", "sum"),
            SS=("SS", "sum"),
            LL=("LL", "sum"),
            Upkeep=("Upkeep", "sum"),
            total_loan=("TotalLoan_with_Upkeep", "sum"),
        )
        .reset_index()
    )

    grouped["scenario"] = label

    # Shares
    grouped["HH_share"] = grouped["HH"] / grouped["program_cost"]
    grouped["SS_share"] = grouped["SS"] / grouped["program_cost"]
    grouped["LL_share"] = grouped["LL"] / grouped["program_cost"]

    return grouped


# =====================================================
# 3. MTI DISTRIBUTION
# =====================================================

def mti_distribution_summary(df, label="scenario", col="MTI_final"):
    """
    Distribution diagnostics.

    This is critical:
        Policy is about WHO gets support, not just totals.
    """

    s = df[col].dropna()

    result = {
        "scenario": label,
        "N": len(s),

        # Moments
        "mean": s.mean(),
        "std": s.std(),
        "min": s.min(),
        "max": s.max(),

        # Percentiles
        "p1": s.quantile(0.01),
        "p5": s.quantile(0.05),
        "p10": s.quantile(0.10),
        "p25": s.quantile(0.25),
        "p50": s.quantile(0.50),
        "p75": s.quantile(0.75),
        "p90": s.quantile(0.90),
        "p95": s.quantile(0.95),
        "p99": s.quantile(0.99),

        # Policy bands
        "share_below_40": (s < 40).mean(),
        "share_40_60": ((s >= 40) & (s < 60)).mean(),
        "share_60_80": ((s >= 60) & (s < 80)).mean(),
        "share_above_80": (s >= 80).mean(),
    }

    return pd.DataFrame([result])


# =====================================================
# 4. COMPARISON ENGINE
# =====================================================

def compare_aggregates(base_df, scenario_df):
    """
    Compare baseline vs scenario aggregate outputs.
    """

    base = base_df.iloc[0].drop("scenario", errors="ignore")
    scen = scenario_df.iloc[0].drop("scenario", errors="ignore")

    common = base.index.intersection(scen.index)

    comparison = pd.DataFrame({
        "baseline": base[common],
        "scenario": scen[common]
    })

    comparison["change"] = comparison["scenario"] - comparison["baseline"]

    return comparison


def compare_distribution(base_df, scen_df):
    """
    Compare MTI distributions.
    """

    base = base_df.iloc[0].drop("scenario", errors="ignore")
    scen = scen_df.iloc[0].drop("scenario", errors="ignore")

    common = base.index.intersection(scen.index)

    comparison = pd.DataFrame({
        "baseline": base[common],
        "scenario": scen[common]
    })

    comparison["change"] = comparison["scenario"] - comparison["baseline"]

    return comparison


# =====================================================
# 5. MASTER REPORT FUNCTION
# =====================================================

def build_full_report(df, label="scenario"):
    """
    Full reporting bundle.
    """

    report = {
        "aggregate": aggregate_summary(df, label),
        "institution": group_summary(df, "InstitutionName", label)
            if "InstitutionName" in df.columns else group_summary(df, "InstitutonName", label),
        "programme": group_summary(df, "ProgramDescription", label),
        "county": group_summary(df, "County", label),
        "mti_distribution": mti_distribution_summary(df, label),
    }

    return report