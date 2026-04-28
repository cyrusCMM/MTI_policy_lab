# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:17:29 2026

@author: hp
"""

"""
visualization_engine.py
-----------------------

Purpose:
    Create visual outputs for the MTI Policy Lab.

Visuals produced:
    1. MTI distribution comparison
    2. Allocation distribution comparison
    3. Aggregate allocation bar chart
    4. MTI band share comparison
    5. Top institution/programme/county effects

These plots are used by:
    - Streamlit app
    - notebook diagnostics
    - policy reports
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# =====================================================
# 1. MTI DISTRIBUTION COMPARISON
# =====================================================

def plot_mti_distribution(base_df, scenario_df=None, scenario_name="Scenario"):
    """
    Plot MTI distribution.

    Shows whether the scenario shifts students toward:
        - lower need scores
        - middle need scores
        - higher need scores
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.kdeplot(
        base_df["MTI_final"].dropna(),
        label="Baseline",
        linewidth=2,
        ax=ax
    )

    if scenario_df is not None:
        sns.kdeplot(
            scenario_df["MTI_final"].dropna(),
            label=scenario_name,
            linewidth=2,
            ax=ax
        )

    ax.set_title("MTI Distribution: Baseline vs Scenario")
    ax.set_xlabel("MTI Score")
    ax.set_ylabel("Density")
    ax.legend()

    return fig


# =====================================================
# 2. OLD VS RECONSTRUCTED MTI DISTRIBUTION
# =====================================================

def plot_old_vs_new_mti(df):
    """
    Compare existing MTIScore with reconstructed MTI_final.

    Useful for validation and explaining the difference between:
        - official/current MTI
        - reconstructed formula-based MTI
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    if "MTIScore" in df.columns:
        sns.kdeplot(
            df["MTIScore"].dropna(),
            label="Existing MTI",
            linewidth=2,
            ax=ax
        )

    sns.kdeplot(
        df["MTI_final"].dropna(),
        label="Reconstructed MTI",
        linewidth=2,
        ax=ax
    )

    ax.set_title("Existing vs Reconstructed MTI Distribution")
    ax.set_xlabel("MTI Score")
    ax.set_ylabel("Density")
    ax.legend()

    return fig


# =====================================================
# 3. SCATTER: OLD VS RECONSTRUCTED MTI
# =====================================================

def plot_old_vs_new_scatter(df):
    """
    Scatter plot of existing MTIScore against reconstructed MTI_final.

    The 45-degree line shows exact agreement.
    Points below the line:
        reconstructed MTI is lower than existing MTI.
    Points above the line:
        reconstructed MTI is higher than existing MTI.
    """

    fig, ax = plt.subplots(figsize=(8, 7))

    if "MTIScore" not in df.columns:
        ax.text(0.5, 0.5, "MTIScore not found", ha="center")
        return fig

    plot_df = df[["MTIScore", "MTI_final"]].dropna()

    sns.scatterplot(
        data=plot_df,
        x="MTIScore",
        y="MTI_final",
        alpha=0.25,
        s=12,
        ax=ax
    )

    ax.plot([0, 100], [0, 100], linestyle="--", linewidth=2)

    corr = plot_df[["MTIScore", "MTI_final"]].corr().iloc[0, 1]

    ax.set_title(f"Existing vs Reconstructed MTI\nCorrelation = {corr:.3f}")
    ax.set_xlabel("Existing MTI")
    ax.set_ylabel("Reconstructed MTI")

    return fig


# =====================================================
# 4. ALLOCATION DISTRIBUTIONS
# =====================================================

def plot_allocation_distribution(df):
    """
    Plot distribution of HH, SS, LL, and Upkeep.

    Helps show who bears the cost:
        - households
        - government scholarship
        - student loans
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    for col in ["HH", "SS", "LL", "Upkeep"]:
        if col in df.columns:
            sns.kdeplot(
                df[col].dropna(),
                label=col,
                linewidth=2,
                ax=ax
            )

    ax.set_title("Allocation Distributions")
    ax.set_xlabel("KES")
    ax.set_ylabel("Density")
    ax.legend()

    return fig


# =====================================================
# 5. AGGREGATE ALLOCATION BAR CHART
# =====================================================

def plot_aggregate_allocation_comparison(base_summary, scenario_summary):
    """
    Compare aggregate HH, SS, LL, and Upkeep across baseline and scenario.
    """

    metrics = ["HH", "SS", "LL", "Upkeep"]

    base = base_summary.iloc[0]
    scen = scenario_summary.iloc[0]

    plot_df = pd.DataFrame({
        "Component": metrics,
        "Baseline": [base[m] for m in metrics],
        "Scenario": [scen[m] for m in metrics],
    })

    plot_long = plot_df.melt(
        id_vars="Component",
        var_name="Run",
        value_name="Amount"
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=plot_long,
        x="Component",
        y="Amount",
        hue="Run",
        ax=ax
    )

    ax.set_title("Aggregate Allocation Comparison")
    ax.set_ylabel("Amount, KES")
    ax.set_xlabel("Component")

    return fig


# =====================================================
# 6. MTI BAND SHARE COMPARISON
# =====================================================

def plot_mti_band_shares(base_dist, scenario_dist):
    """
    Plot MTI band shares.

    Bands:
        <40
        40-60
        60-80
        >=80
    """

    bands = [
        "share_below_40",
        "share_40_60",
        "share_60_80",
        "share_above_80"
    ]

    labels = [
        "Below 40",
        "40-60",
        "60-80",
        "80+"
    ]

    base = base_dist.iloc[0]
    scen = scenario_dist.iloc[0]

    plot_df = pd.DataFrame({
        "Band": labels,
        "Baseline": [base[b] for b in bands],
        "Scenario": [scen[b] for b in bands],
    })

    plot_long = plot_df.melt(
        id_vars="Band",
        var_name="Run",
        value_name="Share"
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=plot_long,
        x="Band",
        y="Share",
        hue="Run",
        ax=ax
    )

    ax.set_title("MTI Distribution by Score Band")
    ax.set_ylabel("Share of Students")
    ax.set_xlabel("MTI Band")

    return fig


# =====================================================
# 7. TOP GROUP EFFECTS
# =====================================================

def plot_top_group_effects(group_df, value_col="program_cost", group_col=None, top_n=15):
    """
    Plot top institutions/programmes/counties by selected metric.

    Example:
        plot_top_group_effects(institution_df, "SS", "InstitutionName")
    """

    if group_df is None or group_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No group data available", ha="center")
        return fig

    if group_col is None:
        group_col = group_df.columns[0]

    plot_df = (
        group_df
        .sort_values(value_col, ascending=False)
        .head(top_n)
        .copy()
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    sns.barplot(
        data=plot_df,
        y=group_col,
        x=value_col,
        ax=ax
    )

    ax.set_title(f"Top {top_n} by {value_col}")
    ax.set_xlabel(value_col)
    ax.set_ylabel(group_col)

    return fig


# =====================================================
# 8. STUDENT-LEVEL IMPACT DISTRIBUTION
# =====================================================

def plot_student_level_change(change_df):
    """
    Plot distribution of changes from baseline to scenario.

    Expected columns:
        MTI_final_change
        HH_change
        SS_change
        LL_change
        Upkeep_change
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    for col in ["MTI_final_change", "HH_change", "SS_change", "LL_change"]:
        if col in change_df.columns:
            sns.kdeplot(
                change_df[col].dropna(),
                label=col,
                linewidth=2,
                ax=ax
            )

    ax.set_title("Student-Level Scenario Impact Distribution")
    ax.set_xlabel("Scenario - Baseline")
    ax.set_ylabel("Density")
    ax.legend()

    return fig