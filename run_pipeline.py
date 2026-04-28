# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:19:10 2026

@author: hp
"""

"""
run_pipeline.py
---------------

Purpose:
    End-to-end execution of MTI system.

This script:
    1. Loads raw intake data
    2. Cleans data
    3. Applies fee mapping
    4. Computes MTI
    5. Computes allocations
    6. Produces reports
    7. Runs scenario comparison

This MUST work perfectly before using Streamlit.

Usage:
    python run_pipeline.py
"""

import pandas as pd
from pathlib import Path

# Core modules
from config import get_policy, validate_policy
from data_cleaning import clean_application_data, cleaning_diagnostics
from fee_mapping import apply_fee_mapping, fee_mapping_diagnostics
from simulation_engine import (
    run_scenario,
    compare_aggregate_outputs,
    compare_mti_distributions,
    compare_student_level
)


# =====================================================
# 1. LOAD DATA
# =====================================================

BASE_PATH = Path("C:/Users/hp/Documents/MTI")

DATA_PATH = BASE_PATH / "Application Data 2025_2026.csv"
OUTPUT_PATH = BASE_PATH / "outputs"

OUTPUT_PATH.mkdir(exist_ok=True)

print("\n==============================")
print("LOADING DATA")
print("==============================")

df_raw = pd.read_csv(DATA_PATH)

print("Rows loaded:", len(df_raw))


# =====================================================
# 2. LOAD POLICY
# =====================================================

policy = get_policy()

print("\n==============================")
print("POLICY CHECK")
print("==============================")

print(validate_policy(policy))


# =====================================================
# 3. CLEAN DATA
# =====================================================

print("\n==============================")
print("DATA CLEANING")
print("==============================")

df_clean = clean_application_data(df_raw, policy)

print(cleaning_diagnostics(df_clean))


# =====================================================
# 4. FEE MAPPING
# =====================================================

print("\n==============================")
print("FEE MAPPING")
print("==============================")

df_mapped = apply_fee_mapping(df_clean)

fee_diag = fee_mapping_diagnostics(df_mapped)
print(fee_diag)

if fee_diag["secondary_missing"] > 0:
    raise ValueError("Secondary fee mapping incomplete. Fix before proceeding.")


# =====================================================
# 5. BASELINE SCENARIO
# =====================================================

print("\n==============================")
print("RUNNING BASELINE")
print("==============================")

baseline = run_scenario(
    clean_df=df_mapped,
    base_policy=policy,
    scenario_name="baseline",
    changes={}
)

df_base = baseline["student_level"]

print("Baseline MTI mean:", df_base["MTI_final"].mean())


# =====================================================
# 6. VALIDATION: EXISTING VS RECONSTRUCTED MTI
# =====================================================

if "MTIScore" in df_base.columns:

    print("\n==============================")
    print("MTI RECONSTRUCTION CHECK")
    print("==============================")

    comparison = df_base[["MTIScore", "MTI_final"]].dropna()

    corr = comparison.corr().iloc[0, 1]

    print("Correlation:", corr)

    diff = comparison["MTIScore"] - comparison["MTI_final"]

    print("Mean difference:", diff.mean())
    print("Std difference:", diff.std())


# =====================================================
# 7. SAVE BASELINE OUTPUTS
# =====================================================

df_base.to_csv(OUTPUT_PATH / "student_level_baseline.csv", index=False)
baseline["aggregate"].to_csv(OUTPUT_PATH / "aggregate_baseline.csv", index=False)
baseline["institution"].to_csv(OUTPUT_PATH / "institution_baseline.csv", index=False)
baseline["programme"].to_csv(OUTPUT_PATH / "programme_baseline.csv", index=False)
baseline["county"].to_csv(OUTPUT_PATH / "county_baseline.csv", index=False)
baseline["mti_distribution"].to_csv(OUTPUT_PATH / "mti_distribution_baseline.csv", index=False)

print("\nBaseline outputs saved.")


# =====================================================
# 8. SCENARIO TEST
# =====================================================

print("\n==============================")
print("RUNNING SCENARIO")
print("==============================")

scenario_changes = {
    # Example: shift weight toward poverty
    "weights.poverty": 30.0,
    "weights.primary": 25.0,
}

scenario = run_scenario(
    clean_df=df_mapped,
    base_policy=policy,
    scenario_name="higher_poverty_weight",
    changes=scenario_changes
)

df_scen = scenario["student_level"]

print("Scenario MTI mean:", df_scen["MTI_final"].mean())


# =====================================================
# 9. COMPARE AGGREGATE
# =====================================================

print("\n==============================")
print("AGGREGATE COMPARISON")
print("==============================")

agg_compare = compare_aggregate_outputs(baseline, scenario)
print(agg_compare)

agg_compare.to_csv(OUTPUT_PATH / "aggregate_comparison.csv")


# =====================================================
# 10. COMPARE MTI DISTRIBUTION
# =====================================================

print("\n==============================")
print("MTI DISTRIBUTION COMPARISON")
print("==============================")

dist_compare = compare_mti_distributions(baseline, scenario)
print(dist_compare)

dist_compare.to_csv(OUTPUT_PATH / "mti_distribution_comparison.csv")


# =====================================================
# 11. STUDENT-LEVEL IMPACT
# =====================================================

print("\n==============================")
print("STUDENT-LEVEL IMPACT")
print("==============================")

student_diff = compare_student_level(baseline, scenario)

student_diff.to_csv(OUTPUT_PATH / "student_level_changes.csv", index=False)

print("Student-level changes saved.")


# =====================================================
# 12. FINAL CHECK
# =====================================================

print("\n==============================")
print("FINAL DIAGNOSTICS")
print("==============================")

print("Max identity error (baseline):",
      df_base["TuitionIdentityCheck"].abs().max())

print("Max identity error (scenario):",
      df_scen["TuitionIdentityCheck"].abs().max())

print("\nPIPELINE COMPLETE.")