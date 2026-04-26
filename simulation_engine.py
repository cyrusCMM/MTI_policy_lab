"""
simulation_engine.py
--------------------

Purpose:
    Run MTI policy simulations.

A scenario changes policy parameters only.
It does not change the MTI or allocation formula logic.

Flow:
    clean + fee-mapped data
        -> apply scenario parameter changes
        -> compute MTI
        -> compute allocations
        -> produce reports
        -> compare baseline vs scenario
"""

from copy import deepcopy
import pandas as pd

from mti_engine import compute_mti
from allocation_engine import compute_allocations
from reporting_engine import (
    aggregate_summary,
    group_summary,
    mti_distribution_summary
)


# =====================================================
# 1. APPLY POLICY CHANGES
# =====================================================

def apply_changes(policy, changes=None):
    """
    Apply scenario changes to a policy dictionary.

    Example:
        changes = {
            "weights.poverty": 30,
            "university_allocation.ss_base": 0.20
        }

    This means:
        policy["weights"]["poverty"] = 30
        policy["university_allocation"]["ss_base"] = 0.20
    """

    if changes is None:
        changes = {}

    policy = deepcopy(policy)

    for path, value in changes.items():
        keys = path.split(".")
        target = policy

        for key in keys[:-1]:
            if key not in target:
                raise KeyError(f"Policy path not found: {path}")
            target = target[key]

        target[keys[-1]] = value

    return policy


# =====================================================
# 2. RUN ONE SCENARIO
# =====================================================

def run_scenario(clean_df, base_policy, scenario_name="scenario", changes=None):
    """
    Run one MTI policy scenario.

    Inputs:
        clean_df:
            Data already cleaned and fee-mapped.

        base_policy:
            Baseline policy dictionary.

        scenario_name:
            Name of scenario.

        changes:
            Dictionary of scenario parameter changes.

    Output:
        Dictionary with:
            student_level
            aggregate
            institution
            programme
            county
            mti_distribution
            policy
    """

    if changes is None:
        changes = {}

    policy = apply_changes(base_policy, changes)

    # Step 1: compute MTI
    out = compute_mti(clean_df, policy)

    # Step 2: compute allocation
    out = compute_allocations(out, policy, mti_col="MTI_final")

    # Step 3: scenario label
    out["Scenario"] = scenario_name

    # Step 4: institution column handling
    if "InstitutionName" in out.columns:
        inst_col = "InstitutionName"
    elif "InstitutonName" in out.columns:
        inst_col = "InstitutonName"
    else:
        inst_col = None

    institution_summary = (
        group_summary(out, inst_col, scenario_name)
        if inst_col is not None
        else pd.DataFrame()
    )

    result = {
        "scenario_name": scenario_name,
        "policy": policy,
        "student_level": out,
        "aggregate": aggregate_summary(out, scenario_name),
        "institution": institution_summary,
        "programme": group_summary(out, "ProgramDescription", scenario_name),
        "county": group_summary(out, "County", scenario_name),
        "mti_distribution": mti_distribution_summary(out, scenario_name),
    }

    return result


# =====================================================
# 3. RUN MULTIPLE SCENARIOS
# =====================================================

def run_multiple_scenarios(clean_df, base_policy, scenarios):
    """
    Run many scenarios.

    Example:
        scenarios = {
            "baseline": {},
            "higher_poverty_weight": {
                "weights.poverty": 30,
                "weights.primary": 25
            }
        }
    """

    results = {}

    for scenario_name, changes in scenarios.items():
        results[scenario_name] = run_scenario(
            clean_df=clean_df,
            base_policy=base_policy,
            scenario_name=scenario_name,
            changes=changes
        )

    return results


# =====================================================
# 4. COMPARE AGGREGATE OUTPUTS
# =====================================================

def compare_aggregate_outputs(base_result, scenario_result):
    """
    Compare aggregate outputs between baseline and scenario.

    Difference convention:
        scenario - baseline
    """

    base = base_result["aggregate"].iloc[0].drop("scenario", errors="ignore")
    scen = scenario_result["aggregate"].iloc[0].drop("scenario", errors="ignore")

    common = base.index.intersection(scen.index)

    comparison = pd.DataFrame({
        "baseline": base[common],
        "scenario": scen[common]
    })

    comparison["change"] = comparison["scenario"] - comparison["baseline"]

    return comparison


# =====================================================
# 5. COMPARE MTI DISTRIBUTIONS
# =====================================================

def compare_mti_distributions(base_result, scenario_result):
    """
    Compare MTI distribution summaries.

    Difference convention:
        scenario - baseline
    """

    base = base_result["mti_distribution"].iloc[0].drop("scenario", errors="ignore")
    scen = scenario_result["mti_distribution"].iloc[0].drop("scenario", errors="ignore")

    common = base.index.intersection(scen.index)

    comparison = pd.DataFrame({
        "baseline": base[common],
        "scenario": scen[common]
    })

    comparison["change"] = comparison["scenario"] - comparison["baseline"]

    return comparison


# =====================================================
# 6. COMPARE STUDENT-LEVEL OUTPUTS
# =====================================================

def compare_student_level(base_result, scenario_result, id_col="user_id"):
    """
    Compare baseline and scenario at student level.

    Difference convention:
        scenario - baseline

    Positive HH_change:
        Household pays more under scenario.

    Positive SS_change:
        Scholarship increases under scenario.

    Positive LL_change:
        Loan increases under scenario.
    """

    base = base_result["student_level"].copy()
    scen = scenario_result["student_level"].copy()

    comparison_cols = [
        "MTI_final",
        "HH",
        "SS",
        "LL",
        "Upkeep",
        "TotalLoan_with_Upkeep",
        "PC_allocation"
    ]

    if id_col in base.columns and id_col in scen.columns:
        keep_base = [id_col] + [c for c in comparison_cols if c in base.columns]
        keep_scen = [id_col] + [c for c in comparison_cols if c in scen.columns]

        merged = base[keep_base].merge(
            scen[keep_scen],
            on=id_col,
            how="inner",
            suffixes=("_baseline", "_scenario")
        )
    else:
        # Fallback: compare by row order if no ID exists
        base = base.reset_index().rename(columns={"index": "_row_id"})
        scen = scen.reset_index().rename(columns={"index": "_row_id"})

        keep_base = ["_row_id"] + [c for c in comparison_cols if c in base.columns]
        keep_scen = ["_row_id"] + [c for c in comparison_cols if c in scen.columns]

        merged = base[keep_base].merge(
            scen[keep_scen],
            on="_row_id",
            how="inner",
            suffixes=("_baseline", "_scenario")
        )

    for col in comparison_cols:
        b = f"{col}_baseline"
        s = f"{col}_scenario"

        if b in merged.columns and s in merged.columns:
            merged[f"{col}_change"] = merged[s] - merged[b]

    return merged