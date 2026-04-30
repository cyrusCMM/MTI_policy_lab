"""simulation_engine.py
Scenario runner and comparison helpers for the MTI Policy Lab.
"""

import copy
import pandas as pd


def _set_dotted(policy, dotted_key, value):
    target = policy
    parts = str(dotted_key).split(".")
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            target[part] = {}
        target = target[part]
    target[parts[-1]] = value


def apply_policy_changes(base_policy, changes):
    policy = copy.deepcopy(base_policy)
    if not changes:
        return policy

    for key, value in changes.items():
        if "." in str(key):
            _set_dotted(policy, key, value)
        elif isinstance(value, dict):
            if key not in policy or not isinstance(policy[key], dict):
                policy[key] = {}
            policy[key].update(value)
        else:
            policy[key] = value
    return policy


def run_scenario(clean_df, base_policy, scenario_name="baseline", changes=None):
    if clean_df is None or len(clean_df) == 0:
        raise ValueError("Empty dataframe")
    if base_policy is None:
        raise ValueError("Policy is None")

    policy = apply_policy_changes(base_policy, changes)

    from config import safe_policy
    from mti_engine import compute_mti
    from allocation_engine import compute_allocations
    from reporting_engine import build_full_report

    policy = safe_policy(policy)
    df = compute_mti(clean_df.copy(), policy)
    df = compute_allocations(df, policy)
    report = build_full_report(df, scenario_name)

    return {
        "student_level": df,
        "aggregate": report["aggregate"],
        "institution": report["institution"],
        "programme": report["programme"],
        "county": report["county"],
        "mti_distribution": report["mti_distribution"],
        "scenario": scenario_name,
        "policy": policy,
    }


def compare_aggregate_outputs(baseline, scenario):
    from reporting_engine import compare_aggregates
    return compare_aggregates(baseline["aggregate"], scenario["aggregate"])


def compare_mti_distributions(baseline, scenario):
    from reporting_engine import compare_distribution
    return compare_distribution(baseline["mti_distribution"], scenario["mti_distribution"])


def compare_student_level(baseline, scenario, key=None):
    base = baseline["student_level"]
    scen = scenario["student_level"]

    if key is None:
        key = "user_id" if "user_id" in base.columns and "user_id" in scen.columns else None

    cols = ["MTI_final", "HH", "SS", "LL", "Upkeep", "PC_allocation"]
    cols = [c for c in cols if c in base.columns and c in scen.columns]

    if key is not None:
        out = base[[key] + cols].merge(
            scen[[key] + cols], on=key, suffixes=("_baseline", "_scenario")
        )
    else:
        out = pd.DataFrame(index=base.index)
        for col in cols:
            out[f"{col}_baseline"] = base[col].values
            out[f"{col}_scenario"] = scen[col].values

    for col in cols:
        out[f"{col}_change"] = out[f"{col}_scenario"] - out[f"{col}_baseline"]
    return out
