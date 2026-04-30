"""config.py
Central policy defaults for the MTI Policy Lab.

The defaults are set to reproduce the Excel FY2026/2027 baseline first.
Scenario changes should be layered on top of these defaults, not hard-coded
inside the engines.
"""

from copy import deepcopy


BASE_POLICY = {
    "fee_benchmark": 150000,
    "university_cap": 150000,
    "tvet_cost": 67189,
    "allocation_mode": "excel_common",

    "weights": {
        "primary": 29.5,
        "secondary": 24.8,
        "poverty": 24.8,
        "family": 20.9,
    },

    "thresholds": {
        "primary_poverty": 39.8,
        "poverty_score": 60.0,
    },

    "secondary_score": {
        "method": "linear",
        "max_score": 24.8,
        "fee_benchmark": 150000,
        "decay_lambda": 3.0e-5,
    },

    "poverty_score": {
        "method": "linear",
        "max_score": 24.8,
        "threshold": 60.0,
        "midpoint": 0.40,
        "steepness": 10.0,
    },

    "family_scores": {
        "method": "log",
        "max_score": 20.9,
        "fmax": 10,
        "small": 9.3,
        "medium": 16.4,
        "large": 20.9,
    },

    "equity_adjustment": {
        "enabled": True,
        "female_alpha": 0.05,
        "ncpwd_alpha": 0.50,
        "one_parent_alpha": 0.50,
        "orphan_alpha": 1.00,
        "parent_disability_alpha": 0.50,
        "cash_transfer_alpha": 0.50,
        "family_structure_alpha": {
            "ORPHANED": 1.00,
            "ORPHAN": 1.00,
            "NO_PARENTS": 1.00,
            "NO PARENTS": 1.00,
            "ABANDONED": 1.00,
            "DECEASED_SINGLE": 1.00,
            "DECEASED SINGLE": 1.00,
            "BOTH_PARENTS": 0.00,
            "BOTH PARENTS": 0.00,
            "SINGLE_MOTHER": 0.50,
            "SINGLE MOTHER": 0.50,
            "SINGLE_FATHER": 0.50,
            "SINGLE FATHER": 0.50,
            "ONE_PARENT_DECEASED": 0.50,
            "ONE PARENT DECEASED": 0.50,
        },
    },

    "income_adjustment": {
        "enabled": True,
        "threshold": 399996,
        "k": 15,
        "lambda": 0.20,
        "curve": "smoothstep",
        "exclude_equity_adjusted": False,
        "round_final_mti": True,
    },

    "university_allocation": {
        "hh_formula_mode": "program_cost_share",
        "hh_base_share": 0.10,
        "hh_ability_share": 0.30,
        "hh_cap": 150000,
        "hh_discount": 0.90,
        "ss_base_share": 0.15,
        "ss_need_share": 0.30,
        "upkeep_intercept": 0,
        "upkeep_coefficient": 0,
    },

    "tvet_allocation": {
        "hh_base": 0.10,
        "hh_slope": 0.30,
        "ss_base": 0.15,
        "ss_slope": 0.30,
        "upkeep_base": 0,
        "upkeep_slope": 0,
    },
}


def deep_merge(defaults, override):
    """Recursively merge override into defaults while preserving missing nested keys."""
    merged = deepcopy(defaults)
    if not isinstance(override, dict):
        return merged
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def get_policy():
    return deepcopy(BASE_POLICY)


def safe_policy(policy):
    return deep_merge(BASE_POLICY, policy)


def validate_policy(policy):
    """Return basic policy diagnostics used by run_pipeline.py."""
    p = safe_policy(policy)
    weights = p["weights"]
    total_weight = sum(float(v) for v in weights.values())
    ua = p["university_allocation"]
    mode = ua.get("hh_formula_mode")
    if mode not in {"program_cost_share", "fixed_cap_curve"}:
        raise ValueError(f"Unsupported hh_formula_mode: {mode}")
    return {
        "weights_total": total_weight,
        "weights_sum_to_100": abs(total_weight - 100.0) < 1e-9,
        "hh_formula_mode": mode,
        "allocation_mode": p.get("allocation_mode"),
    }
