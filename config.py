"""
config_policy_aligned.py
------------------------
Policy-aligned MTI configuration for FY2026/2027.

This file keeps every policy parameter explicit. It also records which
policy levers are available in the current dataset and which require
additional fields before they can be implemented without approximation.
"""

from copy import deepcopy


BASE_POLICY = {
    # Programme-specific affordability references
    "university_cap": 150000,
    "tvet_cost": 67189,

    # Core MTI weights. These must sum to 100.
    "weights": {
        "primary": 29.5,
        "secondary": 24.8,
        "poverty": 24.8,
        "family": 20.9,
    },

    # Poverty thresholds used in the MTI document.
    "thresholds": {
        "primary_poverty": 0.40,
        "poverty_score": 0.40,
    },

    "secondary_score": {
        "max_score": 24.8,
        "decay_lambda": 3.0e-5,
    },

    "poverty_score": {
        "max_score": 24.8,
        "midpoint": 0.40,
        "steepness": 10.0,
    },

    # Household-size score bands.
    "family_scores": {
        "method": "log",
        "max_score": 20.9,
        "fmax": 10,
        "small": 9.3,
        "medium": 16.4,
        "large": 20.9,
    },

    # Data availability audit for document levers.
    # These are not used as formulas; they document what is implementable
    # from the currently available intake fields.
    "available_data_levers": {
        "primary_sponsorship": True,
        "secondary_sponsorship": True,
        "gender": True,
        "student_disability_ncpwd": True,
        "family_structure_orphan_one_parent": True,
        "verified_kra_income": True,
        "declared_school_fee_amounts": False,
        "primary_arrears": False,
        "secondary_arrears": False,
        "school_feeding_programme": False,
        "secondary_placement_downgrade": False,
        "sha_catastrophic_health_expenditure": False,
        "cash_transfer_beneficiary": False,
        "parent_disability": False,
    },

    # Equity adjustment: M_new = M_old + alpha * (100 - M_old).
    # With current fields, one-parent status is inferred from FamilyStructure.
    "equity_adjustment": {
        "enabled": False,
        "female_alpha": 0.05,
        "one_parent_alpha": 0.50,
        "ncpwd_alpha": 0.50,
        "orphan_alpha": 1.00,
    },

    # Income adjustment. The MTI document states this should not apply where
    # an equity adjustment has already been applied, unless policy is changed.
    "income_adjustment": {
        "enabled": True,
        "threshold": 399_996,
        "k": 15,
        "lambda": 0.20,
        "curve": "smoothstep",
        "exclude_equity_adjusted": True,
    },


    # Optional policy-safety layer for programme-cost HH mode.
    # When enabled in allocation_engine.py, HH is capped after the selected
    # HH formula and before computing the financing gap.
    # Identity remains exact because LL is still residual.
    "hh_safety": {
        "enabled": False,
        "cap_amount": 150000,
        "warning_threshold": 200000,
        "hh_share_warning": 0.50,
        "hh_increase_share_warning": 0.40,
    },
    "university_allocation": {
        "hh_formula": "official_cap_curve",
        "hh_cap": 150000,
        "hh_need_discount": 0.90,

        "hh_intercept_mode": "official_cap_curve",
        "hh_intercept_amount": 150000,
        "hh_coefficient": -135000,

        "ss_intercept": 0.15,
        "ss_coefficient": 0.40,

        "ll_intercept": 0.85,
        "ll_coefficient": -0.40,

        "upkeep_intercept": 40000,
        "upkeep_coefficient": 20000,
    },

    "tvet_allocation": {
        # PC fixed at 67,189.
        # HH = PC(0.40 - 0.30x)
        # SS = PC(0.15 + 0.40x)
        # LL is residual: PC - HH - SS, equivalent to PC(0.45 - 0.10x).
        "hh_base": 0.40,
        "hh_slope": 0.30,
        "ss_base": 0.15,
        "ss_slope": 0.40,
        "ll_base": 0.45,
        "ll_slope": 0.10,
        "upkeep_base": 13600,
        "upkeep_slope": 5000,
    },
}


def get_policy():
    return deepcopy(BASE_POLICY)


def validate_policy(policy):
    if not isinstance(policy, dict):
        raise TypeError("policy must be a dictionary")

    weight_sum = sum(policy["weights"].values())
    uni = policy["university_allocation"]
    tvet = policy["tvet_allocation"]

    uni_ss_min = uni["ss_intercept"] + min(0, uni["ss_coefficient"])
    uni_ss_max = uni["ss_intercept"] + max(0, uni["ss_coefficient"])

    # TVET derived loan identity conditions.
    tvet_ll_base_derived = 1.0 - tvet["hh_base"] - tvet["ss_base"]
    tvet_ll_slope_derived = tvet["ss_slope"] - tvet["hh_slope"]

    return {
        "weight_sum": weight_sum,
        "weights_sum_to_100": abs(weight_sum - 100) < 1e-6,
        "family_weight_non_negative": policy["weights"].get("family", 0) >= 0,
        "family_scoring_valid": policy.get("family_scores", {}).get("fmax", 0) > 1 and policy.get("family_scores", {}).get("max_score", 0) >= 0,
        "secondary_score_valid": policy.get("secondary_score", {}).get("decay_lambda", 0) >= 0,
        "poverty_logistic_valid": policy.get("poverty_score", {}).get("steepness", 0) > 0,
        "hh_intercept_mode_valid": uni.get("hh_intercept_mode") in ["fixed_amount", "programme_cost", "official_cap_curve"],
        "university_ss_share_valid_for_x_0_to_1": uni_ss_min >= 0 and uni_ss_max <= 1,
        "university_allocation_identity_coefficients": (
            abs(uni["ss_intercept"] + uni["ll_intercept"] - 1.0) < 1e-6
            and abs(uni["ss_coefficient"] + uni["ll_coefficient"]) < 1e-6
        ),
        "tvet_allocation_identity_coefficients": (
            abs(tvet["hh_base"] + tvet["ss_base"] + tvet["ll_base"] - 1.0) < 1e-6
            and abs(-tvet["hh_slope"] + tvet["ss_slope"] - tvet["ll_slope"]) < 1e-6
        ),
        "tvet_ll_base_derived": tvet_ll_base_derived,
        "tvet_ll_slope_derived": tvet_ll_slope_derived,
        "hh_safety_valid": (not policy.get("hh_safety", {}).get("enabled", False)) or policy.get("hh_safety", {}).get("cap_amount", 0) >= 0,
        "hh_warning_threshold": policy.get("hh_safety", {}).get("warning_threshold", None),
        "income_k_valid": policy["income_adjustment"]["k"] > 1,
        "income_lambda_valid": 0 <= policy["income_adjustment"]["lambda"] <= 1,
        "university_hh_formula_valid": uni.get("hh_cap", 0) >= 0 and 0 <= uni.get("hh_need_discount", 0) <= 1,
    }
