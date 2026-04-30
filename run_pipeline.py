"""run_pipeline.py
End-to-end command-line validation for the MTI Policy Lab.

Usage:
    python run_pipeline.py
    python run_pipeline.py --data "Application Data _2026_2027.csv" --out outputs
"""

from pathlib import Path
import argparse
import pandas as pd

from config import get_policy, safe_policy, validate_policy
from data_cleaning import clean_application_data, cleaning_diagnostics
from fee_mapping import apply_fee_mapping, fee_mapping_diagnostics
from simulation_engine import (
    run_scenario,
    compare_aggregate_outputs,
    compare_mti_distributions,
    compare_student_level,
)


def read_csv_robust(path):
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="latin1", low_memory=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="Application Data _2026_2027.csv")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parent
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = base_path / data_path
    output_path = Path(args.out)
    if not output_path.is_absolute():
        output_path = base_path / output_path
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n==============================")
    print("LOADING DATA")
    print("==============================")
    print("Data:", data_path)
    df_raw = read_csv_robust(data_path)
    print("Rows loaded:", f"{len(df_raw):,}")

    print("\n==============================")
    print("POLICY CHECK")
    print("==============================")
    policy = safe_policy(get_policy())
    print(validate_policy(policy))

    print("\n==============================")
    print("DATA CLEANING")
    print("==============================")
    df_clean = clean_application_data(df_raw, policy)
    print(cleaning_diagnostics(df_clean))

    print("\n==============================")
    print("FEE MAPPING")
    print("==============================")
    df_mapped = apply_fee_mapping(df_clean)
    fee_diag = fee_mapping_diagnostics(df_mapped)
    print(fee_diag)
    if fee_diag["primary_missing"] > 0 or fee_diag["secondary_missing"] > 0:
        print("Warning: some fee categories are unmapped. The run continues for diagnostics.")

    print("\n==============================")
    print("RUNNING BASELINE")
    print("==============================")
    baseline = run_scenario(df_mapped, policy, scenario_name="baseline", changes={})
    df_base = baseline["student_level"]
    print(baseline["aggregate"].T)

    print("\n==============================")
    print("BASELINE TARGET CHECK")
    print("==============================")
    targets = {
        "mean_mti": 70.34,
        "HH": 5_081_345_516,
        "SS": 7_904_754_956,
        "LL": 13_858_346_289,
        "program_cost": 26_844_446_761,
    }
    agg = baseline["aggregate"].iloc[0]
    for metric, target in targets.items():
        actual = float(agg[metric])
        print(f"{metric}: actual={actual:,.2f} | target={target:,.2f} | diff={actual-target:,.2f}")

    print("\n==============================")
    print("SAVING BASELINE OUTPUTS")
    print("==============================")
    df_base.to_csv(output_path / "student_level_baseline.csv", index=False)
    for name in ["aggregate", "institution", "programme", "county", "mti_distribution"]:
        baseline[name].to_csv(output_path / f"{name}_baseline.csv", index=False)
    print("Saved to:", output_path)

    print("\n==============================")
    print("RUNNING TEST SCENARIO")
    print("==============================")
    scenario_changes = {
        "university_allocation.hh_formula_mode": "fixed_cap_curve",
    }
    scenario = run_scenario(df_mapped, policy, scenario_name="fixed_cap_curve", changes=scenario_changes)
    print(scenario["aggregate"].T)

    compare_aggregate_outputs(baseline, scenario).to_csv(output_path / "aggregate_comparison.csv")
    compare_mti_distributions(baseline, scenario).to_csv(output_path / "mti_distribution_comparison.csv")
    compare_student_level(baseline, scenario).to_csv(output_path / "student_level_changes.csv", index=False)

    print("\n==============================")
    print("FINAL DIAGNOSTICS")
    print("==============================")
    print("Max identity error baseline:", df_base["TuitionIdentityCheck"].abs().max())
    print("Max identity error scenario:", scenario["student_level"]["TuitionIdentityCheck"].abs().max())
    print("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
