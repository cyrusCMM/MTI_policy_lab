"""reporting_engine.py
Policy reporting bundle for MTI scenarios.
"""

import pandas as pd


def _safe_div(num, den):
    return 0 if den == 0 else num / den


def aggregate_summary(df, label="scenario"):
    total_pc = float(df["PC_allocation"].sum())
    hh = float(df["HH"].sum())
    ss = float(df["SS"].sum())
    ll = float(df["LL"].sum())
    upkeep = float(df.get("Upkeep", pd.Series(0, index=df.index)).sum())
    total_loan = float(df.get("TotalLoan_with_Upkeep", df["LL"] + df.get("Upkeep", 0)).sum())

    result = {
        "scenario": label,
        "students": int(len(df)),
        "mean_mti": float(df["MTI_final"].mean()),
        "median_mti": float(df["MTI_final"].median()),
        "mean_mti_baseline": float(df["MTI_baseline"].mean()) if "MTI_baseline" in df.columns else None,
        "mean_mti_equity": float(df["MTI_equity"].mean()) if "MTI_equity" in df.columns else None,
        "program_cost": total_pc,
        "HH": hh,
        "SS": ss,
        "LL": ll,
        "Upkeep": upkeep,
        "TotalLoan_with_Upkeep": total_loan,
        "HH_share": _safe_div(hh, total_pc),
        "SS_share": _safe_div(ss, total_pc),
        "LL_share": _safe_div(ll, total_pc),
        "GovSupport_SS_LL": ss + ll,
        "max_identity_error": float(df["TuitionIdentityCheck"].abs().max()) if "TuitionIdentityCheck" in df.columns else None,
    }
    return pd.DataFrame([result])


def group_summary(df, group_col, label="scenario"):
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
    grouped["HH_share"] = grouped["HH"] / grouped["program_cost"].replace(0, pd.NA)
    grouped["SS_share"] = grouped["SS"] / grouped["program_cost"].replace(0, pd.NA)
    grouped["LL_share"] = grouped["LL"] / grouped["program_cost"].replace(0, pd.NA)
    return grouped


def mti_distribution_summary(df, label="scenario", col="MTI_final"):
    s = df[col].dropna()
    result = {
        "scenario": label,
        "N": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": float(s.min()),
        "max": float(s.max()),
        "p1": float(s.quantile(0.01)),
        "p5": float(s.quantile(0.05)),
        "p10": float(s.quantile(0.10)),
        "p25": float(s.quantile(0.25)),
        "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "share_below_40": float((s < 40).mean()),
        "share_40_60": float(((s >= 40) & (s < 60)).mean()),
        "share_60_80": float(((s >= 60) & (s < 80)).mean()),
        "share_above_80": float((s >= 80).mean()),
    }
    return pd.DataFrame([result])


def compare_aggregates(base_df, scenario_df):
    base = base_df.iloc[0].drop("scenario", errors="ignore")
    scen = scenario_df.iloc[0].drop("scenario", errors="ignore")
    common = base.index.intersection(scen.index)
    comparison = pd.DataFrame({"baseline": base[common], "scenario": scen[common]})
    comparison["change"] = comparison["scenario"] - comparison["baseline"]
    return comparison


def compare_distribution(base_df, scen_df):
    return compare_aggregates(base_df, scen_df)


def build_full_report(df, label="scenario"):
    inst_col = "InstitutionName" if "InstitutionName" in df.columns else "InstitutonName"
    return {
        "aggregate": aggregate_summary(df, label),
        "institution": group_summary(df, inst_col, label),
        "programme": group_summary(df, "ProgramDescription", label),
        "county": group_summary(df, "County", label),
        "mti_distribution": mti_distribution_summary(df, label),
    }
