"""
MTI Policy Lab - multi scenario app with university allocation changes,
line distributions, min/max stats, MTI class frequencies, and transition matrix.
"""

from pathlib import Path
import copy

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from config import get_policy, validate_policy
from data_cleaning import clean_application_data, cleaning_diagnostics
from fee_mapping import apply_fee_mapping, fee_mapping_diagnostics
from simulation_engine import run_scenario, compare_aggregate_outputs, compare_mti_distributions, compare_student_level

st.set_page_config(page_title="MTI Policy Lab", layout="wide")

st.markdown("""
<style>
[data-testid="stMetricValue"] {
    font-size: 1.45rem !important;
    line-height: 1.15 !important;
    white-space: normal !important;
}
[data-testid="stMetricLabel"] { font-size: 0.88rem !important; }
.block-container { padding-top: 1.4rem; }
</style>
""", unsafe_allow_html=True)

MONEY_COLS = {
    "program_cost", "program_cost_baseline", "program_cost_scenario", "program_cost_change",
    "HH", "HH_baseline", "HH_scenario", "HH_change",
    "SS", "SS_baseline", "SS_scenario", "SS_change",
    "LL", "LL_baseline", "LL_scenario", "LL_change",
    "Upkeep", "Upkeep_baseline", "Upkeep_scenario", "Upkeep_change",
    "total_loan", "total_loan_baseline", "total_loan_scenario", "total_loan_change",
    "TotalLoan_with_Upkeep", "TotalLoan_with_Upkeep_baseline", "TotalLoan_with_Upkeep_scenario", "TotalLoan_with_Upkeep_change",
    "PC_allocation", "PC_allocation_baseline", "PC_allocation_scenario", "PC_allocation_change",
}
MONEY_COLS.update({"baseline", "scenario", "change", "mean_increase", "mean_decrease", "largest_increase", "largest_decrease", "HH_change_mean", "HH_change_median", "HH_change_min", "HH_change_max", "SS_change_mean", "SS_change_median", "SS_change_min", "SS_change_max", "LL_change_mean", "LL_change_median", "LL_change_min", "LL_change_max", "Upkeep_change_mean", "Upkeep_change_median", "Upkeep_change_min", "Upkeep_change_max", "TotalLoan_with_Upkeep_change_mean", "TotalLoan_with_Upkeep_change_median", "TotalLoan_with_Upkeep_change_min", "TotalLoan_with_Upkeep_change_max"})
MONEY_ROWS = {"program_cost", "HH", "SS", "LL", "Upkeep", "TotalLoan_with_Upkeep", "PC_allocation", "max_identity_error"}
SHARE_ROWS = {"HH_share", "SS_share", "LL_share", "share_below_40", "share_40_60", "share_60_80", "share_above_80"}


def fmt_ksh(x):
    try:
        return f"KSh {float(x):,.0f}"
    except Exception:
        return x


def fmt_num(x):
    try:
        return f"{float(x):,.3f}"
    except Exception:
        return x


def fmt_share(x):
    try:
        return f"{float(x):.2%}"
    except Exception:
        return x


def style_indexed(df):
    out = df.copy()
    for idx in out.index:
        for col in out.columns:
            v = out.loc[idx, col]
            if idx in MONEY_ROWS:
                out.loc[idx, col] = fmt_ksh(v)
            elif idx in SHARE_ROWS:
                out.loc[idx, col] = fmt_share(v)
            elif idx in {"students", "N"}:
                out.loc[idx, col] = f"{float(v):,.0f}" if pd.notna(v) else v
            else:
                out.loc[idx, col] = fmt_num(v)
    return out


def style_table(df):
    out = df.copy()
    for col in out.columns:
        if col in MONEY_COLS:
            out[col] = out[col].map(fmt_ksh)
        elif "share" in col.lower() or col.endswith("_rate"):
            out[col] = out[col].map(fmt_share)
        elif col == "students" or col.endswith("_students") or col in {"baseline_students", "scenario_students", "student_change", "cumulative_student_change"}:
            out[col] = out[col].map(lambda x: f"{float(x):,.0f}" if pd.notna(x) else x)
        elif col in {"mean_mti", "median_mti", "mean_mti_baseline", "mean_mti_scenario", "mean_mti_change"}:
            out[col] = out[col].map(lambda x: f"{float(x):,.2f}" if pd.notna(x) else x)
    return out


def fresh_policy():
    p = get_policy()
    if not isinstance(p, dict):
        raise RuntimeError("get_policy() did not return a policy dictionary.")
    return p

def deep_merge_policy(default_policy, user_policy):
    """
    Merge a possibly incomplete/None policy into the default policy.
    Critical rule: None never overwrites a dictionary section.
    This prevents errors like: NoneType has no attribute 'get'.
    """
    merged = copy.deepcopy(default_policy) if isinstance(default_policy, dict) else {}

    if not isinstance(user_policy, dict):
        user_policy = {}

    def _merge(dst, src):
        if not isinstance(src, dict):
            return dst
        for k, v in src.items():
            if v is None:
                # Never overwrite defaults with None.
                continue
            if isinstance(dst.get(k), dict):
                if isinstance(v, dict):
                    _merge(dst[k], v)
                else:
                    # Do not replace a section dict with a scalar.
                    continue
            else:
                dst[k] = v
        return dst

    _merge(merged, user_policy)

    # Replace any accidentally non-dict sections with default dictionaries.
    def ensure_section(name):
        default_section = default_policy.get(name, {}) if isinstance(default_policy, dict) else {}
        if not isinstance(merged.get(name), dict):
            merged[name] = copy.deepcopy(default_section) if isinstance(default_section, dict) else {}
        return merged[name]

    weights = ensure_section("weights")
    weights.setdefault("primary", 29.5)
    weights.setdefault("secondary", 24.8)
    weights.setdefault("poverty", 24.8)
    weights.setdefault("family", 20.9)

    thresholds = ensure_section("thresholds")
    thresholds.setdefault("primary_poverty", 0.40)
    thresholds.setdefault("poverty_score", 0.60)

    family_scores = ensure_section("family_scores")
    family_scores.setdefault("small", 9.3)
    family_scores.setdefault("medium", 16.4)
    family_scores.setdefault("large", 20.9)

    eq = ensure_section("equity_adjustment")
    eq.setdefault("enabled", False)
    eq.setdefault("female_alpha", 0.05)
    eq.setdefault("one_parent_alpha", 0.50)
    eq.setdefault("ncpwd_alpha", 0.50)
    eq.setdefault("orphan_alpha", 1.00)

    inc = ensure_section("income_adjustment")
    inc.setdefault("enabled", True)
    inc.setdefault("threshold", 1200000)
    inc.setdefault("k", 3)
    inc.setdefault("lambda", 0.20)

    uni = ensure_section("university_allocation")
    uni.setdefault("hh_intercept_mode", "fixed_amount")
    uni.setdefault("hh_intercept_amount", 150000)
    uni.setdefault("hh_coefficient", -135000)
    uni.setdefault("ss_intercept", 0.15)
    uni.setdefault("ss_coefficient", 0.40)
    uni.setdefault("ll_intercept", 0.85)
    uni.setdefault("ll_coefficient", -0.40)
    uni.setdefault("upkeep_intercept", 40000)
    uni.setdefault("upkeep_coefficient", 20000)

    tvet = ensure_section("tvet_allocation")
    tvet.setdefault("hh_base", 0.40)
    tvet.setdefault("hh_slope", 0.30)
    tvet.setdefault("ss_base", 0.15)
    tvet.setdefault("ss_slope", 0.40)
    tvet.setdefault("ll_base", 0.45)
    tvet.setdefault("ll_slope", 0.10)
    tvet.setdefault("upkeep_base", 13600)
    tvet.setdefault("upkeep_slope", 5000)

    hs = ensure_section("hh_safety")
    hs.setdefault("enabled", False)
    hs.setdefault("cap_amount", 150000)
    hs.setdefault("warning_threshold", 200000)
    hs.setdefault("hh_share_warning", 0.50)
    hs.setdefault("hh_increase_share_warning", 0.40)

    return merged


def safe_session_policy():
    p = deep_merge_policy(fresh_policy(), st.session_state.get("scenario_policy"))
    st.session_state["scenario_policy"] = p
    return p


def valid_result(obj):
    return (
        isinstance(obj, dict)
        and isinstance(obj.get("student_level"), pd.DataFrame)
        and not obj["student_level"].empty
    )

def result_or_none(obj):
    """Return a valid simulation result dict or None. Prevents None subscript errors."""
    return obj if valid_result(obj) else None


def result_student_df(obj):
    """Safely return student_level dataframe from a result object."""
    if not valid_result(obj):
        return pd.DataFrame()
    df = safe_get(obj, "student_level")
    return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()


def safe_get(obj, key, default=None):
    """Safe .get replacement: works even when obj is None or not a dict."""
    return obj.get(key, default) if isinstance(obj, dict) else default




def build_aggregate_from_student_df(df):
    """Build one-row aggregate table from student-level output when result['aggregate'] is missing/None."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = {}
    out["students"] = len(df)
    for col in ["MTI_final", "MTI_baseline", "MTI_after_equity"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            out[f"mean_{col}"] = vals.mean()
            out[f"median_{col}"] = vals.median()
    money_cols = ["ProgramCost", "PC_allocation", "HH", "SS", "LL", "Upkeep", "TotalLoan_with_Upkeep"]
    for col in money_cols:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").sum()
    pc = out.get("PC_allocation", out.get("ProgramCost", np.nan))
    if pc and pd.notna(pc) and pc != 0:
        for col in ["HH", "SS", "LL"]:
            if col in out:
                out[f"{col}_share"] = out[col] / pc
    if "TuitionIdentityCheck" in df.columns:
        out["max_identity_error"] = pd.to_numeric(df["TuitionIdentityCheck"], errors="coerce").abs().max()
    elif all(c in df.columns for c in ["HH", "SS", "LL", "PC_allocation"]):
        err = (
            pd.to_numeric(df["HH"], errors="coerce")
            + pd.to_numeric(df["SS"], errors="coerce")
            + pd.to_numeric(df["LL"], errors="coerce")
            - pd.to_numeric(df["PC_allocation"], errors="coerce")
        )
        out["max_identity_error"] = err.abs().max()
    return pd.DataFrame([out])


def safe_aggregate_df(result):
    """Return result aggregate if valid, otherwise rebuild from student_level."""
    agg = safe_get(result, "aggregate")
    if isinstance(agg, pd.DataFrame) and not agg.empty:
        return agg.copy()
    return build_aggregate_from_student_df(result_student_df(result))


def safe_compare_aggregate_outputs(base_result, scenario_result):
    """
    Safe replacement for compare_aggregate_outputs().
    Avoids NoneType errors when result['aggregate'] is missing or None.
    """
    base_agg = safe_aggregate_df(base_result)
    scen_agg = safe_aggregate_df(scenario_result)
    if base_agg.empty or scen_agg.empty:
        return pd.DataFrame()

    base = base_agg.iloc[0].drop("scenario", errors="ignore")
    scen = scen_agg.iloc[0].drop("scenario", errors="ignore")
    common = base.index.intersection(scen.index)

    rows = []
    for metric in common:
        b = pd.to_numeric(pd.Series([base.get(metric)]), errors="coerce").iloc[0]
        s = pd.to_numeric(pd.Series([scen.get(metric)]), errors="coerce").iloc[0]
        if pd.isna(b) and pd.isna(s):
            continue
        change = s - b
        pct_change = change / b if pd.notna(b) and b != 0 else np.nan
        rows.append({
            "metric": metric,
            "baseline": b,
            "scenario": s,
            "change": change,
            "pct_change": pct_change,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("metric")


def init_state():
    if "scenario_policy" not in st.session_state or not isinstance(st.session_state.get("scenario_policy"), dict):
        st.session_state["scenario_policy"] = fresh_policy()
    else:
        st.session_state["scenario_policy"] = deep_merge_policy(fresh_policy(), st.session_state.get("scenario_policy"))
    if "scenario_name" not in st.session_state:
        st.session_state["scenario_name"] = "Scenario 1"
    if "baseline_result" not in st.session_state:
        st.session_state["baseline_result"] = None
    if "scenarios" not in st.session_state or not isinstance(st.session_state["scenarios"], dict):
        st.session_state["scenarios"] = {}
    if "active_scenario" not in st.session_state:
        st.session_state["active_scenario"] = None
    if "analysis_population" not in st.session_state:
        st.session_state["analysis_population"] = "All"
    if "baseline_population" not in st.session_state:
        st.session_state["baseline_population"] = None


def reset_all():
    current_population = st.session_state.get("analysis_population", "All")
    st.session_state["scenario_policy"] = fresh_policy()
    st.session_state["scenario_name"] = "Scenario 1"
    st.session_state["baseline_result"] = None
    st.session_state["baseline_population"] = None
    st.session_state["scenarios"] = {}
    st.session_state["active_scenario"] = None
    st.session_state["analysis_population"] = current_population


def reset_to_baseline_keep_run():
    """Clear scenarios and policy edits, but keep the already-run baseline."""
    st.session_state["scenario_policy"] = fresh_policy()
    st.session_state["scenario_name"] = "Scenario 1"
    st.session_state["scenarios"] = {}
    st.session_state["active_scenario"] = None


def run_baseline_from_raw(raw_df, base_policy):
    """Run baseline only. Scenarios use this stored baseline until refreshed."""
    base_clean = clean_application_data(raw_df, base_policy)
    base_mapped = apply_fee_mapping(base_clean)
    baseline_run = run_scenario(base_mapped, base_policy, "Baseline", {})
    if not valid_result(baseline_run):
        raise RuntimeError("Baseline did not produce student-level output.")
    return baseline_run


def run_scenario_from_raw(raw_df, scenario_policy, scenario_name):
    """Run one named scenario against the already stored baseline."""
    scen_clean = clean_application_data(raw_df, scenario_policy)
    scen_mapped = apply_fee_mapping(scen_clean)
    scenario_run = run_scenario(scen_mapped, scenario_policy, scenario_name, {})
    if not valid_result(scenario_run):
        raise RuntimeError("Scenario did not produce student-level output.")
    return scenario_run


def smooth_density(series, points=300):
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([]), np.array([]), "empty"
    if values.size < 3 or np.nanstd(values) == 0:
        v = float(np.nanmedian(values))
        return np.array([v]), np.array([1.0]), "constant"
    lo = np.nanpercentile(values, 0.5)
    hi = np.nanpercentile(values, 99.5)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = np.nanmin(values)
        hi = np.nanmax(values)
    pad = (hi - lo) * 0.08 if hi > lo else 1
    x = np.linspace(lo - pad, hi + pad, points)
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(values)
        y = kde(x)
        return x, y, "kde"
    except Exception:
        counts, edges = np.histogram(values, bins=80, range=(lo, hi), density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        kernel = np.ones(7) / 7
        y = np.convolve(counts, kernel, mode="same")
        return centers, y, "smoothed_hist"


def series_stats(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"N": 0, "Mean": np.nan, "Median": np.nan, "Min": np.nan, "Max": np.nan, "SD": np.nan}
    return {"N": int(s.count()), "Mean": s.mean(), "Median": s.median(), "Min": s.min(), "Max": s.max(), "SD": s.std()}


def stats_text(stats, money=False):
    if stats.get("N", 0) == 0:
        return "N=0"
    if money:
        return f"N={stats['N']:,}\nMean={fmt_ksh(stats['Mean'])}\nMedian={fmt_ksh(stats['Median'])}\nMin={fmt_ksh(stats['Min'])}\nMax={fmt_ksh(stats['Max'])}"
    return f"N={stats['N']:,}\nMean={stats['Mean']:,.3f}\nMedian={stats['Median']:,.3f}\nMin={stats['Min']:,.3f}\nMax={stats['Max']:,.3f}"


def plot_smooth_distribution(base_series, scen_series, title, xlabel, money=False):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    bx, by, _ = smooth_density(base_series)
    sx, sy, _ = smooth_density(scen_series)
    if bx.size == 1:
        ax.axvline(bx[0], linewidth=2.5, label="Baseline")
    elif bx.size:
        ax.plot(bx, by, linewidth=2.5, label="Baseline")
    if sx.size == 1:
        ax.axvline(sx[0], linewidth=2.5, linestyle="--", label="Scenario")
    elif sx.size:
        ax.plot(sx, sy, linewidth=2.5, label="Scenario")
    note = "Baseline\n" + stats_text(series_stats(base_series), money) + "\n\nScenario\n" + stats_text(series_stats(scen_series), money)
    ax.set_title(f"{title}: Smooth distribution, baseline vs scenario")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    ax.text(0.015, 0.98, note, transform=ax.transAxes, va="top", fontsize=8, bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "0.6"})
    plt.tight_layout()
    return fig


def plot_change_distribution(base_series, scen_series, title, xlabel, money=False):
    base = pd.to_numeric(base_series, errors="coerce")
    scen = pd.to_numeric(scen_series, errors="coerce")
    idx = base.dropna().index.intersection(scen.dropna().index)
    change = (scen.loc[idx] - base.loc[idx]).dropna()
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x, y, _ = smooth_density(change)
    if x.size == 1:
        ax.axvline(x[0], linewidth=2.5, label="All comparable students have same change")
        ax.set_ylim(0, 1)
    elif x.size:
        ax.plot(x, y, linewidth=2.5, label="Scenario - Baseline")
        ax.axvline(change.mean(), linestyle="--", linewidth=2, label="Mean change")
        ax.axvline(change.median(), linestyle=":", linewidth=2, label="Median change")
    ax.set_title(f"{title}: Smooth change distribution")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    ax.text(0.015, 0.98, stats_text(series_stats(change), money), transform=ax.transAxes, va="top", fontsize=8, bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "0.6"})
    plt.tight_layout()
    return fig


def show_distribution_pair(base_series, scen_series, title, xlabel, change_xlabel, money=False):
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_smooth_distribution(base_series, scen_series, title, xlabel, money=money), use_container_width=True)
    with c2:
        st.pyplot(plot_change_distribution(base_series, scen_series, title, change_xlabel, money=money), use_container_width=True)


def component_summary(base_df, scen_df, cols):
    rows = []
    for col in cols:
        b = pd.to_numeric(base_df[col], errors="coerce").dropna()
        s = pd.to_numeric(scen_df[col], errors="coerce").dropna()
        idx = b.index.intersection(s.index)
        b = b.loc[idx]
        s = s.loc[idx]
        ch = s - b
        rows.append({
            "component": col,
            "baseline_mean": b.mean(), "scenario_mean": s.mean(), "mean_change": ch.mean(),
            "baseline_median": b.median(), "scenario_median": s.median(), "median_change": ch.median(),
            "baseline_min": b.min(), "scenario_min": s.min(), "min_change": ch.min(),
            "baseline_max": b.max(), "scenario_max": s.max(), "max_change": ch.max(),
            "baseline_sd": b.std(), "scenario_sd": s.std(), "change_sd": ch.std(),
            "share_increase": (ch > 0).mean(), "share_decrease": (ch < 0).mean(), "share_no_change": (ch == 0).mean(),
        })
    return pd.DataFrame(rows)


def mti_band_labels():
    return [f"{i}-{i + 10}" if i == 0 else f"{i + 1}-{i + 10}" for i in range(0, 100, 10)]


def assign_mti_band(series):
    values = pd.to_numeric(series, errors="coerce").clip(0, 100)
    bins = [-0.001] + list(range(10, 101, 10))
    return pd.cut(values, bins=bins, labels=mti_band_labels(), include_lowest=True, right=True)


def mti_frequency(df, col="MTI_final"):
    labels = mti_band_labels()
    bands = assign_mti_band(df[col])
    counts = bands.value_counts(sort=False).reindex(labels, fill_value=0).astype(int)
    out = pd.DataFrame({"MTI_class": labels, "students": counts.values})
    total = max(int(out["students"].sum()), 1)
    out["frequency_share"] = out["students"] / total
    out["cumulative_students"] = out["students"].cumsum()
    out["cumulative_share"] = out["cumulative_students"] / total
    return out


def aligned_student_frames(base_df, scen_df, id_col="user_id"):
    if id_col in base_df.columns and id_col in scen_df.columns:
        b = base_df.copy()
        s = scen_df.copy()
        b["_join_id"] = b[id_col].astype(str)
        s["_join_id"] = s[id_col].astype(str)
        return b.merge(s, on="_join_id", how="inner", suffixes=("_baseline", "_scenario"))
    b = base_df.reset_index().rename(columns={"index": "_join_id"})
    s = scen_df.reset_index().rename(columns={"index": "_join_id"})
    return b.merge(s, on="_join_id", how="inner", suffixes=("_baseline", "_scenario"))


def mti_transition(base_df, scen_df, id_col="user_id"):
    merged = aligned_student_frames(base_df, scen_df, id_col=id_col)
    labels = mti_band_labels()
    merged["baseline_band"] = assign_mti_band(merged["MTI_final_baseline"])
    merged["scenario_band"] = assign_mti_band(merged["MTI_final_scenario"])
    counts = pd.crosstab(merged["baseline_band"], merged["scenario_band"], dropna=False).reindex(index=labels, columns=labels, fill_value=0).astype(int)
    shares = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    moves = merged[["_join_id", "baseline_band", "scenario_band", "MTI_final_baseline", "MTI_final_scenario"]].copy()
    moves["MTI_change"] = moves["MTI_final_scenario"] - moves["MTI_final_baseline"]
    moves["movement"] = np.select([moves["MTI_change"] > 0, moves["MTI_change"] < 0], ["Moved up", "Moved down"], default="No change")
    summary = moves.groupby("movement", dropna=False).size().reset_index(name="students")
    summary["share"] = summary["students"] / max(len(moves), 1)
    return counts, shares, summary, moves


def compare_group(base_group, scen_group):
    if not isinstance(base_group, pd.DataFrame) or not isinstance(scen_group, pd.DataFrame) or base_group.empty or scen_group.empty:
        return pd.DataFrame()
    group_col = base_group.columns[0]
    metrics = ["students", "mean_mti", "program_cost", "HH", "SS", "LL", "Upkeep", "total_loan"]
    keep = [group_col] + [m for m in metrics if m in base_group.columns and m in scen_group.columns]
    out = base_group[keep].merge(scen_group[keep], on=group_col, how="outer", suffixes=("_baseline", "_scenario"))
    for m in metrics:
        b = f"{m}_baseline"
        s = f"{m}_scenario"
        if b in out.columns and s in out.columns:
            out[f"{m}_change"] = out[s].fillna(0) - out[b].fillna(0)
    return out


def allocation_policy_stats(base_df, scen_df, components):
    rows = []
    for col in components:
        if col not in base_df.columns or col not in scen_df.columns:
            continue
        b = pd.to_numeric(base_df[col], errors="coerce")
        sc = pd.to_numeric(scen_df[col], errors="coerce")
        idx = b.dropna().index.intersection(sc.dropna().index)
        b = b.loc[idx]
        sc = sc.loc[idx]
        ch = sc - b
        rows.append({
            "component": col,
            "baseline_mean": b.mean(), "scenario_mean": sc.mean(), "mean_change": ch.mean(),
            "baseline_median": b.median(), "scenario_median": sc.median(), "median_change": ch.median(),
            "baseline_min": b.min(), "scenario_min": sc.min(), "min_change": ch.min(),
            "baseline_max": b.max(), "scenario_max": sc.max(), "max_change": ch.max(),
            "students_increased": int((ch > 0).sum()), "students_decreased": int((ch < 0).sum()), "students_no_change": int((ch == 0).sum()),
            "share_increased": (ch > 0).mean(), "share_decreased": (ch < 0).mean(), "share_no_change": (ch == 0).mean(),
        })
    return pd.DataFrame(rows)


def student_extremes(base_df, scen_df, components, id_col="user_id", top_n=10):
    merged = aligned_student_frames(base_df, scen_df, id_col=id_col)
    keep_identity = [c for c in ["_join_id", f"{id_col}_baseline", f"{id_col}_scenario", "InstitutionName_baseline", "InstitutonName_baseline", "ProgramDescription_baseline", "County_baseline"] if c in merged.columns]
    out = {}
    for col in components:
        b = f"{col}_baseline"
        s_col = f"{col}_scenario"
        if b not in merged.columns or s_col not in merged.columns:
            continue
        tmp = merged[keep_identity + [b, s_col]].copy()
        tmp[f"{col}_change"] = pd.to_numeric(tmp[s_col], errors="coerce") - pd.to_numeric(tmp[b], errors="coerce")
        out[col] = {
            "lowest_scenario": tmp.sort_values(s_col, ascending=True).head(top_n),
            "highest_scenario": tmp.sort_values(s_col, ascending=False).head(top_n),
            "largest_increase": tmp.sort_values(f"{col}_change", ascending=False).head(top_n),
            "largest_decrease": tmp.sort_values(f"{col}_change", ascending=True).head(top_n),
        }
    return out




def programme_cost_band(series):
    values = pd.to_numeric(series, errors="coerce")
    bins = [-0.001, 100000, 150000, 200000, 300000, np.inf]
    labels = ["<=100K", "100K-150K", "150K-200K", "200K-300K", ">300K"]
    return pd.cut(values, bins=bins, labels=labels, include_lowest=True, right=True)


def fiscal_savings_panel(agg_compare):
    rows = []
    for metric in ["HH", "SS", "LL", "Upkeep", "TotalLoan_with_Upkeep"]:
        if metric in agg_compare.index:
            rows.append({
                "metric": metric,
                "baseline": agg_compare.loc[metric, "baseline"],
                "scenario": agg_compare.loc[metric, "scenario"],
                "change": agg_compare.loc[metric, "change"],
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out, {}
    hh_change = float(agg_compare.loc["HH", "change"]) if "HH" in agg_compare.index else 0.0
    ss_change = float(agg_compare.loc["SS", "change"]) if "SS" in agg_compare.index else 0.0
    ll_change = float(agg_compare.loc["LL", "change"]) if "LL" in agg_compare.index else 0.0
    gov_change = ss_change + ll_change
    summary = {
        "household_shift": hh_change,
        "scholarship_change": ss_change,
        "loan_change": ll_change,
        "government_tuition_change": gov_change,
        "government_tuition_saving": -gov_change,
    }
    return out, summary


def winners_losers_summary(base_df, scen_df, components, id_col="user_id"):
    merged = aligned_student_frames(base_df, scen_df, id_col=id_col)
    rows = []
    for col in components:
        b = f"{col}_baseline"
        s = f"{col}_scenario"
        if b not in merged.columns or s not in merged.columns:
            continue
        change = pd.to_numeric(merged[s], errors="coerce") - pd.to_numeric(merged[b], errors="coerce")
        inc = change[change > 0]
        dec = change[change < 0]
        same = change[change == 0]
        rows.append({
            "component": col,
            "students": int(change.notna().sum()),
            "students_increased": int(inc.count()),
            "share_increased": inc.count() / max(change.notna().sum(), 1),
            "mean_increase": inc.mean(),
            "students_decreased": int(dec.count()),
            "share_decreased": dec.count() / max(change.notna().sum(), 1),
            "mean_decrease": dec.mean(),
            "students_no_change": int(same.count()),
            "share_no_change": same.count() / max(change.notna().sum(), 1),
            "largest_increase": change.max(),
            "largest_decrease": change.min(),
        })
    return pd.DataFrame(rows)


def programme_cost_band_summary(base_df, scen_df, id_col="user_id"):
    merged = aligned_student_frames(base_df, scen_df, id_col=id_col)
    pc_col = "ProgramCost_baseline" if "ProgramCost_baseline" in merged.columns else "PC_allocation_baseline"
    if pc_col not in merged.columns:
        return pd.DataFrame()
    merged["programme_cost_band"] = programme_cost_band(merged[pc_col])
    for col in ["HH", "SS", "LL", "Upkeep", "TotalLoan_with_Upkeep"]:
        b = f"{col}_baseline"
        s = f"{col}_scenario"
        if b in merged.columns and s in merged.columns:
            merged[f"{col}_change"] = pd.to_numeric(merged[s], errors="coerce") - pd.to_numeric(merged[b], errors="coerce")
    agg_dict = {"_join_id": "count"}
    for col in ["HH_change", "SS_change", "LL_change", "Upkeep_change", "TotalLoan_with_Upkeep_change"]:
        if col in merged.columns:
            agg_dict[col] = ["mean", "median", "min", "max"]
    out = merged.groupby("programme_cost_band", dropna=False).agg(agg_dict)
    out.columns = ["students" if a == "_join_id" else f"{a}_{b}" for a, b in out.columns]
    out = out.reset_index()
    return out


def policy_warning_table(base_df, scen_df, agg_compare, scenario_policy):
    warnings = []
    ua = scenario_policy.get("university_allocation", {}) if isinstance(scenario_policy, dict) else {}
    safety = scenario_policy.get("hh_safety", {}) if isinstance(scenario_policy, dict) else {}
    hh_share = None
    if "HH_share" in agg_compare.index:
        hh_share = float(agg_compare.loc["HH_share", "scenario"])
        if hh_share > float(safety.get("hh_share_warning", 0.50)):
            warnings.append({"level": "High", "warning": "Household share exceeds policy warning threshold", "value": fmt_share(hh_share), "policy_meaning": "Households are financing most tuition cost."})
    if "HH" in scen_df.columns:
        max_hh = pd.to_numeric(scen_df["HH"], errors="coerce").max()
        threshold = float(safety.get("warning_threshold", 200000))
        if max_hh > threshold:
            warnings.append({"level": "High", "warning": "Maximum HH exceeds warning threshold", "value": fmt_ksh(max_hh), "policy_meaning": "Some students face very high household contribution."})
    if "HH" in base_df.columns and "HH" in scen_df.columns:
        ch = pd.to_numeric(scen_df["HH"], errors="coerce") - pd.to_numeric(base_df["HH"], errors="coerce")
        share_inc = (ch > 0).mean()
        if share_inc > float(safety.get("hh_increase_share_warning", 0.40)):
            warnings.append({"level": "Medium", "warning": "Large share of students have increased HH", "value": fmt_share(share_inc), "policy_meaning": "Policy shifts burden to many households."})
    if ua.get("hh_intercept_mode") == "programme_cost" and not safety.get("enabled", False):
        warnings.append({"level": "Medium", "warning": "Programme-cost HH mode without HH safety cap", "value": "No cap", "policy_meaning": "HH scales with programme cost and may create high-cost programme shocks."})
    return pd.DataFrame(warnings)

# =====================================================
# PUBLIC / PRIVATE UNIVERSITY AND TVET CLASSIFICATION
# =====================================================

PUBLIC_UNIVERSITY_NAMES = {
    "ALUPE UNIVERSITY", "ALUPE UNIVERSITY COLLEGE", "BOMET UNIVERSITY COLLEGE",
    "CHUKA UNIVERSITY", "CO-OPERATIVE UNIVERSITY OF KENYA",
    "DEDAN KIMATHI UNIVERSITY OF TECHNOLOGY", "EGERTON UNIVERSITY",
    "GARISSA UNIVERSITY", "JARAMOGI OGINGA ODINGA UNIVERSITY OF SCIENCE AND TECHNOLOGY",
    "JOMO KENYATTA UNIVERSITY OF AGRICULTURE AND TECHNOLOGY", "KABARNET UNIVERSITY COLLEGE",
    "KAIMOSI FRIENDS UNIVERSITY", "KARATINA UNIVERSITY", "KENYATTA UNIVERSITY",
    "KIBABII UNIVERSITY", "KIRINYAGA UNIVERSITY", "KISII UNIVERSITY",
    "KOITALEL SAMOEI UNIVERSITY COLLEGE", "LAIKIPIA UNIVERSITY", "MACHAKOS UNIVERSITY",
    "MAASAI MARA UNIVERSITY", "MAMA NGINA UNIVERSITY COLLEGE", "MASENO UNIVERSITY",
    "MASINDE MULIRO UNIVERSITY OF SCIENCE AND TECHNOLOGY", "MERU UNIVERSITY OF SCIENCE AND TECHNOLOGY",
    "MOI UNIVERSITY", "MULTIMEDIA UNIVERSITY", "MULTIMEDIA UNIVERSITY OF KENYA",
    "MURANGA UNIVERSITY OF TECHNOLOGY", "MURANG'A UNIVERSITY OF TECHNOLOGY",
    "NYANDARUA UNIVERSITY COLLEGE", "PWANI UNIVERSITY", "RONGO UNIVERSITY",
    "SOUTH EASTERN KENYA UNIVERSITY", "TAITA TAVETA UNIVERSITY", "TECHNICAL UNIVERSITY OF KENYA",
    "TECHNICAL UNIVERSITY OF MOMBASA", "THARAKA UNIVERSITY", "TOM MBOYA UNIVERSITY",
    "TURKANA UNIVERSITY COLLEGE", "UNIVERSITY OF ELDORET", "UNIVERSITY OF EMBU",
    "UNIVERSITY OF KABIANGA", "UNIVERSITY OF NAIROBI"
}


def normalize_institution_name(value):
    x = str(value).upper().strip()
    x = x.replace("’", "'").replace("`", "'")
    return " ".join(x.split())


def institution_col(df):
    if "InstitutionName" in df.columns:
        return "InstitutionName"
    if "InstitutonName" in df.columns:
        return "InstitutonName"
    return None


def add_track_and_ownership(df):
    out = df.copy()
    if "is_tvet" in out.columns:
        tvet_mask = pd.to_numeric(out["is_tvet"], errors="coerce").fillna(0).astype(int).eq(1)
    else:
        level = out.get("StudyLevel", pd.Series("", index=out.index)).astype(str).str.upper()
        tvet_mask = level.str.contains("TVET|TECHNICAL|VOCATIONAL|DIPLOMA|CERTIFICATE", na=False)
    out["education_track"] = np.where(tvet_mask, "TVET", "University")
    inst = institution_col(out)
    if inst is not None:
        public_mask = out[inst].map(normalize_institution_name).isin(PUBLIC_UNIVERSITY_NAMES)
    else:
        public_mask = pd.Series(False, index=out.index)
    out["ownership_group"] = np.select(
        [tvet_mask, public_mask],
        ["TVET", "Public University"],
        default="Private University / Other"
    )
    out["sector_group"] = out["ownership_group"]
    return out


def aggregate_student_level(df, group_cols):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    group_cols = [c for c in group_cols if c in df.columns]
    if not group_cols:
        return pd.DataFrame()

    tmp = df.copy()
    for col in ["MTI_final", "PC_allocation", "HH", "SS", "LL", "Upkeep", "TotalLoan_with_Upkeep"]:
        if col not in tmp.columns:
            tmp[col] = np.nan

    # Count rows using a guaranteed helper column, not MTI_final, because MTI_final may be absent/all missing.
    tmp["_row_count_for_agg"] = 1

    agg = (
        tmp.groupby(group_cols, dropna=False)
        .agg(
            students=("_row_count_for_agg", "sum"),
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

    denom = agg["program_cost"].replace(0, np.nan)
    agg["HH_share"] = agg["HH"] / denom
    agg["SS_share"] = agg["SS"] / denom
    agg["LL_share"] = agg["LL"] / denom
    return agg


def compare_student_aggregates(base_df, scen_df, group_cols):
    base = aggregate_student_level(base_df, group_cols)
    scen = aggregate_student_level(scen_df, group_cols)
    if base.empty or scen.empty:
        return pd.DataFrame()
    out = base.merge(scen, on=group_cols, how="outer", suffixes=("_baseline", "_scenario"))
    metrics = ["students", "mean_mti", "median_mti", "program_cost", "HH", "SS", "LL", "Upkeep", "total_loan", "HH_share", "SS_share", "LL_share"]
    for m in metrics:
        b = f"{m}_baseline"
        s = f"{m}_scenario"
        if b in out.columns and s in out.columns:
            out[f"{m}_change"] = out[s].fillna(0) - out[b].fillna(0)
    return out


def segment_filter(df, segment):
    if segment == "Public University":
        return df[df["ownership_group"].eq("Public University")].copy()
    if segment == "Private University / Other":
        return df[df["ownership_group"].eq("Private University / Other")].copy()
    if segment == "TVET":
        return df[df["education_track"].eq("TVET")].copy()
    return df.copy()


def show_segment_detail(segment_name, base_seg, scen_seg):
    st.subheader(segment_name)
    if scen_seg.empty:
        st.info(f"No records found for {segment_name}.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students", f"{len(scen_seg):,}")
    c2.metric("Mean MTI", f"{pd.to_numeric(scen_seg['MTI_final'], errors='coerce').mean():,.2f}")
    c3.metric("HH total", fmt_ksh(pd.to_numeric(scen_seg['HH'], errors='coerce').sum()))
    c4.metric("Gov tuition", fmt_ksh(pd.to_numeric(scen_seg['SS'], errors='coerce').sum() + pd.to_numeric(scen_seg['LL'], errors='coerce').sum()))
    seg_summary = compare_student_aggregates(base_seg, scen_seg, ["ownership_group"])
    if not seg_summary.empty:
        st.dataframe(style_table(seg_summary), use_container_width=True, hide_index=True)
    inst = institution_col(scen_seg)
    tab_a, tab_b, tab_c, tab_d = st.tabs(["Institutions", "Programmes", "Counties", "Allocation diagnostics"])
    with tab_a:
        if inst is not None:
            inst_tbl = compare_student_aggregates(base_seg, scen_seg, [inst])
            if not inst_tbl.empty:
                st.dataframe(style_table(inst_tbl.sort_values("program_cost_scenario", ascending=False)), use_container_width=True, hide_index=True)
            else:
                st.info("Institution table not available.")
        else:
            st.info("Institution column not available.")
    with tab_b:
        if "ProgramDescription" in scen_seg.columns:
            prog_tbl = compare_student_aggregates(base_seg, scen_seg, ["ProgramDescription"])
            if not prog_tbl.empty:
                st.dataframe(style_table(prog_tbl.sort_values("program_cost_scenario", ascending=False).head(100)), use_container_width=True, hide_index=True)
            else:
                st.info("Programme table not available.")
        else:
            st.info("Programme column not available.")
    with tab_c:
        if "County" in scen_seg.columns:
            county_tbl = compare_student_aggregates(base_seg, scen_seg, ["County"])
            if not county_tbl.empty:
                st.dataframe(style_table(county_tbl.sort_values("program_cost_scenario", ascending=False)), use_container_width=True, hide_index=True)
            else:
                st.info("County table not available.")
        else:
            st.info("County column not available.")
    with tab_d:
        diag = allocation_policy_stats(base_seg, scen_seg, ["HH", "SS", "LL", "Upkeep", "TotalLoan_with_Upkeep"])
        if not diag.empty:
            st.dataframe(style_table(diag), use_container_width=True, hide_index=True)
        opts = [c for c in ["HH", "SS", "LL", "Upkeep", "TotalLoan_with_Upkeep"] if c in base_seg.columns and c in scen_seg.columns]
        if opts:
            alloc_col = st.selectbox(f"Distribution component - {segment_name}", opts, key=f"dist_{segment_name}")
            show_distribution_pair(base_seg[alloc_col], scen_seg[alloc_col], f"{segment_name}: {alloc_col}", "KSh", f"{alloc_col} Change", money=True)



# =====================================================
# ONSET ANALYSIS POPULATION FILTER
# =====================================================

ANALYSIS_POPULATION_OPTIONS = [
    "All",
    "Universities only",
    "Public universities only",
    "Private universities only",
    "TVET only",
]


def filter_analysis_population_from_raw(raw_df, policy, selected_population):
    """
    Clean + fee-map raw data, classify education track/ownership, then filter
    BEFORE baseline/scenario is run. This guarantees that baseline, scenario,
    distributions, transitions, totals and downloads all refer to the same
    selected analysis population.
    """
    clean = clean_application_data(raw_df, policy)
    mapped = apply_fee_mapping(clean)
    classified = add_track_and_ownership(mapped)

    if selected_population == "All":
        filtered = classified.copy()
    elif selected_population == "Universities only":
        filtered = classified[classified["education_track"].eq("University")].copy()
    elif selected_population == "Public universities only":
        filtered = classified[classified["ownership_group"].eq("Public University")].copy()
    elif selected_population == "Private universities only":
        filtered = classified[classified["ownership_group"].eq("Private University / Other")].copy()
    elif selected_population == "TVET only":
        filtered = classified[classified["education_track"].eq("TVET")].copy()
    else:
        filtered = classified.copy()

    if filtered.empty:
        raise RuntimeError(f"No records found for selected population: {selected_population}")

    return filtered


def population_counts_from_raw(raw_df, policy):
    """Counts used only to guide the user before running baseline."""
    try:
        clean = clean_application_data(raw_df, policy)
        mapped = apply_fee_mapping(clean)
        classified = add_track_and_ownership(mapped)
        return {
            "All": len(classified),
            "Universities only": int(classified["education_track"].eq("University").sum()),
            "Public universities only": int(classified["ownership_group"].eq("Public University").sum()),
            "Private universities only": int(classified["ownership_group"].eq("Private University / Other").sum()),
            "TVET only": int(classified["education_track"].eq("TVET").sum()),
        }
    except Exception:
        return {k: None for k in ANALYSIS_POPULATION_OPTIONS}


def run_baseline_filtered(raw_df, base_policy, selected_population):
    filtered = filter_analysis_population_from_raw(raw_df, base_policy, selected_population)
    baseline_run = run_scenario(filtered, base_policy, f"Baseline - {selected_population}", {})
    if not valid_result(baseline_run):
        raise RuntimeError("Baseline did not produce student-level output.")
    baseline_run["analysis_population"] = selected_population
    return baseline_run


def run_scenario_filtered(raw_df, scenario_policy, scenario_name, selected_population):
    filtered = filter_analysis_population_from_raw(raw_df, scenario_policy, selected_population)
    scenario_run = run_scenario(filtered, scenario_policy, scenario_name, {})
    if not valid_result(scenario_run):
        raise RuntimeError("Scenario did not produce student-level output.")
    scenario_run["analysis_population"] = selected_population
    return scenario_run


def ensure_required_simulation_columns(df, label):
    """
    Verify that a simulated student-level dataframe has the columns required
    for MTI/allocation analysis. Returns a list of missing columns.
    """
    required = ["MTI_final", "HH", "SS", "LL", "Upkeep", "PC_allocation"]
    if not isinstance(df, pd.DataFrame) or df.empty:
        return required
    return [c for c in required if c not in df.columns]


def safe_numeric_col(df, col):
    """Return numeric series if column exists, otherwise a NaN series aligned to df."""
    if isinstance(df, pd.DataFrame) and col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    if isinstance(df, pd.DataFrame):
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.Series(dtype=float)


def safe_mti_frequency(df, col="MTI_final"):
    if not isinstance(df, pd.DataFrame) or col not in df.columns:
        labels = mti_band_labels()
        out = pd.DataFrame({"MTI_class": labels, "students": [0] * len(labels)})
        out["frequency_share"] = 0.0
        out["cumulative_students"] = 0
        out["cumulative_share"] = 0.0
        return out
    return mti_frequency(df, col=col)


def safe_mti_transition(base_df, scen_df, id_col="user_id"):
    if "MTI_final" not in base_df.columns or "MTI_final" not in scen_df.columns:
        labels = mti_band_labels()
        counts = pd.DataFrame(0, index=labels, columns=labels)
        shares = counts.astype(float)
        summary = pd.DataFrame({"movement": ["No MTI output"], "students": [0], "share": [0.0]})
        return counts, shares, summary, pd.DataFrame()
    return mti_transition(base_df, scen_df, id_col=id_col)


def baseline_aggregate_from_student_df(df):
    """Build a readable baseline aggregate table from one student-level dataframe."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    rows = {
        "students": len(df),
        "mean_mti": safe_numeric_col(df, "MTI_final").mean(),
        "median_mti": safe_numeric_col(df, "MTI_final").median(),
        "program_cost": safe_numeric_col(df, "PC_allocation").sum(),
        "HH": safe_numeric_col(df, "HH").sum(),
        "SS": safe_numeric_col(df, "SS").sum(),
        "LL": safe_numeric_col(df, "LL").sum(),
        "Upkeep": safe_numeric_col(df, "Upkeep").sum(),
        "TotalLoan_with_Upkeep": safe_numeric_col(df, "TotalLoan_with_Upkeep").sum(),
    }
    pc = rows.get("program_cost", np.nan)
    if pd.notna(pc) and pc != 0:
        rows["HH_share"] = rows["HH"] / pc
        rows["SS_share"] = rows["SS"] / pc
        rows["LL_share"] = rows["LL"] / pc
    return pd.DataFrame({"baseline": rows})


def baseline_component_summary(df, cols):
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        s = safe_numeric_col(df, col).dropna()
        if s.empty:
            continue
        rows.append({
            "component": col,
            "students": int(s.count()),
            "mean": s.mean(),
            "median": s.median(),
            "min": s.min(),
            "max": s.max(),
            "sd": s.std(),
        })
    return pd.DataFrame(rows)


def baseline_allocation_stats(df, components):
    rows = []
    for col in components:
        if col not in df.columns:
            continue
        s = safe_numeric_col(df, col).dropna()
        if s.empty:
            continue
        rows.append({
            "component": col,
            "students": int(s.count()),
            "mean": s.mean(),
            "median": s.median(),
            "min": s.min(),
            "max": s.max(),
            "sd": s.std(),
            "total": s.sum(),
        })
    return pd.DataFrame(rows)


def plot_single_smooth_distribution(series, title, xlabel, money=False):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x, y, _ = smooth_density(series)
    if x.size == 1:
        ax.axvline(x[0], linewidth=2.5, label="Baseline")
        ax.set_ylim(0, 1)
    elif x.size:
        ax.plot(x, y, linewidth=2.5, label="Baseline")
    ax.set_title(f"{title}: Baseline smooth distribution")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    ax.text(
        0.015,
        0.98,
        stats_text(series_stats(series), money),
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "0.6"},
    )
    plt.tight_layout()
    return fig


def show_baseline_only_dashboard(baseline, analysis_population):
    """Full baseline dashboard shown before any scenario is created."""
    base_student_df = result_student_df(baseline)
    if base_student_df.empty:
        st.warning("Baseline exists but has no valid student-level output. Click Run / Refresh Baseline.")
        st.stop()

    base_df = add_track_and_ownership(base_student_df)
    missing = ensure_required_simulation_columns(base_df, "Baseline")
    if missing:
        st.warning(
            "Baseline output is missing some expected simulation columns. "
            f"Missing columns: {missing}. Sections requiring missing fields will be skipped."
        )

    st.success("Baseline is ready. Now edit policy levers and click **Create / Update Scenario** when you want to compare policy changes.")
    st.header(f"Baseline Dashboard | Population: {analysis_population}")

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Students", f"{len(base_df):,}")
    mean_mti_val = safe_numeric_col(base_df, "MTI_final").mean()
    b2.metric("Mean MTI", "N/A" if pd.isna(mean_mti_val) else f"{mean_mti_val:,.2f}")
    b3.metric("Programme cost", fmt_ksh(safe_numeric_col(base_df, "PC_allocation").sum()))
    b4.metric("Gov tuition", fmt_ksh(safe_numeric_col(base_df, "SS").sum() + safe_numeric_col(base_df, "LL").sum()))

    st.subheader("Baseline aggregate financing structure")
    baseline_agg = baseline_aggregate_from_student_df(base_df)
    if not baseline_agg.empty:
        st.dataframe(style_indexed(baseline_agg), use_container_width=True)

    st.subheader("Public / Private University and TVET baseline split")
    sector_base = aggregate_student_level(base_df, ["education_track", "ownership_group"])
    if not sector_base.empty:
        st.dataframe(style_table(sector_base), use_container_width=True, hide_index=True)
    else:
        st.info("Sector baseline split is not available.")

    st.subheader("Baseline MTI distribution")
    if "MTI_final" in base_df.columns:
        st.pyplot(plot_single_smooth_distribution(base_df["MTI_final"] if "MTI_final" in base_df.columns else pd.Series(dtype=float), "Final MTI", "MTI Score", money=False), use_container_width=True)
    else:
        st.info("MTI_final is not available in the baseline output, so the MTI distribution is skipped.")

    st.subheader("Baseline MTI class frequency and cumulative frequency")
    freq_base = safe_mti_frequency(base_df).rename(columns={
        "students": "baseline_students",
        "frequency_share": "baseline_frequency_share",
        "cumulative_students": "baseline_cumulative_students",
        "cumulative_share": "baseline_cumulative_share",
    })
    st.dataframe(style_table(freq_base), use_container_width=True, hide_index=True)

    st.subheader("Baseline MTI component statistics")
    score_components = ["S_primary", "S_secondary", "S_poverty", "S_family", "MTI_baseline", "MTI_after_equity", "MTI_final", "IncomeAdjustmentRatio"]
    available_components = [c for c in score_components if c in base_df.columns]
    comp_base = baseline_component_summary(base_df, available_components)
    if not comp_base.empty:
        st.dataframe(style_table(comp_base), use_container_width=True, hide_index=True)
        selected_score = st.selectbox("Baseline MTI score/component", available_components, key="baseline_component_dist")
        st.pyplot(plot_single_smooth_distribution(base_df[selected_score], selected_score, "Score", money=False), use_container_width=True)
    else:
        st.info("No baseline MTI component columns found.")

    st.subheader("Baseline allocation diagnostics")
    alloc_components = [c for c in ["HH", "SS", "LL", "Upkeep", "TotalLoan_with_Upkeep", "SS_gap_share", "LL_gap_share"] if c in base_df.columns]
    alloc_base = baseline_allocation_stats(base_df, alloc_components)
    if not alloc_base.empty:
        st.caption("Min/max are student-level extremes. Total is aggregate amount where applicable.")
        st.dataframe(style_table(alloc_base), use_container_width=True, hide_index=True)
        selected_alloc = st.selectbox("Baseline allocation component", alloc_components, key="baseline_alloc_dist")
        is_money = selected_alloc not in {"SS_gap_share", "LL_gap_share"}
        st.pyplot(
            plot_single_smooth_distribution(base_df[selected_alloc], selected_alloc, "KSh" if is_money else "Share", money=is_money),
            use_container_width=True,
        )

    st.subheader("Baseline institution, programme and county tables")
    inst = institution_col(base_df)
    tab1, tab2, tab3 = st.tabs(["Institutions", "Programmes", "Counties"])
    with tab1:
        if inst:
            inst_tbl = aggregate_student_level(base_df, [inst])
            if not inst_tbl.empty:
                st.dataframe(style_table(inst_tbl.sort_values("program_cost", ascending=False)), use_container_width=True, hide_index=True)
            else:
                st.info("Institution table not available.")
        else:
            st.info("Institution column not available.")
    with tab2:
        if "ProgramDescription" in base_df.columns:
            prog_tbl = aggregate_student_level(base_df, ["ProgramDescription"])
            if not prog_tbl.empty:
                st.dataframe(style_table(prog_tbl.sort_values("program_cost", ascending=False).head(200)), use_container_width=True, hide_index=True)
            else:
                st.info("Programme table not available.")
        else:
            st.info("Programme column not available.")
    with tab3:
        if "County" in base_df.columns:
            county_tbl = aggregate_student_level(base_df, ["County"])
            if not county_tbl.empty:
                st.dataframe(style_table(county_tbl.sort_values("program_cost", ascending=False)), use_container_width=True, hide_index=True)
            else:
                st.info("County table not available.")
        else:
            st.info("County column not available.")

    st.subheader("Baseline detailed views by segment")
    seg_tabs = st.tabs(["Public Universities", "Private / Other Universities", "TVET"])
    for tab, seg in zip(seg_tabs, ["Public University", "Private University / Other", "TVET"]):
        with tab:
            seg_df = segment_filter(base_df, seg)
            st.write(f"Records: {len(seg_df):,}")
            if seg_df.empty:
                st.info(f"No records found for {seg}.")
            else:
                seg_agg = baseline_aggregate_from_student_df(seg_df)
                st.dataframe(style_indexed(seg_agg), use_container_width=True)
                seg_inst = institution_col(seg_df)
                if seg_inst:
                    seg_tbl = aggregate_student_level(seg_df, [seg_inst])
                    st.dataframe(style_table(seg_tbl.sort_values("program_cost", ascending=False)), use_container_width=True, hide_index=True)

    st.subheader("Download baseline outputs")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button("Download baseline student-level CSV", base_df.to_csv(index=False).encode("utf-8"), f"baseline_{analysis_population}_student_level.csv", "text/csv")
    with d2:
        st.download_button("Download baseline aggregate CSV", baseline_agg.to_csv().encode("utf-8"), f"baseline_{analysis_population}_aggregate.csv", "text/csv")
    with d3:
        st.download_button("Download baseline sector split CSV", sector_base.to_csv(index=False).encode("utf-8"), f"baseline_{analysis_population}_sector_split.csv", "text/csv")

    st.caption("Reset to baseline clears scenarios and returns policy sliders to config.py while keeping this stored baseline run.")
    st.stop()


def plot_transition_heatmap(counts):
    fig, ax = plt.subplots(figsize=(9, 7))
    data = counts.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto")
    ax.set_xticks(np.arange(len(counts.columns)))
    ax.set_yticks(np.arange(len(counts.index)))
    ax.set_xticklabels(counts.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(counts.index, fontsize=8)
    ax.set_xlabel("Scenario MTI class")
    ax.set_ylabel("Baseline MTI class")
    ax.set_title("MTI transition heatmap: baseline class to scenario class")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] > 0:
                ax.text(j, i, f"{int(data[i, j]):,}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


init_state()
base_policy = fresh_policy()

st.title("MTI Policy Lab")
st.caption("Select analysis population first, run baseline for that population, then create scenarios against that stored baseline.")

st.sidebar.header("1. Data Source")

# Deployment-ready data loading:
# 1) Prefer Streamlit secret DATA_URL if configured.
# 2) Else use bundled repo file Application Data 2025_2026.csv if it exists.
# 3) Else allow manual upload as a fallback.
#
# For PS/easy live use, push the CSV to the repo OR set DATA_URL in Streamlit secrets.
# Example Streamlit secret:
# DATA_URL = "https://raw.githubusercontent.com/cyrusCMM/MTI_policy_lab/main/Application%20Data%202025_2026.csv"

DEFAULT_DATA_FILE = Path("Application Data 2025_2026.csv")
DATA_URL = None

try:
    DATA_URL = st.secrets.get("DATA_URL", None)
except Exception:
    DATA_URL = None

uploaded_file = st.sidebar.file_uploader(
    "Optional: upload a different intake CSV",
    type=["csv"],
    help="The app automatically loads the live/default dataset. Upload only if you want to override it."
)

raw_df = None
data_name = None

try:
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        data_name = f"Uploaded file: {uploaded_file.name}"
    elif DATA_URL:
        raw_df = pd.read_csv(DATA_URL)
        data_name = "Live data from configured DATA_URL"
    elif DEFAULT_DATA_FILE.exists():
        raw_df = pd.read_csv(DEFAULT_DATA_FILE)
        data_name = f"Bundled repo data: {DEFAULT_DATA_FILE.name}"
    else:
        raw_df = None
        data_name = None
except Exception as exc:
    st.error("Could not read the intake CSV from upload, DATA_URL, or bundled repo file.")
    st.exception(exc)
    st.stop()

if not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
    st.error(
        "No intake data found. For live deployment, either push Application Data 2025_2026.csv "
        "to the GitHub repo, or set DATA_URL in Streamlit secrets."
    )
    st.stop()

st.sidebar.success(f"Loaded: {data_name} | Rows: {len(raw_df):,}")


st.sidebar.header("2. Analysis Population")
pop_counts = population_counts_from_raw(raw_df, base_policy)
pop_labels = []
for opt in ANALYSIS_POPULATION_OPTIONS:
    n = pop_counts.get(opt)
    pop_labels.append(f"{opt} ({n:,})" if isinstance(n, int) else opt)

current_pop = st.session_state.get("analysis_population", "All")
current_idx = ANALYSIS_POPULATION_OPTIONS.index(current_pop) if current_pop in ANALYSIS_POPULATION_OPTIONS else 0
selected_label = st.sidebar.selectbox(
    "Select population before running baseline",
    pop_labels,
    index=current_idx,
    help="Baseline and scenarios are run only on this selected population."
)
analysis_population = ANALYSIS_POPULATION_OPTIONS[pop_labels.index(selected_label)]

if analysis_population != st.session_state.get("analysis_population", "All"):
    st.session_state["analysis_population"] = analysis_population
    st.session_state["baseline_result"] = None
    st.session_state["baseline_population"] = None
    st.session_state["scenarios"] = {}
    st.session_state["active_scenario"] = None
    st.session_state["scenario_policy"] = fresh_policy()
    st.rerun()

st.sidebar.info(
    f"Analysis population: {analysis_population}"
    + (f" | Rows: {pop_counts.get(analysis_population):,}" if isinstance(pop_counts.get(analysis_population), int) else "")
)


st.sidebar.header("3. Baseline Control")
run_baseline_clicked = st.sidebar.button("Run / Refresh Baseline", type="primary")

if run_baseline_clicked:
    with st.spinner("Running baseline from the fixed config.py policy..."):
        try:
            baseline_run = run_baseline_filtered(raw_df, base_policy, analysis_population)
            st.session_state["baseline_result"] = baseline_run
            st.session_state["baseline_population"] = analysis_population
            st.session_state["scenarios"] = {}
            st.session_state["active_scenario"] = None
            st.session_state["scenario_policy"] = fresh_policy()
            st.success("Baseline saved. You can now create scenarios.")
        except Exception as exc:
            st.error("Baseline failed before outputs were created.")
            st.exception(exc)

baseline_obj_sidebar = result_or_none(st.session_state.get("baseline_result"))
baseline_pop_sidebar = None
if baseline_obj_sidebar is not None:
    baseline_pop_sidebar = st.session_state.get("baseline_population") or safe_get(baseline_obj_sidebar, "analysis_population")
baseline_ready_sidebar = baseline_obj_sidebar is not None and baseline_pop_sidebar == analysis_population

if baseline_ready_sidebar:
    base_rows_sidebar = len(result_student_df(baseline_obj_sidebar))
    st.sidebar.success(f"Baseline ready for {baseline_pop_sidebar} | Rows: {base_rows_sidebar:,}")
elif baseline_obj_sidebar is not None and baseline_pop_sidebar != analysis_population:
    st.sidebar.warning("Baseline exists for a different population. Click Run / Refresh Baseline.")
else:
    st.sidebar.warning("Run baseline for the selected population before creating scenarios.")

st.sidebar.header("4. Scenario Control")
st.session_state["scenario_name"] = st.sidebar.text_input("New scenario name", value=st.session_state["scenario_name"], disabled=not baseline_ready_sidebar)

if not isinstance(st.session_state.get("scenarios"), dict):
    st.session_state["scenarios"] = {}
saved_names = list(st.session_state.get("scenarios") if isinstance(st.session_state.get("scenarios"), dict) else {}.keys())
if saved_names:
    current_active = st.session_state.get("active_scenario")
    idx = saved_names.index(current_active) if current_active in saved_names else 0
    st.session_state["active_scenario"] = st.sidebar.selectbox("Compare saved scenario", saved_names, index=idx)
else:
    st.sidebar.info("No saved scenarios yet.")

ca, cb = st.sidebar.columns(2)
with ca:
    if st.button("Reset to baseline"):
        reset_to_baseline_keep_run()
        st.rerun()
with cb:
    if st.button("Clear all"):
        reset_all()
        st.rerun()

old_policy = copy.deepcopy(safe_session_policy())
scenario_policy = deep_merge_policy(fresh_policy(), old_policy)

with st.sidebar.form("scenario_form"):
    st.subheader("3. MTI Weights")
    primary_w = st.slider("Primary weight", 0.0, 80.0, float(old_policy.get("weights", {}).get("primary", 29.5)), 0.1)
    secondary_w = st.slider("Secondary weight", 0.0, 80.0, float(old_policy.get("weights", {}).get("secondary", 24.8)), 0.1)
    poverty_w = st.slider("Poverty weight", 0.0, 80.0, float(old_policy.get("weights", {}).get("poverty", 24.8)), 0.1)
    family_w = round(100.0 - primary_w - secondary_w - poverty_w, 6)
    scenario_policy["weights"]["primary"] = primary_w
    scenario_policy["weights"]["secondary"] = secondary_w
    scenario_policy["weights"]["poverty"] = poverty_w
    scenario_policy["weights"]["family"] = family_w
    st.metric("Family weight", f"{family_w:.1f}")

    st.subheader("4. Poverty Thresholds")
    scenario_policy["thresholds"]["primary_poverty"] = st.slider("Primary poverty threshold", 0.10, 1.00, float(old_policy.get("thresholds", {}).get("primary_poverty", 0.40)), 0.01)
    scenario_policy["thresholds"]["poverty_score"] = st.slider("Poverty score threshold", 0.10, 1.00, float(old_policy.get("thresholds", {}).get("poverty_score", 0.60)), 0.01)

    st.subheader("5. Family Scores")
    scenario_policy["family_scores"]["small"] = st.number_input("Family score: 1-3", 0.0, 50.0, float(old_policy.get("family_scores", {}).get("small", 9.3)), 0.1)
    scenario_policy["family_scores"]["medium"] = st.number_input("Family score: 4-6", 0.0, 50.0, float(old_policy.get("family_scores", {}).get("medium", 16.4)), 0.1)
    scenario_policy["family_scores"]["large"] = st.number_input("Family score: 7+", 0.0, 50.0, float(old_policy.get("family_scores", {}).get("large", 20.9)), 0.1)

    st.subheader("6. Equity Adjustment")
    eq = scenario_policy["equity_adjustment"]
    old_eq = old_policy.get("equity_adjustment") if isinstance(old_policy.get("equity_adjustment"), dict) else {}
    eq["enabled"] = st.checkbox("Enable equity adjustment", value=bool(old_eq["enabled"]))
    eq["female_alpha"] = st.slider("Female alpha", 0.0, 1.0, float(old_eq.get("female_alpha", 0.05)), 0.01)
    eq["one_parent_alpha"] = st.slider("One-parent alpha", 0.0, 1.0, float(old_eq.get("one_parent_alpha", 0.50)), 0.01)
    eq["ncpwd_alpha"] = st.slider("Student disability alpha", 0.0, 1.0, float(old_eq.get("ncpwd_alpha", 0.50)), 0.01)
    eq["orphan_alpha"] = st.slider("Orphan alpha", 0.0, 1.0, float(old_eq.get("orphan_alpha", 1.00)), 0.01)
    st.caption("Equity equation: M_new = M_old + alpha × (100 - M_old). Income adjustment is excluded for equity-adjusted students by default.")

    st.subheader("7. Income Adjustment")
    inc = scenario_policy["income_adjustment"]
    old_inc = old_policy.get("income_adjustment") if isinstance(old_policy.get("income_adjustment"), dict) else {}
    inc["enabled"] = st.checkbox("Enable income adjustment", value=bool(old_inc["enabled"]))
    inc["threshold"] = st.number_input("Income threshold T", 0, 10000000, int(old_inc.get("threshold", 1200000)), 50000)
    inc["k"] = st.slider("Income scaling factor k", 1.1, 10.0, float(old_inc.get("k", 3)), 0.1)
    inc["lambda"] = st.slider("Maximum income lambda", 0.0, 0.50, float(old_inc.get("lambda", 0.20)), 0.01)

    st.subheader("8. University Allocation")
    ua = scenario_policy["university_allocation"]
    old_ua = old_policy.get("university_allocation") if isinstance(old_policy.get("university_allocation"), dict) else {}
    ua["hh_intercept_mode"] = st.selectbox("HH intercept mode", ["fixed_amount", "programme_cost"], index=["fixed_amount", "programme_cost"].index(old_ua.get("hh_intercept_mode", "fixed_amount") if old_ua.get("hh_intercept_mode", "fixed_amount") in ["fixed_amount", "programme_cost"] else "fixed_amount"))
    ua["hh_intercept_amount"] = st.number_input("HH intercept amount", 0, 1000000, int(old_ua.get("hh_intercept_amount", 150000)), 5000)
    ua["hh_coefficient"] = st.number_input("HH coefficient on MTI x", -1000000, 1000000, int(old_ua.get("hh_coefficient", -135000)), 5000)
    ua["ss_intercept"] = st.slider("Scholarship intercept", 0.0, 1.0, float(old_ua.get("ss_intercept", 0.15)), 0.01)
    ua["ss_coefficient"] = st.slider("Scholarship coefficient on MTI x", -1.0, 1.0, float(old_ua.get("ss_coefficient", 0.40)), 0.01)
    ua["ll_intercept"] = 1 - ua["ss_intercept"]
    ua["ll_coefficient"] = -ua["ss_coefficient"]
    ua["upkeep_intercept"] = st.number_input("Upkeep intercept", 0, 200000, int(old_ua.get("upkeep_intercept", 40000)), 1000)
    ua["upkeep_coefficient"] = st.number_input("Upkeep coefficient on MTI x", -100000, 200000, int(old_ua.get("upkeep_coefficient", 20000)), 1000)


    st.subheader("8B. HH Safety / Hybrid Cap")
    hs = scenario_policy.setdefault("hh_safety", {})
    old_hs = old_policy.get("hh_safety") if isinstance(old_policy.get("hh_safety"), dict) else {}
    hs["enabled"] = st.checkbox("Enable HH safety cap / hybrid protection", value=bool(old_hs.get("enabled", False)))
    hs["cap_amount"] = st.number_input("HH safety cap amount", 0, 1000000, int(old_hs.get("cap_amount", 150000)), 5000)
    hs["warning_threshold"] = st.number_input("HH warning threshold", 0, 1000000, int(old_hs.get("warning_threshold", 200000)), 5000)
    hs["hh_share_warning"] = st.slider("HH share warning threshold", 0.0, 1.0, float(old_hs.get("hh_share_warning", 0.50)), 0.01)
    hs["hh_increase_share_warning"] = st.slider("HH increase share warning threshold", 0.0, 1.0, float(old_hs.get("hh_increase_share_warning", 0.40)), 0.01)
    st.caption("When enabled, university HH is capped after the selected HH formula: HH = min(HH_formula, HH safety cap, PC). Identity still holds because LL remains residual.")
    st.subheader("9. TVET Allocation")
    ta = scenario_policy["tvet_allocation"]
    old_ta = old_policy.get("tvet_allocation") if isinstance(old_policy.get("tvet_allocation"), dict) else {}
    ta["hh_base"] = st.slider("TVET HH base", 0.0, 1.0, float(old_ta.get("hh_base", 0.40)), 0.01)
    ta["hh_slope"] = st.slider("TVET HH need adjustment", 0.0, 1.0, float(old_ta.get("hh_slope", 0.30)), 0.01)
    ta["ss_base"] = st.slider("TVET scholarship base", 0.0, 1.0, float(old_ta.get("ss_base", 0.15)), 0.01)
    ta["ss_slope"] = st.slider("TVET scholarship need top-up", 0.0, 1.0, float(old_ta.get("ss_slope", 0.40)), 0.01)
    ta["ll_base"] = 1.0 - ta["hh_base"] - ta["ss_base"]
    ta["ll_slope"] = ta["ss_slope"] - ta["hh_slope"]
    st.caption(f"TVET loan is derived, not directly edited: LL = PC({ta['ll_base']:.2f} - {ta['ll_slope']:.2f}x). The engine computes LL as residual so HH + SS + LL = PC.")
    ta["upkeep_base"] = st.number_input("TVET upkeep base", 0, 100000, int(old_ta.get("upkeep_base", 13600)), 500)
    ta["upkeep_slope"] = st.number_input("TVET upkeep top-up", 0, 50000, int(old_ta.get("upkeep_slope", 5000)), 500)

    create_scenario = st.form_submit_button("Create / Update Scenario", disabled=not baseline_ready_sidebar)

if create_scenario:
    errors = []
    if family_w < 0:
        errors.append("MTI weights exceed 100%.")
    if ua["ss_intercept"] + ua["ss_coefficient"] < 0:
        errors.append("Scholarship share at MTI=100 cannot be negative.")
    if ua["ss_intercept"] + ua["ss_coefficient"] > 1:
        errors.append("Scholarship share at MTI=100 cannot exceed 100%.")
    tvet_ll_x0 = 1.0 - ta["hh_base"] - ta["ss_base"]
    tvet_ll_x1 = 1.0 - (ta["hh_base"] - ta["hh_slope"]) - (ta["ss_base"] + ta["ss_slope"])
    if min(tvet_ll_x0, tvet_ll_x1) < -1e-9:
        errors.append("TVET residual loan cannot be negative at MTI=0 or MTI=100.")
    if ta["ss_base"] + ta["ss_slope"] > 1:
        errors.append("TVET scholarship share at MTI=100 cannot exceed 100%.")
    if ta["hh_base"] - ta["hh_slope"] < -1e-9:
        errors.append("TVET household contribution at MTI=100 cannot be negative.")
    scenario_name = str(st.session_state["scenario_name"]).strip() or "Scenario"
    if errors:
        for e in errors:
            st.error(e)
    else:
        with st.spinner("Running baseline and scenario..."):
            try:
                baseline_run = st.session_state.get("baseline_result")
                active_population = st.session_state.get("baseline_population") or safe_get(baseline_run, "analysis_population")
                if not valid_result(baseline_run) or active_population != analysis_population:
                    raise RuntimeError("Run baseline first for the selected population before creating scenarios.")
                scenario_run = run_scenario_filtered(raw_df, scenario_policy, scenario_name, active_population)
                st.session_state["scenario_policy"] = copy.deepcopy(scenario_policy)
                st.session_state["scenarios"][scenario_name] = {"policy": copy.deepcopy(scenario_policy), "result": scenario_run}
                st.session_state["active_scenario"] = scenario_name
                st.success(f"Saved scenario: {scenario_name}")
            except Exception as exc:
                st.error("Scenario failed before outputs were created.")
                st.exception(exc)

baseline = result_or_none(st.session_state.get("baseline_result"))
scenarios = st.session_state.get("scenarios")
if not isinstance(scenarios, dict):
    scenarios = {}
    st.session_state["scenarios"] = scenarios
active_name = st.session_state.get("active_scenario")

baseline_population = (
    st.session_state.get("baseline_population")
    or safe_get(baseline, "analysis_population")
)
if baseline is None:
    st.info("Step 1: select an analysis population, then click **Run / Refresh Baseline** in the sidebar.")
    st.stop()
if baseline_population != analysis_population:
    st.warning("The selected analysis population has changed. Click **Run / Refresh Baseline** so baseline and scenarios use the selected population.")
    st.stop()

if not scenarios or active_name not in scenarios:
    show_baseline_only_dashboard(baseline, analysis_population)

scenario_pack = safe_get(scenarios, active_name)
if not isinstance(scenario_pack, dict):
    st.warning("Selected scenario is missing. Re-create it from the sidebar.")
    st.stop()

scenario_result_obj = safe_get(scenario_pack, "result")
scenario = result_or_none(scenario_result_obj)
scenario_policy_obj = safe_get(scenario_pack, "policy")
scenario_policy_current = scenario_policy_obj if isinstance(scenario_policy_obj, dict) else fresh_policy()
if scenario is None:
    st.warning("Selected scenario is incomplete. Re-create it from the sidebar.")
    st.stop()

base_student_df = result_student_df(baseline)
scen_student_df = result_student_df(scenario)
if base_student_df.empty or scen_student_df.empty:
    st.warning("Baseline or scenario student-level output is empty. Re-run baseline and scenario.")
    st.stop()

base_df = add_track_and_ownership(base_student_df)
scen_df = add_track_and_ownership(scen_student_df)

missing_base = ensure_required_simulation_columns(base_df, "Baseline")
missing_scen = ensure_required_simulation_columns(scen_df, "Scenario")
if missing_base or missing_scen:
    st.error(
        "Simulation output is incomplete for the selected population. "
        f"Baseline missing: {missing_base}; Scenario missing: {missing_scen}. "
        "Run / Refresh Baseline and recreate the scenario. If this persists, check that simulation_engine.run_scenario returns student_level with MTI_final and allocation columns."
    )
    st.stop()

st.subheader(f"Active comparison: Baseline vs {active_name} | Population: {analysis_population}")
st.dataframe(pd.DataFrame({"saved_scenario": list(scenarios.keys())}), use_container_width=True, hide_index=True)

st.header("Public / Private University and TVET Split")
st.caption("This separates results into public universities, private/other universities, and TVET. Institutions not in the public-university list are shown as Private University / Other for review.")
sector_summary = compare_student_aggregates(base_df, scen_df, ["education_track", "ownership_group"])
if not sector_summary.empty:
    st.dataframe(style_table(sector_summary), use_container_width=True, hide_index=True)
else:
    st.info("Sector summary could not be computed.")

st.header("Baseline vs Scenario Diagnostics")
try:
    policy_check_base = validate_policy(base_policy)
    policy_check_scen = validate_policy(scenario_policy_current)
    scen_clean_diag = filter_analysis_population_from_raw(raw_df, scenario_policy_current, analysis_population)
    clean_diag = cleaning_diagnostics(scen_clean_diag)
    fee_diag = fee_mapping_diagnostics(scen_clean_diag)
except Exception as exc:
    policy_check_base = {"error": str(exc)}
    policy_check_scen = {"error": str(exc)}
    clean_diag = {"error": str(exc)}
    fee_diag = {"primary_missing": 0, "secondary_missing": 0, "error": str(exc)}
    st.warning("Diagnostics could not be fully rebuilt, but outputs are shown.")

identity_col = "TuitionIdentityCheck" if "TuitionIdentityCheck" in scen_df.columns else "identity_check" if "identity_check" in scen_df.columns else None
max_error = scen_df[identity_col].abs().max() if identity_col else float("nan")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Students", f"{len(scen_df):,}")
m2.metric("Saved scenarios", f"{len(scenarios):,}")
m3.metric("Primary fee missing", f"{int(fee_diag.get('primary_missing', 0)):,}")
m4.metric("Secondary fee missing", f"{int(fee_diag.get('secondary_missing', 0)):,}")
m5.metric("Max identity error", fmt_ksh(max_error))

with st.expander("Diagnostics details"):
    st.write("Baseline policy check")
    st.json(policy_check_base)
    st.write("Scenario policy check")
    st.json(policy_check_scen)
    st.write("Cleaning diagnostics")
    st.json(clean_diag)
    st.write("Fee diagnostics")
    st.json(fee_diag)

st.header("Aggregate Effects: Baseline vs Scenario")
agg_compare = safe_compare_aggregate_outputs(baseline, scenario)
if agg_compare.empty:
    st.error("Could not build aggregate comparison from baseline/scenario outputs.")
    st.stop()
st.dataframe(style_indexed(agg_compare), use_container_width=True)

st.subheader("Headline fiscal shifts")
headline_rows = []
for label, metric in [("HH change", "HH"), ("Scholarship change", "SS"), ("Loan change", "LL"), ("Upkeep change", "Upkeep")]:
    if metric in agg_compare.index:
        headline_rows.append({"Measure": label, "Change_KSh": fmt_ksh(agg_compare.loc[metric, "change"])})
headline_df = pd.DataFrame(headline_rows)
if not headline_df.empty:
    st.dataframe(headline_df, use_container_width=True, hide_index=True)


st.header("Policy Safety and Transmission Dashboard")
fs_table, fs_summary = fiscal_savings_panel(agg_compare)
if fs_summary:
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Household shift", fmt_ksh(fs_summary.get("household_shift", 0)))
    f2.metric("Scholarship change", fmt_ksh(fs_summary.get("scholarship_change", 0)))
    f3.metric("Loan change", fmt_ksh(fs_summary.get("loan_change", 0)))
    f4.metric("Gov tuition saving", fmt_ksh(fs_summary.get("government_tuition_saving", 0)))
    st.caption("Gov tuition saving is the negative of scholarship plus loan change. Positive saving means the scenario shifts tuition burden away from government financing.")
if not fs_table.empty:
    st.dataframe(style_table(fs_table), use_container_width=True, hide_index=True)

warning_df = policy_warning_table(base_df, scen_df, agg_compare, scenario_policy_current)
if not warning_df.empty:
    st.subheader("Policy risk warnings")
    st.dataframe(warning_df, use_container_width=True, hide_index=True)
else:
    st.success("No policy warning thresholds breached for this scenario.")

st.subheader("Winners and losers by allocation component")
wl = winners_losers_summary(base_df, scen_df, ["HH", "SS", "LL", "Upkeep", "TotalLoan_with_Upkeep"], id_col="user_id")
if not wl.empty:
    st.caption("For HH, increase means household pays more. For SS, LL and Upkeep, increase means the student receives more through that component.")
    st.dataframe(style_table(wl), use_container_width=True, hide_index=True)
else:
    st.info("Winner/loser summary could not be computed.")

st.subheader("Programme cost band transmission")
pc_band = programme_cost_band_summary(base_df, scen_df, id_col="user_id")
if not pc_band.empty:
    st.caption("This traces whether the policy shock is coming from low-, middle-, or high-cost programmes.")
    st.dataframe(style_table(pc_band), use_container_width=True, hide_index=True)
else:
    st.info("Programme cost band summary could not be computed.")
st.header("University Allocation Change from Baseline")
base_inst = safe_get(baseline, "institution") if isinstance(safe_get(baseline, "institution"), pd.DataFrame) else pd.DataFrame()
scen_inst = safe_get(scenario, "institution") if isinstance(safe_get(scenario, "institution"), pd.DataFrame) else pd.DataFrame()
inst_change = compare_group(base_inst, scen_inst)
if not inst_change.empty:
    sort_col = "SS_change" if "SS_change" in inst_change.columns else "program_cost_change" if "program_cost_change" in inst_change.columns else inst_change.columns[0]
    st.dataframe(style_table(inst_change.sort_values(sort_col, ascending=False)), use_container_width=True)
else:
    st.info("Institution-level allocation change is not available.")

st.header("MTI Distribution Effects - Smooth Density Distributions")
try:
    dist_compare = compare_mti_distributions(baseline, scenario)
except Exception:
    dist_compare = component_summary(base_df, scen_df, ["MTI_final"]).set_index("component") if "MTI_final" in base_df.columns and "MTI_final" in scen_df.columns else pd.DataFrame()
if "MTI_final" in base_df.columns and "MTI_final" in scen_df.columns:
    show_distribution_pair(base_df["MTI_final"] if "MTI_final" in base_df.columns else pd.Series(dtype=float), scen_df["MTI_final"] if "MTI_final" in scen_df.columns else pd.Series(dtype=float), "Final MTI", "MTI Score", "MTI Change", money=False)
else:
    st.warning("MTI_final is missing, so MTI distribution cannot be displayed.")
st.subheader("MTI Distribution Summary")
st.dataframe(style_indexed(dist_compare), use_container_width=True)

st.subheader("MTI Class Frequencies and Cumulative Frequency")
freq_base = safe_mti_frequency(base_df).rename(columns={"students": "baseline_students", "frequency_share": "baseline_frequency_share", "cumulative_students": "baseline_cumulative_students", "cumulative_share": "baseline_cumulative_share"})
freq_scen = safe_mti_frequency(scen_df).rename(columns={"students": "scenario_students", "frequency_share": "scenario_frequency_share", "cumulative_students": "scenario_cumulative_students", "cumulative_share": "scenario_cumulative_share"})
freq_compare = freq_base.merge(freq_scen, on="MTI_class", how="outer")
freq_compare["student_change"] = freq_compare["scenario_students"].fillna(0) - freq_compare["baseline_students"].fillna(0)
freq_compare["cumulative_student_change"] = freq_compare["scenario_cumulative_students"].fillna(0) - freq_compare["baseline_cumulative_students"].fillna(0)
st.dataframe(style_table(freq_compare), use_container_width=True, hide_index=True)

st.subheader("MTI Transition Matrix: Baseline Class to Scenario Class")
transition_counts, transition_shares, movement_summary, transition_records = safe_mti_transition(base_df, scen_df, id_col="user_id")
tc1, tc2, tc3, tc4 = st.tabs(["Heatmap", "Counts", "Row shares", "Movement summary"])
with tc1:
    st.caption("This is the policy transition view: each cell shows how many students moved from a baseline MTI class to a scenario MTI class. The diagonal means no class transition.")
    st.pyplot(plot_transition_heatmap(transition_counts), use_container_width=True)
with tc2:
    st.caption("Rows are baseline MTI classes. Columns are scenario MTI classes. Diagonal cells stayed in the same class.")
    st.dataframe(transition_counts, use_container_width=True)
with tc3:
    st.caption("Each row sums to 100% where students existed in that baseline class.")
    st.dataframe(transition_shares.applymap(fmt_share), use_container_width=True)
with tc4:
    st.caption("Moved up means the student shifted to a higher MTI score; moved down means the scenario reduced the MTI score.")
    st.dataframe(style_table(movement_summary), use_container_width=True, hide_index=True)

st.header("MTI Component Score Distributions - Smooth Densities")
score_components = ["S_primary", "S_secondary", "S_poverty", "S_family", "MTI_baseline", "MTI_after_equity", "MTI_final", "IncomeAdjustmentRatio"]
available_components = [c for c in score_components if c in base_df.columns and c in scen_df.columns]
if available_components:
    selected_score = st.selectbox("Select MTI score/component", available_components)
    show_distribution_pair(base_df[selected_score], scen_df[selected_score], selected_score, "Score", f"{selected_score} Change", money=False)
    comp_summary = component_summary(base_df, scen_df, available_components)
    st.subheader("Component Summary with Mean, Median, Min and Max")
    st.dataframe(style_table(comp_summary), use_container_width=True)
else:
    comp_summary = pd.DataFrame()
    st.info("No MTI component columns found.")

st.header("Allocation Diagnostics: Student-Level Statistics")
allocation_stats = allocation_policy_stats(base_df, scen_df, ["HH", "SS", "LL", "Upkeep", "TotalLoan_with_Upkeep", "SS_gap_share", "LL_gap_share"])
st.caption("This table shows who is getting the least and most, and how the scenario changes the distribution. Min/max are student-level extremes, not averages.")
st.dataframe(style_table(allocation_stats), use_container_width=True, hide_index=True)

with st.expander("Show students at allocation extremes"):
    ext_component = st.selectbox("Allocation extreme component", [c for c in ["HH", "SS", "LL", "Upkeep", "TotalLoan_with_Upkeep"] if c in base_df.columns and c in scen_df.columns])
    extreme_sets = student_extremes(base_df, scen_df, [ext_component], id_col="user_id", top_n=15).get(ext_component, {})
    e_tabs = st.tabs(["Lowest scenario", "Highest scenario", "Largest increase", "Largest decrease"])
    for tab, key in zip(e_tabs, ["lowest_scenario", "highest_scenario", "largest_increase", "largest_decrease"]):
        with tab:
            st.dataframe(style_table(extreme_sets.get(key, pd.DataFrame())), use_container_width=True, hide_index=True)

st.header("Allocation Distributions - Smooth Densities")
alloc_options = [c for c in ["HH", "SS", "LL", "Upkeep", "TotalLoan_with_Upkeep", "SS_gap_share", "LL_gap_share"] if c in base_df.columns and c in scen_df.columns]
if alloc_options:
    allocation_component = st.selectbox("Select allocation component", alloc_options)
    is_money = allocation_component not in {"SS_gap_share", "LL_gap_share"}
    show_distribution_pair(base_df[allocation_component], scen_df[allocation_component], allocation_component, "KSh" if is_money else "Share", f"{allocation_component} Change", money=is_money)

st.header("Separate Detailed Views: Public Universities, Private Universities and TVET")
seg_tabs = st.tabs(["Public Universities", "Private / Other Universities", "TVET"])
with seg_tabs[0]:
    show_segment_detail("Public University", segment_filter(base_df, "Public University"), segment_filter(scen_df, "Public University"))
with seg_tabs[1]:
    show_segment_detail("Private University / Other", segment_filter(base_df, "Private University / Other"), segment_filter(scen_df, "Private University / Other"))
with seg_tabs[2]:
    show_segment_detail("TVET", segment_filter(base_df, "TVET"), segment_filter(scen_df, "TVET"))


st.header("Institution, Programme and County Effects")
tab1, tab2, tab3 = st.tabs(["Institution", "Programme", "County"])
with tab1:
    st.dataframe(style_table(scen_inst.sort_values("program_cost", ascending=False)) if not scen_inst.empty and "program_cost" in scen_inst.columns else scen_inst, use_container_width=True)
with tab2:
    prog = safe_get(scenario, "programme") if isinstance(safe_get(scenario, "programme"), pd.DataFrame) else pd.DataFrame()
    st.dataframe(style_table(prog.sort_values("program_cost", ascending=False)) if not prog.empty and "program_cost" in prog.columns else prog, use_container_width=True)
with tab3:
    county = safe_get(scenario, "county") if isinstance(safe_get(scenario, "county"), pd.DataFrame) else pd.DataFrame()
    st.dataframe(style_table(county.sort_values("program_cost", ascending=False)) if not county.empty and "program_cost" in county.columns else county, use_container_width=True)

st.header("Student-Level Scenario Impact")
try:
    student_changes = compare_student_level(baseline, scenario, id_col="user_id")
except Exception:
    student_changes = aligned_student_frames(base_df, scen_df, id_col="user_id")
st.dataframe(style_table(student_changes.head(1000)), use_container_width=True)

st.header("Download Outputs")
d1, d2, d3, d4 = st.columns(4)
with d1:
    st.download_button("Download scenario student-level CSV", scen_df.to_csv(index=False).encode("utf-8"), f"{active_name}_student_level.csv", "text/csv")
with d2:
    st.download_button("Download aggregate comparison CSV", agg_compare.to_csv().encode("utf-8"), f"{active_name}_aggregate_comparison.csv", "text/csv")
with d3:
    st.download_button("Download component summary CSV", comp_summary.to_csv(index=False).encode("utf-8"), f"{active_name}_component_summary.csv", "text/csv")
with d4:
    st.download_button("Download student-level changes CSV", student_changes.to_csv(index=False).encode("utf-8"), f"{active_name}_student_level_changes.csv", "text/csv")

e1, e2, e3 = st.columns(3)
with e1:
    st.download_button("Download university allocation changes CSV", inst_change.to_csv(index=False).encode("utf-8"), f"{active_name}_university_allocation_changes.csv", "text/csv")
with e2:
    st.download_button("Download MTI class frequency CSV", freq_compare.to_csv(index=False).encode("utf-8"), f"{active_name}_mti_class_frequency.csv", "text/csv")
with e3:
    st.download_button("Download MTI transition matrix CSV", transition_counts.to_csv().encode("utf-8"), f"{active_name}_mti_transition_matrix.csv", "text/csv")

e4, e5, e6 = st.columns(3)
with e4:
    st.download_button("Download public/private/TVET split CSV", sector_summary.to_csv(index=False).encode("utf-8"), f"{active_name}_public_private_tvet_split.csv", "text/csv")
with e5:
    st.download_button("Download public university students CSV", segment_filter(scen_df, "Public University").to_csv(index=False).encode("utf-8"), f"{active_name}_public_university_students.csv", "text/csv")
with e6:
    st.download_button("Download TVET students CSV", segment_filter(scen_df, "TVET").to_csv(index=False).encode("utf-8"), f"{active_name}_tvet_students.csv", "text/csv")
