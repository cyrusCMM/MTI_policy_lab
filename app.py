import copy
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from config import get_policy, safe_policy
from data_cleaning import clean_application_data
from fee_mapping import apply_fee_mapping
from simulation_engine import run_scenario


st.set_page_config(page_title="MTI Policy Lab", layout="wide")
st.title("MTI Policy Lab")
st.caption("Baseline-first policy simulation engine for MTI, HH, SS and LL.")

if "baseline_result" not in st.session_state:
    st.session_state["baseline_result"] = None
if "scenario_result" not in st.session_state:
    st.session_state["scenario_result"] = None
if "saved_scenarios" not in st.session_state:
    st.session_state["saved_scenarios"] = {}


def read_csv_robust(path_or_file):
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            if hasattr(path_or_file, "seek"):
                path_or_file.seek(0)
            return pd.read_csv(path_or_file, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    if hasattr(path_or_file, "seek"):
        path_or_file.seek(0)
    return pd.read_csv(path_or_file, encoding="latin1", low_memory=False)


def find_data_file():
    candidates = [
        "Application Data _2026_2027.csv",
        "Application_Data_2026_2027.csv",
        "Application_Data_2025_2026_cleaned.csv",
        "Application Data 2025_2026.csv",
        "Application_Data_2025_2026.csv",
    ]
    for name in candidates:
        p = Path(name)
        if p.exists():
            return p
    return None


def filter_population(df, population):
    out = df.copy()

    if population == "All":
        return out

    inst_col = None
    for c in ["InstitutionName", "InstitutonName", "Institution", "InstitutionCode"]:
        if c in out.columns:
            inst_col = c
            break

    if "is_tvet" in out.columns:
        if population == "TVET only":
            return out[out["is_tvet"].eq(1)].copy()
        if population in ["Universities only", "Public universities only", "Private universities only"]:
            out = out[out["is_tvet"].eq(0)].copy()

    if inst_col is None:
        return out

    name = out[inst_col].astype(str).str.upper()

    public_keywords = [
        "UNIVERSITY OF NAIROBI", "KENYATTA UNIVERSITY", "MOI UNIVERSITY",
        "EGERTON", "MASENO", "JOMO KENYATTA", "JKUAT", "DEDAN KIMATHI",
        "CHUKA", "KISII", "PWANI", "MERU", "MASINDE", "TECHNICAL UNIVERSITY",
        "KIRINYAGA", "KARATINA", "MAASAI MARA", "LAIKIPIA", "GARISSA",
        "KIBABII", "SOUTH EASTERN", "MULTIMEDIA", "CO-OPERATIVE UNIVERSITY",
        "OPEN UNIVERSITY", "RONGO", "THARAKA", "MACHAKOS", "MURANG"
    ]

    is_public = name.apply(lambda x: any(k in x for k in public_keywords))

    if population == "Public universities only":
        return out[is_public].copy()
    if population == "Private universities only":
        return out[~is_public].copy()

    return out


def run_engine(raw_df, policy, population):
    clean = clean_application_data(raw_df, policy)
    mapped = apply_fee_mapping(clean)
    filtered = filter_population(mapped, population)

    if len(filtered) == 0:
        st.error("Selected population has zero rows.")
        st.stop()

    result = run_scenario(
        clean_df=filtered,
        base_policy=policy,
        scenario_name="run",
        changes={}
    )

    if result is None or not isinstance(result, dict):
        st.error("Engine returned no valid result.")
        st.stop()

    if "student_level" not in result or "aggregate" not in result:
        st.error("Engine output is missing student_level or aggregate.")
        st.write(result)
        st.stop()

    return result


def money(x):
    return f"KSh {float(x):,.0f}"


def short_number(x):
    """Compact labels for large monetary values."""
    try:
        x = float(x)
    except Exception:
        return "NA"

    ax = abs(x)
    if ax >= 1_000_000_000:
        return f"{x / 1_000_000_000:.2f}B"
    if ax >= 1_000_000:
        return f"{x / 1_000_000:.2f}M"
    if ax >= 1_000:
        return f"{x / 1_000:.2f}K"
    return f"{x:.2f}"


def series_stats(df, col):
    """Return policy-grade summary stats for a numeric column.

    Mode is deliberately excluded because most MTI/allocation variables are
    continuous or rounded/capped, so the mode can be misleading.
    """
    if df is None or col not in df.columns:
        return None

    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return None

    return {
        "N": int(s.count()),
        "Mean": float(s.mean()),
        "SD": float(s.std(ddof=1)) if s.count() > 1 else 0.0,
        "Min": float(s.min()),
        "Max": float(s.max()),
    }

def stats_table(df1, col, label1, df2=None, label2=None):
    rows = []
    for df, label in [(df1, label1), (df2, label2)]:
        if df is None or label is None:
            continue
        stt = series_stats(df, col)
        if stt is not None:
            rows.append({"Run": label, **stt})
    return pd.DataFrame(rows)


def density_plot(df1, col, label1, df2=None, label2=None, title=None):
    """Density plot with mean, SD, minimum and maximum shown on the chart.

    Mode is excluded because it is unstable for rounded/capped MTI scores and
    continuous allocation values.
    """
    fig, ax = plt.subplots(figsize=(9, 4.8))
    plotted = False
    text_blocks = []
    x_values = []

    for df, label, linestyle, ypos in [
        (df1, label1, "-", 0.98),
        (df2, label2, "--", 0.66),
    ]:
        if df is None or label is None or col not in df.columns:
            continue

        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty or s.nunique() <= 1:
            continue

        s.plot(kind="density", ax=ax, label=label, linestyle=linestyle)
        plotted = True
        x_values.append(s)

        stt = series_stats(df, col)
        if stt is None:
            continue

        # Reference lines: mean plus observed min/max.
        ax.axvline(stt["Mean"], linestyle="--", linewidth=1.4, alpha=0.95)
        ax.axvline(stt["Min"], linestyle=":", linewidth=1.1, alpha=0.85)
        ax.axvline(stt["Max"], linestyle=":", linewidth=1.1, alpha=0.85)

        text_blocks.append((
            ypos,
            f"{label}\n"
            f"Mean: {short_number(stt['Mean'])}\n"
            f"SD: {short_number(stt['SD'])}\n"
            f"Min: {short_number(stt['Min'])}\n"
            f"Max: {short_number(stt['Max'])}"
        ))

    ax.set_title(title or col)
    ax.set_xlabel(col)
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)

    if plotted:
        combined = pd.concat(x_values)
        xmin, xmax = float(combined.min()), float(combined.max())
        if xmin < xmax:
            pad = (xmax - xmin) * 0.03
            ax.set_xlim(xmin - pad, xmax + pad)

        ax.legend(loc="best")
        for ypos, txt in text_blocks:
            ax.text(
                0.02, ypos, txt,
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
            )
    else:
        ax.text(0.5, 0.5, f"No variation to plot for {col}", ha="center", va="center")

    st.pyplot(fig)
    plt.close(fig)

    stt_df = stats_table(df1, col, label1, df2, label2)
    if not stt_df.empty:
        display_df = stt_df.copy()
        for c in ["Mean", "SD", "Min", "Max"]:
            display_df[c] = display_df[c].map(lambda v: f"{v:,.2f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
st.sidebar.header("1. Data")
uploaded = st.sidebar.file_uploader("Upload intake CSV", type=["csv"])

try:
    if uploaded is not None:
        raw_df = read_csv_robust(uploaded)
        data_name = uploaded.name
    else:
        p = find_data_file()
        if p is None:
            st.error("No data file found. Upload CSV or place Application Data _2026_2027.csv in this folder.")
            st.stop()
        raw_df = read_csv_robust(p)
        data_name = p.name
except Exception as e:
    st.error("Could not load data.")
    st.exception(e)
    st.stop()

st.sidebar.success(f"Loaded: {data_name}")
st.sidebar.write(f"Rows: {len(raw_df):,}")

st.sidebar.header("2. Analysis Population")
population = st.sidebar.selectbox(
    "Select population first",
    [
        "All",
        "Universities only",
        "Public universities only",
        "Private universities only",
        "TVET only",
    ],
)

base_policy = safe_policy(get_policy())

st.sidebar.header("3. MTI Component Levers")

with st.sidebar.expander("Primary score", expanded=False):
    base_policy["thresholds"]["primary_poverty"] = st.number_input(
        "Primary poverty threshold",
        1.0, 100.0,
        float(base_policy["thresholds"].get("primary_poverty", 39.8)),
        0.1,
    )

with st.sidebar.expander("Secondary score", expanded=False):
    base_policy["secondary_score"]["method"] = st.selectbox(
        "Secondary formula",
        ["linear", "exponential"],
        index=0 if base_policy["secondary_score"].get("method", "linear") == "linear" else 1,
    )
    base_policy["secondary_score"]["fee_benchmark"] = st.number_input(
        "Secondary fee benchmark",
        1000, 500000,
        int(base_policy["secondary_score"].get("fee_benchmark", 150000)),
        1000,
    )
    base_policy["secondary_score"]["decay_lambda"] = st.number_input(
        "Exponential decay lambda",
        0.000001, 0.000100,
        float(base_policy["secondary_score"].get("decay_lambda", 3e-5)),
        0.000001,
        format="%.6f",
    )

with st.sidebar.expander("Poverty / PPI score", expanded=False):
    base_policy["poverty_score"]["method"] = st.selectbox(
        "Poverty formula",
        ["linear", "logistic"],
        index=0 if base_policy["poverty_score"].get("method", "linear") == "linear" else 1,
    )
    base_policy["poverty_score"]["threshold"] = st.number_input(
        "Linear PPI threshold",
        1.0, 100.0,
        float(base_policy["poverty_score"].get("threshold", 60.0)),
        0.1,
    )
    base_policy["poverty_score"]["midpoint"] = st.slider(
        "Logistic midpoint",
        0.0, 1.0,
        float(base_policy["poverty_score"].get("midpoint", 0.40)),
        0.01,
    )
    base_policy["poverty_score"]["steepness"] = st.slider(
        "Logistic steepness",
        1.0, 30.0,
        float(base_policy["poverty_score"].get("steepness", 10.0)),
        0.5,
    )

with st.sidebar.expander("Family score", expanded=False):
    base_policy["family_scores"]["method"] = st.selectbox(
        "Family formula",
        ["log", "linear", "band"],
        index=["log", "linear", "band"].index(base_policy["family_scores"].get("method", "log")),
    )
    base_policy["family_scores"]["fmax"] = st.number_input(
        "Family Fmax",
        2, 30,
        int(base_policy["family_scores"].get("fmax", 10)),
        1,
    )

st.sidebar.header("4. Adjustment Levers")

with st.sidebar.expander("Equity adjustment", expanded=False):
    eq = base_policy["equity_adjustment"]
    eq["enabled"] = st.checkbox("Enable equity adjustment", value=bool(eq.get("enabled", True)))
    eq["orphan_alpha"] = st.slider("Orphan alpha", 0.0, 1.0, float(eq.get("orphan_alpha", 1.0)), 0.01)
    eq["one_parent_alpha"] = st.slider("One parent alpha", 0.0, 1.0, float(eq.get("one_parent_alpha", 0.5)), 0.01)
    eq["ncpwd_alpha"] = st.slider("Student disability alpha", 0.0, 1.0, float(eq.get("ncpwd_alpha", 0.5)), 0.01)
    eq["female_alpha"] = st.slider("Female alpha", 0.0, 0.2, float(eq.get("female_alpha", 0.05)), 0.01)

with st.sidebar.expander("Income adjustment", expanded=True):
    inc = base_policy["income_adjustment"]
    inc["enabled"] = st.checkbox("Enable income adjustment", value=bool(inc.get("enabled", True)))
    inc["threshold"] = st.number_input("T_IPH threshold", 0, 5000000, int(inc.get("threshold", 399996)), 1000)
    inc["k"] = st.slider("Income scaling k", 1.1, 50.0, float(inc.get("k", 15)), 0.1)
    inc["lambda"] = st.slider("Maximum income adjustment λ", 0.0, 0.5, float(inc.get("lambda", 0.20)), 0.01)
    inc["curve"] = st.selectbox("Income curve", ["smoothstep", "linear"], index=0)
    inc["exclude_equity_adjusted"] = st.checkbox(
        "Exclude equity-adjusted from income adjustment",
        value=bool(inc.get("exclude_equity_adjusted", False)),
    )
    inc["round_final_mti"] = st.checkbox("Round final MTI like Excel", value=bool(inc.get("round_final_mti", True)))

st.sidebar.header("5. Allocation Levers")

with st.sidebar.expander("HH / SS / LL allocation", expanded=True):
    ua = base_policy["university_allocation"]

    hh_mode_options = {
        "Programme cost share": "program_cost_share",
        "Fixed cap curve": "fixed_cap_curve",
    }
    current_mode = ua.get("hh_formula_mode", "program_cost_share")
    current_label = next(
        (label for label, value in hh_mode_options.items() if value == current_mode),
        "Programme cost share",
    )
    selected_hh_mode = st.selectbox(
        "HH formula mode",
        list(hh_mode_options.keys()),
        index=list(hh_mode_options.keys()).index(current_label),
    )
    ua["hh_formula_mode"] = hh_mode_options[selected_hh_mode]

    ua["hh_base_share"] = st.slider("HH base share", 0.0, 1.0, float(ua.get("hh_base_share", 0.10)), 0.01)
    ua["hh_ability_share"] = st.slider("HH ability share", 0.0, 1.0, float(ua.get("hh_ability_share", 0.30)), 0.01)
    ua["hh_cap"] = st.number_input("HH cap", 0, 1000000, int(ua.get("hh_cap", 150000)), 5000)
    ua["hh_discount"] = st.slider("HH discount for fixed cap curve", 0.0, 1.0, float(ua.get("hh_discount", 0.90)), 0.01)
    ua["ss_base_share"] = st.slider("SS base share of gap", 0.0, 1.0, float(ua.get("ss_base_share", 0.15)), 0.01)
    ua["ss_need_share"] = st.slider("SS need share", 0.0, 1.0, float(ua.get("ss_need_share", 0.30)), 0.01)

st.sidebar.header("6. Baseline and Scenarios")
run_baseline = st.sidebar.button("Run / Reset Baseline", type="primary")
scenario_name = st.sidebar.text_input("Scenario name", value="Scenario 1")
run_scenario_btn = st.sidebar.button("Create / Update Scenario")

if run_baseline:
    with st.spinner("Running baseline..."):
        st.session_state["baseline_result"] = run_engine(raw_df, copy.deepcopy(base_policy), population)
        st.session_state["scenario_result"] = None

if run_scenario_btn:
    if st.session_state["baseline_result"] is None:
        st.error("Run baseline first.")
        st.stop()
    with st.spinner("Running scenario..."):
        scen = run_engine(raw_df, copy.deepcopy(base_policy), population)
        st.session_state["scenario_result"] = scen
        st.session_state["saved_scenarios"][scenario_name] = scen

baseline = st.session_state["baseline_result"]
scenario = st.session_state["scenario_result"]

if baseline is None:
    st.info("Select population and click **Run / Reset Baseline**.")
    st.stop()

base_df = baseline["student_level"]
scen_df = scenario["student_level"] if scenario is not None else None

st.subheader(f"Baseline Summary | Population: {population}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Students", f"{len(base_df):,}")
c2.metric("Mean MTI", f"{base_df['MTI_final'].mean():.2f}")
c3.metric("Programme Cost", money(base_df["PC_allocation"].sum()))
c4.metric("Gov Support SS+LL", money(base_df["SS"].sum() + base_df["LL"].sum()))

c5, c6, c7 = st.columns(3)
c5.metric("HH", money(base_df["HH"].sum()))
c6.metric("SS", money(base_df["SS"].sum()))
c7.metric("LL", money(base_df["LL"].sum()))

if scen_df is not None:
    st.subheader("Baseline vs Scenario Aggregate Change")
    comp = pd.DataFrame({
        "baseline": {
            "mean_mti": base_df["MTI_final"].mean(),
            "HH": base_df["HH"].sum(),
            "SS": base_df["SS"].sum(),
            "LL": base_df["LL"].sum(),
            "program_cost": base_df["PC_allocation"].sum(),
        },
        "scenario": {
            "mean_mti": scen_df["MTI_final"].mean(),
            "HH": scen_df["HH"].sum(),
            "SS": scen_df["SS"].sum(),
            "LL": scen_df["LL"].sum(),
            "program_cost": scen_df["PC_allocation"].sum(),
        },
    })
    comp["change"] = comp["scenario"] - comp["baseline"]
    st.dataframe(comp, use_container_width=True)

tabs = st.tabs([
    "MTI Distribution",
    "Component Distributions",
    "Allocation Distributions",
    "Transition Matrix",
    "Institution View",
    "Student Data"
])

with tabs[0]:
    density_plot(base_df, "MTI_final", "Baseline", scen_df, "Scenario", "MTI density")

with tabs[1]:
    for col in ["S_primary", "S_secondary", "S_poverty", "S_family", "MTI_baseline", "Income_adjusted"]:
        if col in base_df.columns:
            density_plot(base_df, col, "Baseline", scen_df, "Scenario", col)

with tabs[2]:
    for col in ["HH", "SS", "LL"]:
        density_plot(base_df, col, "Baseline", scen_df, "Scenario", col)

with tabs[3]:
    if scen_df is None:
        st.info("Run scenario to view transition matrix.")
    else:
        key = "user_id" if "user_id" in base_df.columns else None
        if key is None:
            st.warning("No user_id column found.")
        else:
            trans = base_df[[key, "MTI_final"]].merge(
                scen_df[[key, "MTI_final"]],
                on=key,
                suffixes=("_baseline", "_scenario")
            )
            bins = [0, 40, 60, 80, 100]
            labels = ["0-40", "40-60", "60-80", "80-100"]
            trans["baseline_band"] = pd.cut(trans["MTI_final_baseline"], bins=bins, labels=labels, include_lowest=True)
            trans["scenario_band"] = pd.cut(trans["MTI_final_scenario"], bins=bins, labels=labels, include_lowest=True)
            st.dataframe(pd.crosstab(trans["baseline_band"], trans["scenario_band"], margins=True), use_container_width=True)
            trans["MTI_change"] = trans["MTI_final_scenario"] - trans["MTI_final_baseline"]
            density_plot(trans.rename(columns={"MTI_change": "change"}), "change", "MTI change", title="MTI change density")

with tabs[4]:
    inst_col = None
    for c in ["InstitutionName", "InstitutonName", "Institution", "InstitutionCode"]:
        if c in base_df.columns:
            inst_col = c
            break

    if inst_col is None:
        st.warning("No institution column found.")
    else:
        inst = base_df.groupby(inst_col).agg(
            students=("MTI_final", "count"),
            mean_mti=("MTI_final", "mean"),
            program_cost=("PC_allocation", "sum"),
            HH=("HH", "sum"),
            SS=("SS", "sum"),
            LL=("LL", "sum"),
        ).reset_index().sort_values("program_cost", ascending=False)
        st.dataframe(inst.head(100), use_container_width=True)
        top_n = st.slider("Top institutions", 5, 50, 20)
        st.bar_chart(inst.head(top_n).set_index(inst_col)[["HH", "SS", "LL"]])

with tabs[5]:
    show = [
        "user_id", "InstitutionName", "InstitutonName", "ProgramDescription",
        "ProgramCost", "S_primary", "S_secondary", "S_poverty", "S_family",
        "MTI_baseline", "Income_adjusted", "MTI_final", "HH", "SS", "LL",
        "TuitionIdentityCheck"
    ]
    show = [c for c in show if c in base_df.columns]
    st.dataframe(base_df[show].head(2000), use_container_width=True)
