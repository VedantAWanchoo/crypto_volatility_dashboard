import streamlit as st
import pandas as pd
import numpy as np
import datetime
from typing import Dict, Any, Tuple

# =========================================================
# GLOBAL DATE CONSTRAINTS
# =========================================================

GLOBAL_MIN_DATE = datetime.date(2017, 8, 20)
GLOBAL_MAX_DATE = datetime.date(2025, 11, 2)

# =========================================================
# STREAMLIT CONFIG
# =========================================================

st.set_page_config(
    page_title="Crypto Volatility Backtest Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Crypto Volatility Backtest Dashboard")
st.markdown("Adjust the **Time Period** and **Strategy Parameters** in the sidebar, then click **Run**.")

# =========================================================
# DATA PATHS (DAILY)
# =========================================================

RETURN_PATH = "daily_returns.csv"
VOLUME_PATH = "daily_volumes.csv"
MCAP_PATH   = "daily_mcap.csv"

# =========================================================
# DATA LOADING
# =========================================================

@st.cache_data
def load_and_process_data(ret_path, vol_path, mcap_path):

    def process(df):
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date").sort_index()

    df_return = process(pd.read_csv(ret_path))
    df_volume = process(pd.read_csv(vol_path))
    df_mcap   = process(pd.read_csv(mcap_path))

    return df_return, df_volume, df_mcap


df_return_full, df_volume_full, df_mcap_full = load_and_process_data(
    RETURN_PATH, VOLUME_PATH, MCAP_PATH
)

# =========================================================
# METRICS (IDENTICAL TO MOMENTUM STYLE)
# =========================================================

def infer_periods_per_year(index: pd.DatetimeIndex) -> float:
    diffs = index.to_series().diff().dt.days.dropna()
    if diffs.empty:
        return np.nan
    return 365.25 / diffs.median()

def cagr(series):
    series = series.dropna()
    if len(series) < 2:
        return np.nan
    start, end = series.iloc[0], series.iloc[-1]
    years = (series.index[-1] - series.index[0]).days / 365.25
    if years <= 0 or start <= 0:
        return np.nan
    return (end / start) ** (1 / years) - 1

def ann_std(series):
    series = series.dropna()
    if len(series) < 2:
        return np.nan
    ppy = infer_periods_per_year(series.index)
    return series.std() * np.sqrt(ppy)

def max_drawdown(nav):
    nav = nav.dropna()
    roll_max = nav.cummax()
    dd = (nav - roll_max) / roll_max
    return dd.min()

# =========================================================
# LOW VOL BACKTEST (NO CLEAN_N)
# =========================================================

@st.cache_data(show_spinner=False)
def run_backtest_low_vol(
    data_ret: pd.DataFrame,
    data_aux: pd.DataFrame,
    weighting: str,
    params: Dict[str, Any]
) -> pd.DataFrame:

    lookback_n  = params["lookback_n"]
    skip_n      = params["skip_n"]
    holding_n   = params["holding_n"]
    portfolio_n = params["portfolio_n"]
    kill_bottom = params.get("kill_bottom_filter", 0.0)

    total_n = lookback_n + skip_n + holding_n

    if len(data_ret) < total_n:
        return pd.DataFrame()

    results = []

    for i in range(total_n - 1, len(data_ret), holding_n):

        ret_window = data_ret.iloc[i - (total_n - 1): i + 1]

        # -----------------------------
        # Universe filtering
        # -----------------------------
        if weighting == "Equal":
            valid_cols = ret_window.columns[ret_window.notna().all()]
            full_window = ret_window[valid_cols]
            aux_window = None
        else:
            aux_window = data_aux.loc[ret_window.index]
            common_cols = ret_window.columns.intersection(aux_window.columns)

            ret_sub = ret_window[common_cols]
            aux_sub = aux_window[common_cols]

            mask = (
                ret_sub.notna().all()
                & aux_sub.notna().all()
                & (aux_sub > 0).all()
            )

            valid_cols = common_cols[mask]
            full_window = ret_sub[valid_cols]
            aux_window = aux_sub[valid_cols]

        if len(valid_cols) == 0:
            continue

        # -----------------------------
        # Kill bottom filter
        # -----------------------------
        if kill_bottom > 0 and aux_window is not None:
            signal_date = ret_window.index[lookback_n - 1]
            aux_vals = aux_window.loc[signal_date]

            aux_sorted = aux_vals.sort_values()
            cutoff = kill_bottom * aux_vals.sum()
            survivors = aux_sorted.index[aux_sorted.cumsum() > cutoff]

            if len(survivors) == 0:
                continue

            full_window = full_window[survivors]
            aux_window = aux_window[survivors]
            valid_cols = survivors

        # -----------------------------
        # LOW VOL SIGNAL
        # -----------------------------
        lookback_data = full_window.iloc[:lookback_n]
        vol = lookback_data.std(ddof=0)

        low_vol  = vol.nsmallest(portfolio_n).index
        high_vol = vol.nlargest(portfolio_n).index

        # -----------------------------
        # Holding returns
        # -----------------------------
        holding_data = full_window.iloc[lookback_n + skip_n:]

        ret_low  = (1 + holding_data[low_vol]).prod() - 1
        ret_high = (1 + holding_data[high_vol]).prod() - 1
        ret_all  = (1 + holding_data[valid_cols]).prod() - 1

        if weighting == "Equal":
            low_ret   = ret_low.mean()
            high_ret  = ret_high.mean()
            bench_ret = ret_all.mean()
        else:
            aux_vals = aux_window.loc[lookback_data.index[-1]]

            w_low  = aux_vals[low_vol]  / aux_vals[low_vol].sum()
            w_high = aux_vals[high_vol] / aux_vals[high_vol].sum()
            w_all  = aux_vals / aux_vals.sum()

            low_ret   = (w_low  * ret_low[w_low.index]).sum()
            high_ret  = (w_high * ret_high[w_high.index]).sum()
            bench_ret = (w_all  * ret_all[w_all.index]).sum()

        results.append({
            "Date": full_window.index[-1],
            "Low_Vol_Return": low_ret,
            "High_Vol_Return": high_ret,
            "Benchmark": bench_ret,
            "Portfolio_N": len(low_vol),
            "Filtered_Universe_N": len(valid_cols),
        })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).set_index("Date")
    df["Low_Vol_NAV"]   = 100 * (1 + df["Low_Vol_Return"]).cumprod()
    df["High_Vol_NAV"]  = 100 * (1 + df["High_Vol_Return"]).cumprod()
    df["Benchmark_NAV"] = 100 * (1 + df["Benchmark"]).cumprod()

    return df

# =========================================================
# SIDEBAR (MATCHES MOMENTUM STRUCTURE)
# =========================================================

with st.sidebar:

    st.header("Time Period")

    min_d = max(GLOBAL_MIN_DATE, df_return_full.index.min().date())
    max_d = min(GLOBAL_MAX_DATE, df_return_full.index.max().date())

    start_date = st.date_input("Start Date", min_d)
    end_date   = st.date_input("End Date", max_d)

    st.header("Strategy Parameters")

    lookback_n  = st.slider("Volatility Lookback (Days)", 1, 400, 365)
    skip_n      = st.slider("Skip (Days)", 0, 60, 0)
    holding_n   = st.slider("Holding Period (Days)", 1, 100, 7)
    portfolio_n = st.slider("Portfolio Size", 2, 60, 10)

    kill_mcap = st.number_input(
        "Kill Bottom Market Cap (%)", 0.0, 10.0, 2.0
    ) / 100.0

    kill_vol = st.number_input(
        "Kill Bottom Volume (%)", 0.0, 10.0, 2.0
    ) / 100.0

    run_button = st.button("Run")

# =========================================================
# RUN + OUTPUT (COPIED FROM MOMENTUM, ONLY LABELS CHANGED)
# =========================================================

if run_button:

    df_return = df_return_full.loc[start_date:end_date]
    df_volume = df_volume_full.loc[start_date:end_date]
    df_mcap   = df_mcap_full.loc[start_date:end_date]

    backtest_cases = {
        "Equally Weighted": (None, "Equal", 0.0),
        "Volume Weighted": (df_volume, "Volume", 0.0),
        "Volume Weighted (with kill filters)": (df_volume, "Volume", kill_vol),
        "Market Cap Weighted": (df_mcap, "MktCap", 0.0),
        "Market Cap Weighted (with kill filters)": (df_mcap, "MktCap", kill_mcap),
    }

    all_dfs = {}

    params = {
        "lookback_n": lookback_n,
        "skip_n": skip_n,
        "holding_n": holding_n,
        "portfolio_n": portfolio_n,
    }

    with st.spinner("Running Backtests..."):
        for name, (aux, wt, filt) in backtest_cases.items():
            params["kill_bottom_filter"] = filt
            df = run_backtest_low_vol(df_return, aux, wt, params)
            all_dfs[name] = df

    st.success("Backtests complete! Review results below.")

    keys = list(all_dfs.keys())
    CUSTOM_COLORS = ["#1f77b4", "#7f7f7f", "#d62728"]
    CUSTOM_COLORS_2 = ["#1f77b4", "#d62728"]
    num_total_backtests = len(keys)

    for index, name in enumerate(keys):

        momentum_df = all_dfs[name]

        with st.container():

            st.markdown("---")
            st.markdown(f"## ({index + 1} / {num_total_backtests}) {name}:")

            if momentum_df.empty:
                st.warning("Not enough successful rebalancing points.")
                continue

            col1, col2, col3 = st.columns([1.5, 1, 1])

            # ---------------- SUMMARY TABLE ----------------
            with col1:
                st.markdown("##### 1. Summary Statistics")

                time_span_days = (momentum_df.index[-1] - momentum_df.index[0]).days

                windows = {
                    "1Y": 365,
                    "3Y": 365 * 3,
                    "5Y": 365 * 5,
                    "7Y": 365 * 7,
                    "All Time": time_span_days
                }

                strategies = {
                    "Low Vol": ("Low_Vol_NAV", "Low_Vol_Return"),
                    "High Vol": ("High_Vol_NAV", "High_Vol_Return"),
                    "Benchmark": ("Benchmark_NAV", "Benchmark")
                }

                rows = []

                for label, days in windows.items():

                    if label != "All Time" and time_span_days < days:
                        row = {"Window": label}
                        for strat in strategies:
                            for metric in ["Return", "Risk", "Sharpe", "MDD"]:
                                row[f"{strat}_{metric}"] = np.nan
                        rows.append(row)
                        continue

                    end_date_win = momentum_df.index[-1]
                    start_date_win = (
                        momentum_df.index[0]
                        if label == "All Time"
                        else end_date_win - pd.Timedelta(days=days)
                    )

                    df_win = momentum_df.loc[momentum_df.index >= start_date_win]
                    row = {"Window": label}

                    for strat, (nav_col, ret_col) in strategies.items():
                        nav = df_win[nav_col].dropna()
                        ret = df_win[ret_col].dropna()

                        if nav.empty or len(ret) < 2:
                            r = s = sharpe = mdd = np.nan
                        else:
                            r = cagr(nav)
                            s = ann_std(ret)
                            sharpe = r / s if s != 0 else np.nan
                            mdd = max_drawdown(nav)

                        row[f"{strat}_Return"] = r
                        row[f"{strat}_Risk"] = s
                        row[f"{strat}_Sharpe"] = sharpe
                        row[f"{strat}_MDD"] = mdd

                    rows.append(row)

                summary = pd.DataFrame(rows).set_index("Window")

                summary_fmt = summary.copy()
                for col in summary_fmt.columns:
                    if "Sharpe" in col:
                        summary_fmt[col] = summary_fmt[col].apply(
                            lambda x: f"{x:.2f}" if pd.notna(x) else "-"
                        )
                    else:
                        summary_fmt[col] = summary_fmt[col].apply(
                            lambda x: f"{x*100:.1f}%" if pd.notna(x) else "-"
                        )

                mi_cols = [tuple(col.split("_")) for col in summary_fmt.columns]
                summary_fmt.columns = pd.MultiIndex.from_tuples(mi_cols)

                st.dataframe(summary_fmt)

            # ---------------- NAV CHART ----------------
            with col2:
                st.markdown("##### 2. NAV Comparison (Base: 100)")
                nav_data = momentum_df[
                    ["Low_Vol_NAV", "High_Vol_NAV", "Benchmark_NAV"]
                ]
                nav_data.columns = ["Low Vol", "High Vol", "Benchmark"]
                st.line_chart(nav_data, color=CUSTOM_COLORS)

            # ---------------- COIN COUNT ----------------
            with col3:
                st.markdown("##### 3. Coin Count Over Time")
                count_data = momentum_df[
                    ["Portfolio_N", "Filtered_Universe_N"]
                ]
                count_data.columns = ["Portfolio Size (N)", "Filtered Universe Size"]
                st.line_chart(count_data, color=CUSTOM_COLORS_2)
