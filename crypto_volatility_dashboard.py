import streamlit as st
import pandas as pd
import numpy as np
import datetime
from typing import Dict, Any

# =========================================================
# GLOBAL DATE CONSTRAINTS (unchanged)
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
# DATA PATHS — EXACTLY AS IN NOTEBOOK
# =========================================================

RETURN_PATH = "daily_returns.csv"
VOLUME_PATH = "daily_volumes.csv"
MCAP_PATH   = "daily_mcap.csv"

# =========================================================
# DATA LOADING (faithful to notebook)
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
# METRICS — FREQUENCY AWARE (NO HARDCODING)
# =========================================================

def infer_periods_per_year(index: pd.DatetimeIndex) -> float:
    diffs = index.to_series().diff().dt.days.dropna()
    if diffs.empty:
        return np.nan
    return 365.25 / diffs.median()

def cagr(nav: pd.Series) -> float:
    nav = nav.dropna()
    if len(nav) < 2:
        return np.nan
    years = (nav.index[-1] - nav.index[0]). / 365.25
    if years <= 0:
        return np.nan
    return (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1

def ann_std(ret: pd.Series) -> float:
    ret = ret.dropna()
    if len(ret) < 2:
        return np.nan
    ppy = infer_periods_per_year(ret.index)
    if pd.isna(ppy):
        return np.nan
    return ret.std() * np.sqrt(ppy)

def max_drawdown(nav: pd.Series) -> float:
    nav = nav.dropna()
    roll_max = nav.cummax()
    return ((nav - roll_max) / roll_max).min()

# =========================================================
# LOW-VOL BACKTEST — CLEAN_N REMOVED
# =========================================================

@st.cache_data(show_spinner=False)
def run_backtest_low_vol(
    df_return: pd.DataFrame,
    df_aux: pd.DataFrame,
    weighting: str,
    params: Dict[str, Any]
) -> pd.DataFrame:

    lookback_n  = params["lookback_n"]
    skip_n      = params["skip_n"]
    holding_n   = params["holding_n"]
    portfolio_n = params["portfolio_n"]
    kill_bottom = params.get("kill_bottom_filter", 0.0)

    total_n = lookback_n + skip_n + holding_n

    if len(df_return) < total_n:
        return pd.DataFrame()

    results = []

    for i in range(total_n - 1, len(df_return), holding_n):

        ret_window = df_return.iloc[i - (total_n - 1): i + 1]

        # -------------------------------------------------
        # Universe filtering
        # -------------------------------------------------
        if weighting == "Equal":
            valid_cols = ret_window.columns[ret_window.notna().all()]
            full_window = ret_window[valid_cols]
            aux_window = None
        else:
            aux_window = df_aux.loc[ret_window.index]
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

       # if len(valid_cols) < portfolio_n:
       #     continue

        # -------------------------------------------------
        # Kill-bottom filter
        # -------------------------------------------------
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

        # -------------------------------------------------
        # LOW VOL SIGNAL
        # -------------------------------------------------
        lookback_data = full_window.iloc[:lookback_n]
        vol = lookback_data.std(ddof=0)

        low_vol  = vol.nsmallest(portfolio_n).index
        high_vol = vol.nlargest(portfolio_n).index

        # -------------------------------------------------
        # Holding-period returns
        # -------------------------------------------------
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
# SIDEBAR — CLEAN_N REMOVED
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
# RUN BACKTESTS
# =========================================================

if run_button:

    df_return = df_return_full.loc[start_date:end_date]
    df_volume = df_volume_full.loc[start_date:end_date]
    df_mcap   = df_mcap_full.loc[start_date:end_date]

    params = {
        "lookback_n": lookback_n,
        "skip_n": skip_n,
        "holding_n": holding_n,
        "portfolio_n": portfolio_n,
    }

    cases = {
        "Equal Weight": (None, "Equal", 0.0),
        "Volume Weight": (df_volume, "Volume", 0.0),
        "Volume Weight (Filtered)": (df_volume, "Volume", kill_vol),
        "Market Cap Weight": (df_mcap, "MktCap", 0.0),
        "Market Cap Weight (Filtered)": (df_mcap, "MktCap", kill_mcap),
    }

    for name, (aux, wt, filt) in cases.items():

        params["kill_bottom_filter"] = filt
        df = run_backtest_low_vol(df_return, aux, wt, params)

        st.markdown("---")
        st.subheader(name)

        if df.empty:
            st.warning("Insufficient data for this configuration.")
            continue

        col1, col2, col3 = st.columns([1.5, 1, 1])

        with col1:
            st.markdown("##### Recent Results")
            st.dataframe(df.tail(10))

        with col2:
            nav = df[["Low_Vol_NAV", "High_Vol_NAV", "Benchmark_NAV"]]
            nav.columns = ["Low Vol", "High Vol", "Benchmark"]
            st.line_chart(nav)

        with col3:
            cnt = df[["Portfolio_N", "Filtered_Universe_N"]]
            cnt.columns = ["Portfolio", "Universe"]
            st.line_chart(cnt)



# cd "C:\\Users\\Vedant Wanchoo\\Desktop\\CGS 2020\\Crypto\\CoinDCX Application\\Trial" ; streamlit run crypto_volatility_dashboard.py
