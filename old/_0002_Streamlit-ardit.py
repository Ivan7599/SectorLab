import pandas as pd
import numpy as np
import polars as pl
from pathlib import Path
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Caching: load once
# ----------------------------
@st.cache_data
def load_data(parquet_path: str):
    df = pl.read_parquet(parquet_path)
    df = df.with_columns([
        pl.col("Date").cast(pl.Datetime),
        pl.col("asset").cast(pl.Utf8),
        pl.col("price").cast(pl.Float64),
        pl.col("ret").cast(pl.Float64)
    ])
    pdf = df.to_pandas()
    pdf["Date"] = pd.to_datetime(pdf["Date"])
    return pdf

DATA_PATH = "Data.parquet"

if not Path(DATA_PATH).exists():
    st.error("Data.parquet not found. Please run the conversion script first.")
    st.stop()

data = load_data(DATA_PATH)

# Asset lists
all_assets = sorted([a for a in data["asset"].unique() if a != "S&P 500"])
benchmark_name = "S&P 500"

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Controls")

# Which risk-budgeting portfolios to display
methods_to_show = st.sidebar.multiselect(
    "Show risk-budgeting portfolios", ["ERC", "MDP"], default=["ERC", "MDP"]
)

# --- Smart "Select all" + dynamic sector selection ---
if "picked" not in st.session_state:
    st.session_state.picked = all_assets.copy()
if "select_all" not in st.session_state:
    st.session_state.select_all = True

def _on_select_all_change():
    # If user checks "Select all", force all sectors selected
    if st.session_state.select_all:
        st.session_state.picked = all_assets.copy()

def _on_picked_change():
    # Auto-update the checkbox depending on what's selected
    st.session_state.select_all = set(st.session_state.picked) == set(all_assets)

# Checkbox
st.sidebar.checkbox(
    "Select all sectors",
    key="select_all",
    on_change=_on_select_all_change,
)

# Multiselect
st.sidebar.multiselect(
    "Sectors to include",
    options=all_assets,
    key="picked",
    on_change=_on_picked_change,
)

picked = st.session_state.picked
if len(picked) < 2:
    st.warning("Select at least two sectors.")
    st.stop()

# Date bounds
min_date = data["Date"].min().date()
max_date = data["Date"].max().date()
start_date = st.sidebar.date_input("Start date", min_value=min_date, max_value=max_date, value=min_date)
end_date = st.sidebar.date_input("End date", min_value=min_date, max_value=max_date, value=max_date)

freq_label = st.sidebar.selectbox("Rebalancing frequency", ["Monthly", "Quarterly", "Yearly"], index=0)
freq_map = {"Monthly": "M", "Quarterly": "Q", "Yearly": "A"}
freq = freq_map[freq_label]

# ----------------------------
# Prep wide frames
# ----------------------------
df_sel = data[(data["asset"].isin(picked + [benchmark_name])) &
              (data["Date"].between(pd.to_datetime(start_date), pd.to_datetime(end_date)))].copy()

prices = df_sel.pivot_table(index="Date", columns="asset", values="price").sort_index()
rets   = df_sel.pivot_table(index="Date", columns="asset", values="ret").sort_index()

rets = rets.dropna(how="all")
prices = prices.reindex(rets.index)

bench_ret = rets[benchmark_name].dropna()
sector_rets = rets[picked].dropna(how="all")
sector_prices = prices[picked].loc[sector_rets.index]

# ----------------------------
# Helpers
# ----------------------------
def trading_lookback_window(df: pd.DataFrame, end_dt: pd.Timestamp, n: int = 252):
    """Return last n trading rows strictly before end_dt."""
    idx = df.index[df.index < end_dt]
    if len(idx) < n:
        return None
    return df.loc[idx[-n:]]

def schedule_rebalances(index: pd.DatetimeIndex, freq_code: str) -> pd.DatetimeIndex:
    # last trading day of each period within the chosen window
    return index.to_series().resample(freq_code).last().dropna().index

def segment_returns(ret_wide: pd.DataFrame, dates: pd.DatetimeIndex):
    """Yield (segment_start_excl, segment_end_incl) where returns are applied."""
    dlist = list(dates)
    for i in range(len(dlist)):
        t0 = dlist[i]
        t1 = dlist[i+1] if i+1 < len(dlist) else ret_wide.index.max()
        seg = ret_wide.loc[(ret_wide.index > t0) & (ret_wide.index <= t1)]
        if len(seg):
            yield t0, seg.index[0], seg.index[-1], seg

# ---------- ERC ----------
def risk_parity_weights(cov: np.ndarray, tol: float = 1e-10, max_iter: int = 10_000) -> np.ndarray:
    """Equalize w_i * (Σ w)_i across i, with w >= 0, sum w = 1."""
    n = cov.shape[0]
    w = np.ones(n) / n
    for _ in range(max_iter):
        m = cov @ w
        rc_num = w * m
        target = np.mean(rc_num)
        w_new = w * (target / (rc_num + 1e-18))
        w_new = np.maximum(w_new, 1e-16)
        w_new = w_new / w_new.sum()
        if np.linalg.norm(w_new - w, 1) < tol:
            return w_new
        w = w_new
    return w

def portfolio_path_erc(ret_wide: pd.DataFrame, freq_code: str) -> pd.Series:
    idx = ret_wide.dropna(how="all").index
    if len(idx) < 260:
        return pd.Series(dtype=float)

    rebal_dates = schedule_rebalances(idx, freq_code)
    valid_rebals = []
    for t0 in rebal_dates:
        win = trading_lookback_window(ret_wide, t0, 252)
        if win is not None and win.dropna(how="all").shape[0] >= 240:
            valid_rebals.append(t0)
    if not valid_rebals:
        return pd.Series(dtype=float)

    vami, times, nav = [], [], 1.0
    for t0, start_incl, end_incl, seg in segment_returns(ret_wide, pd.DatetimeIndex(valid_rebals)):
        lookback = trading_lookback_window(ret_wide, t0, 252).dropna(axis=0, how="any")
        if lookback is None or lookback.shape[0] < 200:
            continue
        cov = np.cov(lookback.values.T, ddof=1)
        if not np.all(np.isfinite(cov)):
            continue
        w = risk_parity_weights(cov)
        seg_ret = (seg.values @ w)
        for dt, r in zip(seg.index, seg_ret):
            nav *= (1.0 + (0.0 if np.isnan(r) else r))
            vami.append(nav); times.append(dt)
    return pd.Series(vami, index=pd.DatetimeIndex(times), name="ERC")

# ---------- Equal Weight ----------
def portfolio_path_equal_weight(ret_wide: pd.DataFrame, freq_code: str) -> pd.Series:
    """Long-only equal weight, rebalanced on the same schedule."""
    idx = ret_wide.dropna(how="all").index
    if len(idx) == 0:
        return pd.Series(dtype=float)
    rebal_dates = schedule_rebalances(idx, freq_code)
    n = ret_wide.shape[1]
    w = np.ones(n) / n
    vami, times, nav = [], [], 1.0
    for t0, start_incl, end_incl, seg in segment_returns(ret_wide, rebal_dates):
        seg_ret = (seg.values @ w)
        for dt, r in zip(seg.index, seg_ret):
            nav *= (1.0 + (0.0 if np.isnan(r) else r))
            vami.append(nav); times.append(dt)
    return pd.Series(vami, index=pd.DatetimeIndex(times), name="Equal Weight")

# ---------- MDP ----------
def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Project vector v onto the probability simplex {w >= 0, sum w = 1}."""
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1)
    w = np.clip(v - theta, 0.0, None)
    s = w.sum()
    return w if s == 0 else w / s

def mdp_unconstrained(cov: np.ndarray) -> np.ndarray:
    """Unconstrained MDP: w ∝ Σ^{-1} σ, then normalized to sum 1."""
    sigma = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
    try:
        w = np.linalg.solve(cov, sigma)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(cov) @ sigma
    w = np.maximum(w, 0)
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w)/w.size

def mdp_weights(cov: np.ndarray, tol: float = 1e-10, max_iter: int = 5000) -> np.ndarray:
    """Long-only MDP via projected gradient ascent on DR(w) = (w·σ)/sqrt(wᵀΣw)."""
    sigma = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
    w = project_to_simplex(mdp_unconstrained(cov))

    def dr(w_):
        a = float(w_.dot(sigma))
        b = float(np.sqrt(w_.dot(cov @ w_) + 1e-18))
        return a / b

    def grad(w_):
        a = float(w_.dot(sigma))
        b = float(np.sqrt(w_.dot(cov @ w_) + 1e-18))
        return (sigma / b) - (a / (b**3)) * (cov @ w_)

    val = dr(w)
    for _ in range(max_iter):
        g = grad(w)
        step = 0.2
        improved = False
        for _bt in range(20):
            w_new = project_to_simplex(w + step * g)
            val_new = dr(w_new)
            if val_new > val + 1e-10:
                w, val = w_new, val_new
                improved = True
                break
            step *= 0.5
        if not improved or np.linalg.norm(step * g, 1) < tol:
            break
    return w

def portfolio_path_mdp(ret_wide: pd.DataFrame, freq_code: str) -> pd.Series:
    idx = ret_wide.dropna(how="all").index
    if len(idx) < 260:
        return pd.Series(dtype=float)

    rebal_dates = schedule_rebalances(idx, freq_code)
    valid_rebals = []
    for t0 in rebal_dates:
        win = trading_lookback_window(ret_wide, t0, 252)
        if win is not None and win.dropna(how="all").shape[0] >= 240:
            valid_rebals.append(t0)
    if not valid_rebals:
        return pd.Series(dtype=float)

    vami, times, nav = [], [], 1.0
    for t0, start_incl, end_incl, seg in segment_returns(ret_wide, pd.DatetimeIndex(valid_rebals)):
        lookback = trading_lookback_window(ret_wide, t0, 252).dropna(axis=0, how="any")
        if lookback is None or lookback.shape[0] < 200:
            continue
        cov = np.cov(lookback.values.T, ddof=1)
        if not np.all(np.isfinite(cov)):
            continue
        w = mdp_weights(cov)
        seg_ret = (seg.values @ w)
        for dt, r in zip(seg.index, seg_ret):
            nav *= (1.0 + (0.0 if np.isnan(r) else r))
            vami.append(nav); times.append(dt)
    return pd.Series(vami, index=pd.DatetimeIndex(times), name="MDP")

# ----------------------------
# Build portfolios (compute both)
# ----------------------------
sector_ret_for_port = sector_rets.dropna(axis=0, how="any")

erc_curve = portfolio_path_erc(sector_ret_for_port, freq)
mdp_curve = portfolio_path_mdp(sector_ret_for_port, freq)
ew_curve  = portfolio_path_equal_weight(sector_ret_for_port, freq)
bench_curve_full = (1 + bench_ret).cumprod().rename("S&P 500")

# Validate selected methods
empty_selected = [m for m in methods_to_show if (m == "ERC" and len(erc_curve) == 0) or (m == "MDP" and len(mdp_curve) == 0)]
if empty_selected:
    st.warning(f"Insufficient lookback for: {', '.join(empty_selected)}. Try a later start date or include more sectors.")
    st.stop()

# ----------------------------
# Align curves only across what we display
# ----------------------------
curves = {}
if "ERC" in methods_to_show: curves["ERC"] = erc_curve
if "MDP" in methods_to_show: curves["MDP"] = mdp_curve
curves["Equal Weight"] = ew_curve
curves["S&P 500"]      = bench_curve_full

# Intersect indexes across displayed series
valid_idx = None
for s in curves.values():
    if len(s) == 0: continue
    valid_idx = s.index if valid_idx is None else valid_idx.intersection(s.index)
if valid_idx is None or len(valid_idx) < 2:
    st.warning("No overlapping dates between the selected series.")
    st.stop()

# Reindex/ffill
for k in list(curves.keys()):
    curves[k] = curves[k].reindex(valid_idx).ffill()

# Daily returns
daily = {k: curves[k].pct_change().dropna() for k in curves}

# ----------------------------
# Metrics helper
# ----------------------------
def metrics_from_curve(curve: pd.Series, daily_ret: pd.Series):
    curve = curve.dropna()
    daily_ret = daily_ret.reindex(curve.index).dropna()
    if len(curve) < 2 or len(daily_ret) < 2:
        return dict(Sharpe=np.nan, Sortino=np.nan, AnnRet=np.nan, AnnVol=np.nan,
                    MaxDD=np.nan, DD_Start=pd.NaT, DD_End=pd.NaT)
    cagr = curve.iloc[-1] ** (252 / len(curve)) - 1
    ann_vol = daily_ret.std(ddof=1) * np.sqrt(252)
    sharpe = (daily_ret.mean() / (daily_ret.std(ddof=1) + 1e-18)) * np.sqrt(252)
    downside = daily_ret[daily_ret < 0.0]
    dd = np.sqrt((downside.pow(2)).mean())
    sortino = (daily_ret.mean() / (dd + 1e-18)) * np.sqrt(252)
    roll_max = curve.cummax()
    drawdown = curve / roll_max - 1.0
    max_dd = drawdown.min()
    end = drawdown.idxmin()
    start = (curve.loc[:end]).idxmax()
    return dict(Sharpe=sharpe, Sortino=sortino, AnnRet=cagr, AnnVol=ann_vol,
                MaxDD=max_dd, DD_Start=start, DD_End=end)

metrics_df = pd.DataFrame({
    k: metrics_from_curve(curves[k], daily[k]) for k in curves
}).T

fmt = {
    "Sharpe": "{:.2f}".format,
    "Sortino": "{:.2f}".format,
    "AnnRet": "{:.2%}".format,
    "AnnVol": "{:.2%}".format,
    "MaxDD": "{:.2%}".format,
    "DD_Start": lambda x: x.strftime("%Y-%m-%d") if pd.notnull(x) else "",
    "DD_End": lambda x: x.strftime("%Y-%m-%d") if pd.notnull(x) else "",
}

# ----------------------------
# Risk contribution helpers
# ----------------------------
def last_valid_rebalance_and_cov(ret_wide: pd.DataFrame, freq_code: str, lookback: int = 252):
    """Find last rebalance with >= lookback days of full data and return (t0, columns, covariance_matrix)."""
    idx = ret_wide.dropna(how="all").index
    if len(idx) < lookback + 10:
        return None, None, None

    rebal_dates = schedule_rebalances(idx, freq_code)
    valid = []
    for t0 in rebal_dates:
        win = trading_lookback_window(ret_wide, t0, lookback)
        if win is not None:
            win = win.dropna(axis=0, how="any")
            if win.shape[0] >= lookback * 0.95:
                valid.append((t0, win))
    if not valid:
        return None, None, None

    t0, win = valid[-1]
    cov = np.cov(win.values.T, ddof=1)
    if not np.all(np.isfinite(cov)):
        return None, None, None
    return t0, win.columns.to_list(), cov

def risk_contrib_table(cov: np.ndarray, cols: list[str], w: np.ndarray) -> pd.DataFrame:
    """Build table with weights and risk contributions."""
    m = cov @ w
    abs_rc = w * m
    tot_var = float(w @ m) + 1e-18
    rel_rc = abs_rc / tot_var
    df_rc = pd.DataFrame({"Weight": w, "Abs_RC": abs_rc, "Rel_RC": rel_rc}, index=cols)
    return df_rc

# Last valid rebalance for sectors
t0_last, cols_last, cov_last = last_valid_rebalance_and_cov(sector_ret_for_port, freq, lookback=252)

# ----------------------------
# TABS
# ----------------------------
tab_intro, tab_main = st.tabs(["Introduction", "Dashboard"])

# ----------------------------
# Tab 1: Introduction / User Guide
# ----------------------------
with tab_intro:
    st.title("Portfolio Dashboard – User Guide")

    st.markdown(f"""
    ### What this app does

    This tool compares several portfolio constructions on a set of equity sectors:

    - **ERC** – Equal Risk Contribution risk-parity portfolio  
    - **MDP** – Maximum Diversification Portfolio  
    - **Equal Weight** – 1/N across all selected sectors  
    - **Benchmark** – {benchmark_name}

    ### How to use the controls (left sidebar)

    1. **Sectors to include**  
       - Select the sectors you want to invest in.  
       - You can use **"Select all sectors"** to quickly include all available ones.

    2. **Date range**  
       - Choose the **Start date** and **End date** over which the portfolio is evaluated.  
       - The lookback windows for risk-based portfolios use up to 252 trading days.

    3. **Rebalancing frequency**  
       - Choose between **Monthly**, **Quarterly**, or **Yearly** rebalancing.  
       - This affects all dynamic portfolios (ERC, MDP, Equal Weight).

    4. **Portfolios to show**  
       - Select which risk-budgeting portfolios (ERC / MDP) you want to display, in addition to Equal Weight and the benchmark.

    ### Interpreting the main outputs (Dashboard tab)

    - **VAMI**: Growth of 1 unit invested in each portfolio (Value Added Monthly Index).  
    - **Performance Metrics**: Sharpe, Sortino, annualized return/volatility, and max drawdown.  
    - **Correlations**:
      - Sectors vs portfolios/benchmark.
      - **Correlation heat map** between selected sectors during the investment period.
    - **Sector Volatility**: Annualized volatility of each sector over the investment period.
    - **Risk Contributions**:
      - For ERC, MDP, Equal Weight, and a proxy of the benchmark.
      - Shows both **weights** and **relative risk contributions**.

    Switch to the **"Dashboard"** tab to see the live results.
    """)

# ----------------------------
# Tab 2: Main Dashboard
# ----------------------------
with tab_main:
    # 1) VAMI chart
    st.subheader("VAMI")
    vami_df = pd.DataFrame({k: curves[k] for k in curves}).dropna()
    st.line_chart(vami_df)

    # 2) Metrics table
    st.subheader("Performance Metrics (MAR = 0)")
    st.dataframe(metrics_df.style.format(fmt))

    # 3) Correlation tables
    st.subheader("Sector correlations vs portfolios/benchmark")
    sec_daily = sector_rets.reindex(vami_df.index).dropna(how="any")
    corr_cols = {k: daily[k].reindex(sec_daily.index) for k in curves}
    corr_df = pd.DataFrame({ col: sec_daily.corrwith(series) for col, series in corr_cols.items() })
    st.dataframe(corr_df.round(3))

    st.subheader("Correlation matrix between selected sectors (investment period)")
    sector_corr_matrix = sec_daily.corr()

    # Optional: still show the raw numbers
    # st.dataframe(sector_corr_matrix.round(3))

    # Heat map
    if not sector_corr_matrix.empty:
        fig, ax = plt.subplots(figsize=(
            0.6 * len(sector_corr_matrix.columns) + 4,
            0.6 * len(sector_corr_matrix.index) + 4
        ))
        sns.heatmap(
            sector_corr_matrix,
            annot=True,          # show numbers in the cells
            fmt=".2f",
            cmap="coolwarm",     # or "viridis", "RdBu_r", etc.
            vmin=-1, vmax=1,     # full correlation range
            square=True,
            cbar=True,
            ax=ax
        )
        ax.set_title("Sector Correlation Heat Map")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Not enough data to compute sector correlations on the investment period.")

    # 4 bis) Sector Volatilities (investment period)
    st.subheader("Sector Volatilities (investment period)")

    sec_daily_vol = sector_rets.reindex(vami_df.index).dropna(how="any")

    if sec_daily_vol.empty:
        st.info("Not enough sector data on the investment period to compute volatilities.")
    else:
        sec_vol = (sec_daily_vol.std(ddof=1) * np.sqrt(252)).sort_values(ascending=False)
        st.dataframe(
            sec_vol.to_frame("Volatility (ann.)").style.format("{:.2%}"),
            use_container_width=True
        )

    # 4) Risk contributions (last valid lookback window)
    st.subheader("Risk Contributions (last lookback window)")

    if t0_last is None:
        st.info("Not enough data to compute risk contributions on the last window. Try a later start date or fewer missing sectors.")
    else:
        st.caption(f"Computed at last rebalance date: **{t0_last.date()}**, lookback ≈ 252 trading days.")

        show_blocks = []
        if "ERC" in methods_to_show:
            # ERC
            w_erc = risk_parity_weights(cov_last)
            erc_rc = risk_contrib_table(cov_last, cols_last, w_erc)
            show_blocks.append(("ERC", erc_rc))
        if "MDP" in methods_to_show:
            # MDP
            w_mdp = mdp_weights(cov_last)
            mdp_rc = risk_contrib_table(cov_last, cols_last, w_mdp)
            show_blocks.append(("MDP", mdp_rc))
        # Equal Weight
        w_ew = np.ones(len(cols_last)) / len(cols_last)
        ew_rc = risk_contrib_table(cov_last, cols_last, w_ew)
        show_blocks.append(("Equal Weight", ew_rc))

        # Benchmark: S&P 500 (proxy) on same lookback window
        win_idx = trading_lookback_window(sector_ret_for_port, t0_last, 252).index

        X_win = sector_ret_for_port.loc[win_idx]
        spx_ret_full = prices["S&P 500"].pct_change()
        y_win = spx_ret_full.loc[win_idx].dropna()
        X_win = X_win.loc[y_win.index]

        import statsmodels.api as sm
        X_ols = sm.add_constant(X_win)
        model = sm.OLS(y_win, X_ols).fit()
        w_sp = np.clip(model.params[1:].values, 0, None)
        w_sp = w_sp / (w_sp.sum() + 1e-18)

        sp_rc = risk_contrib_table(cov_last, cols_last, w_sp)
        show_blocks.append(("S&P 500 (proxy)", sp_rc))

        # Charts in grid: 2 per row
        st.markdown("### Risk Contributions (relative)")
        for i in range(0, len(show_blocks), 2):
            cols_ui = st.columns(2)
            for (name, rc_df), col in zip(show_blocks[i:i + 2], cols_ui):
                with col:
                    st.markdown(f"#### {name}")
                    st.bar_chart(rc_df["Rel_RC"])

        # Detailed tables
        st.markdown("#### Detailed tables")
        for name, rc_df in show_blocks:
            st.markdown(f"**{name} — Weights & Risk Contributions**")

            rc_fmt = rc_df.copy()
            rc_fmt["Weight"] = rc_fmt["Weight"].map(lambda x: f"{x:.2%}")
            rc_fmt["Abs_RC"] = rc_fmt["Abs_RC"].map(lambda x: f"{x:.6f}")
            rc_fmt["Rel_RC"] = rc_fmt["Rel_RC"].map(lambda x: f"{x:.2%}")

            st.dataframe(rc_fmt, use_container_width=True)
            st.divider()
