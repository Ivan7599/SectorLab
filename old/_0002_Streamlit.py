"""to run:

cd "C:\HEC Lausanne\MScF\PortfolioConstruction\QARM"
& "..\.venv\Scripts\python.exe" -m streamlit run "app-m.py"
"""


# _0002_Streamlit.py — Streamlit app with ERC + MDP + EW, selectable display

# The file already contains:
# import streamlit as st
# st.title('Risk Budgeting Portfolio')
# (Keep those lines and append the rest.)

import pandas as pd
import numpy as np
import polars as pl
from pathlib import Path
import streamlit as st

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

# list-based sector picker with 'Select all'
select_all = st.sidebar.checkbox("Select all sectors", value=True)
picked = st.sidebar.multiselect("Sectors to include", all_assets, default=(all_assets if select_all else []))
if select_all and set(picked) != set(all_assets):
    picked = all_assets
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
# 1) VAMI chart
# ----------------------------
st.subheader("VAMI")
vami_df = pd.DataFrame({k: curves[k] for k in curves}).dropna()
st.line_chart(vami_df)

# ----------------------------
# 2) Metrics table
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

st.subheader("Performance Metrics (MAR = 0)")
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
    "DD_End":   lambda x: x.strftime("%Y-%m-%d") if pd.notnull(x) else "",
}
st.dataframe(metrics_df.style.format(fmt))

# ----------------------------
# 3) Correlation tables
# ----------------------------
st.subheader("Sector correlations vs portfolios/benchmark")
sec_daily = sector_rets.reindex(vami_df.index).dropna(how="any")
corr_cols = {k: daily[k].reindex(sec_daily.index) for k in curves}
corr_df = pd.DataFrame({ col: sec_daily.corrwith(series) for col, series in corr_cols.items() })
st.dataframe(corr_df.round(3))

st.subheader("Correlation matrix between selected sectors (investment period)")
sector_corr_matrix = sec_daily.corr()
st.dataframe(sector_corr_matrix.round(3))
