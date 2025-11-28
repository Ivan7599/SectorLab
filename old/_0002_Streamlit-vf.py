import pandas as pd
import numpy as np
import polars as pl
from pathlib import Path
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Risk-Budgeting Portfolio Dashboard",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------
# Caching: load once
# ---------------------------------------------------------
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
    """Yield (segment_start_excl, segment_start_incl, segment_end_incl, seg_df)."""
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
        lookback = trading_lookback_window(ret_wide, t0, 252)
        if lookback is None:
            continue
        lookback = lookback.dropna(axis=0, how="any")
        if lookback.shape[0] < 200:
            continue
        cov = np.cov(lookback.values.T, ddof=1)
        if not np.all(np.isfinite(cov)):
            continue
        w = risk_parity_weights(cov)
        seg_ret = (seg.values @ w)
        for dt, r in zip(seg.index, seg_ret):
            nav *= (1.0 + (0.0 if np.isnan(r) else r))
            vami.append(nav)
            times.append(dt)
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
            vami.append(nav)
            times.append(dt)
    return pd.Series(vami, index=pd.DatetimeIndex(times), name="Equal Weight")


# ---------- MDP ----------
def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Project vector v onto the probability simplex {w >= 0, sum w = 1}."""
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
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
    return w / s if s > 0 else np.ones_like(w) / w.size


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
        return (sigma / b) - (a / (b ** 3)) * (cov @ w_)

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
        lookback = trading_lookback_window(ret_wide, t0, 252)
        if lookback is None:
            continue
        lookback = lookback.dropna(axis=0, how="any")
        if lookback.shape[0] < 200:
            continue
        cov = np.cov(lookback.values.T, ddof=1)
        if not np.all(np.isfinite(cov)):
            continue
        w = mdp_weights(cov)
        seg_ret = (seg.values @ w)
        for dt, r in zip(seg.index, seg_ret):
            nav *= (1.0 + (0.0 if np.isnan(r) else r))
            vami.append(nav)
            times.append(dt)
    return pd.Series(vami, index=pd.DatetimeIndex(times), name="MDP")


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


def compute_weight_history(ret_wide: pd.DataFrame, freq_code: str, method: str) -> pd.DataFrame:
    """
    Compute portfolio weights on each rebalance date for a given method.
    Returns a DataFrame indexed by rebalance dates with sector weights as columns.
    """
    idx = ret_wide.dropna(how="all").index
    if len(idx) < 260:
        return pd.DataFrame()

    rebal_dates = schedule_rebalances(idx, freq_code)
    dates = []
    rows = []

    for t0 in rebal_dates:
        win = trading_lookback_window(ret_wide, t0, 252)
        if win is None:
            continue
        win = win.dropna(axis=0, how="any")
        if win.shape[0] < 200:
            continue

        cov = np.cov(win.values.T, ddof=1)
        if not np.all(np.isfinite(cov)):
            continue

        if method == "ERC":
            w = risk_parity_weights(cov)
        elif method == "MDP":
            w = mdp_weights(cov)
        elif method == "Equal Weight":
            w = np.ones(win.shape[1]) / win.shape[1]
        else:
            continue

        dates.append(t0)
        rows.append(w)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates), columns=ret_wide.columns)



# ---------------------------------------------------------
# Main app
# ---------------------------------------------------------
def main():
    DATA_PATH = "Data.parquet"

    if not Path(DATA_PATH).exists():
        st.error("Data.parquet not found. Please run the conversion script first.")
        st.stop()

    data = load_data(DATA_PATH)

    # Asset lists
    all_assets = sorted([a for a in data["asset"].unique() if a != "S&P 500"])
    benchmark_name = "S&P 500"

    # ----------------------------
    # Sidebar controls (grouped in expanders)
    # ----------------------------
    st.sidebar.header("Controls")

    # --- Sector selection (no benchmark UI) ---
    with st.sidebar.expander("Sector selection", expanded=True):
        if "picked" not in st.session_state:
            st.session_state.picked = all_assets.copy()
        if "select_all" not in st.session_state:
            st.session_state.select_all = True

        def _on_select_all_change():
            if st.session_state.select_all:
                st.session_state.picked = all_assets.copy()

        def _on_picked_change():
            st.session_state.select_all = set(st.session_state.picked) == set(all_assets)

        st.checkbox(
            "Select all sectors",
            key="select_all",
            on_change=_on_select_all_change,
        )

        st.multiselect(
            "Sectors to include",
            options=all_assets,
            key="picked",
            on_change=_on_picked_change,
        )

    picked = st.session_state.picked
    if len(picked) < 2:
        st.warning("Select at least two sectors.")
        st.stop()

    # --- Time period & rebalancing ---
    with st.sidebar.expander("Time period & rebalancing", expanded=True):
        min_date = data["Date"].min().date()
        max_date = data["Date"].max().date()
        start_date = st.date_input("Start date", min_value=min_date, max_value=max_date, value=min_date)
        end_date = st.date_input("End date", min_value=min_date, max_value=max_date, value=max_date)

        freq_label = st.selectbox("Rebalancing frequency", ["Monthly", "Quarterly", "Yearly"], index=0)
        freq_map = {"Monthly": "M", "Quarterly": "Q", "Yearly": "A"}
        freq = freq_map[freq_label]

    # --- Portfolios shown ---
    with st.sidebar.expander("Portfolios shown", expanded=True):
        methods_to_show = st.multiselect(
            "Risk-budgeting portfolios",
            ["ERC", "MDP"],
            default=["ERC", "MDP"]
        )

    # --- Risk-free rate ---
    with st.sidebar.expander("Risk-free rate", expanded=False):
        rf = st.number_input(
            "Annual risk-free rate (%)",
            min_value=-5.0,
            max_value=20.0,
            value=0.0,
            step=0.25
        ) / 100.0
        st.caption("Used in Sharpe & Sortino ratios as excess return.")

    # ----------------------------
    # Prep wide frames
    # ----------------------------
    df_sel = data[(data["asset"].isin(picked + [benchmark_name])) &
                  (data["Date"].between(pd.to_datetime(start_date), pd.to_datetime(end_date)))].copy()

    prices = df_sel.pivot_table(index="Date", columns="asset", values="price").sort_index()
    rets = df_sel.pivot_table(index="Date", columns="asset", values="ret").sort_index()

    rets = rets.dropna(how="all")
    prices = prices.reindex(rets.index)

    bench_ret = rets[benchmark_name].dropna()
    sector_rets = rets[picked].dropna(how="all")
    sector_prices = prices[picked].loc[sector_rets.index]

    # ----------------------------
    # Build portfolios
    # ----------------------------
    sector_ret_for_port = sector_rets.dropna(axis=0, how="any")

    erc_curve = portfolio_path_erc(sector_ret_for_port, freq)
    mdp_curve = portfolio_path_mdp(sector_ret_for_port, freq)
    ew_curve = portfolio_path_equal_weight(sector_ret_for_port, freq)
    bench_curve_full = (1 + bench_ret).cumprod().rename("S&P 500")

    # Validate selected methods
    empty_selected = [
        m for m in methods_to_show
        if (m == "ERC" and len(erc_curve) == 0) or (m == "MDP" and len(mdp_curve) == 0)
    ]
    if empty_selected:
        st.warning(
            "Insufficient lookback for: "
            + ", ".join(empty_selected)
            + ". The dashboard will still show Equal Weight and the benchmark."
        )

    # Align curves only across what we display
    curves = {}
    if "ERC" in methods_to_show and len(erc_curve) > 0:
        curves["ERC"] = erc_curve
    if "MDP" in methods_to_show and len(mdp_curve) > 0:
        curves["MDP"] = mdp_curve
    curves["Equal Weight"] = ew_curve
    curves["S&P 500"] = bench_curve_full

    valid_idx = None
    for s in curves.values():
        if len(s) == 0:
            continue
        valid_idx = s.index if valid_idx is None else valid_idx.intersection(s.index)
    if valid_idx is None or len(valid_idx) < 2:
        st.warning("No overlapping dates between the selected series.")
        st.stop()

    for k in list(curves.keys()):
        curves[k] = curves[k].reindex(valid_idx).ffill()

    # Daily returns
    daily = {k: curves[k].pct_change().dropna() for k in curves}

    # ----------------------------
    # Metrics helper (uses risk-free rate)
    # ----------------------------
    def metrics_from_curve(curve: pd.Series, daily_ret: pd.Series, rf_annual: float):
        curve = curve.dropna()
        daily_ret = daily_ret.reindex(curve.index).dropna()
        if len(curve) < 2 or len(daily_ret) < 2:
            return dict(Sharpe=np.nan, Sortino=np.nan, AnnRet=np.nan, AnnVol=np.nan,
                        MaxDD=np.nan, DD_Start=pd.NaT, DD_End=pd.NaT)

        # annualized return from curve
        cagr = curve.iloc[-1] ** (252 / len(curve)) - 1

        # excess daily returns over risk-free
        rf_daily = rf_annual / 252.0
        excess = daily_ret - rf_daily

        ann_vol = excess.std(ddof=1) * np.sqrt(252)
        sharpe = (excess.mean() / (excess.std(ddof=1) + 1e-18)) * np.sqrt(252)

        downside = excess[excess < 0.0]
        dd = np.sqrt((downside.pow(2)).mean())
        sortino = (excess.mean() / (dd + 1e-18)) * np.sqrt(252)

        roll_max = curve.cummax()
        drawdown = curve / roll_max - 1.0
        max_dd = drawdown.min()
        end = drawdown.idxmin()
        start = (curve.loc[:end]).idxmax()
        return dict(Sharpe=sharpe, Sortino=sortino, AnnRet=cagr, AnnVol=ann_vol,
                    MaxDD=max_dd, DD_Start=start, DD_End=end)

    metrics_df = pd.DataFrame({
        k: metrics_from_curve(curves[k], daily[k], rf) for k in curves
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

    # Weight histories for rebalance dates (for weight evolution plots)
    weight_history = {}
    # Always have Equal Weight; ERC/MDP only if selected
    methods_for_weights = ["Equal Weight"]
    if "ERC" in methods_to_show:
        methods_for_weights.append("ERC")
    if "MDP" in methods_to_show:
        methods_for_weights.append("MDP")

    for m in methods_for_weights:
        weight_history[m] = compute_weight_history(sector_ret_for_port, freq, m)


    # Risk contributions (last window)
    t0_last, cols_last, cov_last = last_valid_rebalance_and_cov(sector_ret_for_port, freq, lookback=252)

    # ----------------------------
    # TABS
    # ----------------------------
    tab_intro, tab_main, tab_tech = st.tabs(["Introduction", "Dashboard", "Technical details"])

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

        1. **Sector selection**  
           - Select the sectors you want to invest in.  
           - You can use **"Select all sectors"** to quickly include all available ones.

        2. **Time period & rebalancing**  
           - Choose the **Start date** and **End date** over which the portfolios are evaluated.  
           - Choose the **Rebalancing frequency** (Monthly / Quarterly / Yearly).

        3. **Portfolios shown**  
           - Select which risk-budgeting portfolios (ERC / MDP) you want to display, in addition to Equal Weight and the benchmark.

        4. **Risk-free rate**  
           - Set an **annual risk-free rate**, used to compute Sharpe and Sortino ratios (excess returns).

        ### Interpreting the main outputs (Dashboard tab)

        - **Growth of 1 unit invested**: value of 1 currency unit invested in each portfolio.  
        - **Performance Metrics**: Sharpe, Sortino, annualized return/volatility, and max drawdown.  
        - **Correlations**:
          - Sectors vs portfolios/benchmark.
          - **Correlation heat map** between selected sectors during the investment period.
        - **Sector Volatility**: Annualized volatility of each sector over the investment period.
        - **Risk Contributions**:
          - For ERC, MDP, Equal Weight, and a proxy of the benchmark.
          - Shows both **weights** and **relative risk contributions**.

        Switch to the **"Dashboard"** tab to see the live results, and to **"Technical details"** for the formulas.
        """)

    # ----------------------------
    # Tab 2: Main Dashboard
    # ----------------------------
    with tab_main:
        st.title("Dashboard")

        # ----- common data for sections & caption -----
        vami_df = pd.DataFrame({k: curves[k] for k in curves}).dropna()

        st.subheader("Portfolio performance & risk overview")
        st.caption(
            f"{len(picked)} sectors · "
            f"{vami_df.index.min().date()} to {vami_df.index.max().date()} · "
            f"Rebalancing: {freq_label}"
        )

        with st.expander("What would you like to see on the dashboard?", expanded=True):

            section_labels = {
                "show_vami": "Growth of 1 unit invested",
                "show_metrics": "Performance metrics",
                "show_rolling_metrics": "Rolling volatility & Sharpe (252-day lookback)",
                "show_risk_return_scatter": "Risk–return scatter (ann. return vs vol)",
                "show_weight_evolution": "Portfolio weight evolution (rebalance dates)",
                "show_sec_vs_port_corr": "Sector correlations vs portfolios/benchmark (investment period)",
                "show_sec_corr_matrix": "Correlation matrix between selected sectors (investment period)",
                "show_sec_vol": "Sector volatilities (investment period)",
                "show_risk_contrib": "Risk contributions (last lookback window)",
            }

            # initialize state once
            if "sections_initialized" not in st.session_state:
                for key in section_labels:
                    st.session_state[key] = True
                st.session_state.select_all_sections = True
                st.session_state.sections_initialized = True

            def toggle_all():
                new_val = st.session_state.select_all_sections
                for k in section_labels:
                    st.session_state[k] = new_val

            def update_select_all():
                all_on = all(st.session_state[k] for k in section_labels)
                st.session_state.select_all_sections = all_on

            # Top-level select-all
            st.checkbox(
                "Select all sections",
                key="select_all_sections",
                on_change=toggle_all,
            )

            # Nicely grouped layout
            col_left, col_right = st.columns(2)

            perf_keys = [
                "show_vami",
                "show_metrics",
                "show_rolling_metrics",
                "show_risk_return_scatter",
                "show_weight_evolution",
            ]

            risk_keys = [
                "show_sec_vs_port_corr",
                "show_sec_corr_matrix",
                "show_sec_vol",
                "show_risk_contrib",
            ]

            with col_left:
                st.markdown("**Performance & allocations**")
                for k in perf_keys:
                    st.checkbox(section_labels[k], key=k, on_change=update_select_all)

            with col_right:
                st.markdown("**Risk & correlations**")
                for k in risk_keys:
                    st.checkbox(section_labels[k], key=k, on_change=update_select_all)

        # 1) Growth of 1 unit invested
        if st.session_state.show_vami:
            st.subheader("Growth of 1 unit invested")

            scale = st.radio(
                "Scale for growth chart:",
                ["Linear", "Log"],
                index=0,
                horizontal=True
            )

            if scale == "Linear":
                st.line_chart(vami_df)
            else:
                st.line_chart(np.log(vami_df))

            csv_curves = vami_df.to_csv().encode("utf-8")
            st.download_button(
                "Download growth data (CSV)",
                data=csv_curves,
                file_name="growth_curves.csv",
                mime="text/csv"
            )

        # 2) Performance metrics
        if st.session_state.show_metrics:
            st.subheader("Performance metrics")
            st.caption(f"Annual risk-free rate used: {rf:.2%}")
            st.dataframe(metrics_df.style.format(fmt))

            csv_metrics = metrics_df.to_csv().encode("utf-8")
            st.download_button(
                "Download metrics (CSV)",
                data=csv_metrics,
                file_name="metrics.csv",
                mime="text/csv"
            )

        # prep for correlations
        sec_daily = sector_rets.reindex(vami_df.index).dropna(how="any")
        corr_cols = {k: daily[k].reindex(sec_daily.index) for k in curves}
        corr_df = pd.DataFrame({
            col: sec_daily.corrwith(series) for col, series in corr_cols.items()
        })

        # --- Risk–return scatter (AnnRet vs AnnVol) ---
        if st.session_state.show_risk_return_scatter:
            st.subheader("Risk–return scatter (annualized return vs volatility)")

            if metrics_df[["AnnRet", "AnnVol"]].dropna().empty:
                st.info("Not enough data to compute risk–return scatter.")
            else:
                fig, ax = plt.subplots(figsize=(5, 3))  # <-- smaller plot size

                ax.scatter(metrics_df["AnnVol"], metrics_df["AnnRet"])

                for label in metrics_df.index:
                    x = metrics_df.loc[label, "AnnVol"]
                    y = metrics_df.loc[label, "AnnRet"]
                    if pd.notnull(x) and pd.notnull(y):
                        ax.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points")

                ax.set_xlabel("Annualized volatility")
                ax.set_ylabel("Annualized return (CAGR)")
                ax.axhline(0.08, linewidth=0.5)
                ax.axvline(0.16, linewidth=0.5)
                ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

                st.pyplot(fig)

        # --- Rolling volatility & Sharpe ---
        if st.session_state.show_rolling_metrics:
            st.subheader("Rolling volatility & Sharpe (252-day window)")

            rf_daily = rf / 252.0
            window = 252  # could expose as a slider if you want

            rolling_vol = {}
            rolling_sharpe = {}

            for name, ret in daily.items():
                r = ret.reindex(vami_df.index).dropna()
                if len(r) < window + 5:
                    continue

                # Rolling vol
                vol = r.rolling(window).std(ddof=1) * np.sqrt(252)
                rolling_vol[name] = vol

                # Rolling Sharpe on excess returns
                excess = r - rf_daily
                mean_ex = excess.rolling(window).mean()
                std_ex = excess.rolling(window).std(ddof=1)
                sharpe = (mean_ex / (std_ex + 1e-18)) * np.sqrt(252)
                rolling_sharpe[name] = sharpe

            if rolling_vol:
                vol_df = pd.DataFrame(rolling_vol).dropna(how="all")
                sharpe_df = pd.DataFrame(rolling_sharpe).dropna(how="all")

                st.markdown("**Rolling annualized volatility**")
                st.line_chart(vol_df)

                st.markdown("**Rolling Sharpe ratio**")
                st.line_chart(sharpe_df)
            else:
                st.info("Not enough data to compute rolling metrics for the selected period.")

        # 3) sector vs portfolios
        if st.session_state.show_sec_vs_port_corr:
            st.subheader("Sector correlations vs portfolios/benchmark (investment period)")
            st.dataframe(corr_df.round(3))

        # 4) sector correlation matrix
        if st.session_state.show_sec_corr_matrix:
            st.subheader("Correlation matrix between selected sectors (investment period)")
            sector_corr_matrix = sec_daily.corr()

            if not sector_corr_matrix.empty:
                fig, ax = plt.subplots(figsize=(
                    0.6 * len(sector_corr_matrix.columns) + 4,
                    0.6 * len(sector_corr_matrix.index) + 4
                ))
                sns.heatmap(
                    sector_corr_matrix,
                    annot=True,
                    fmt=".2f",
                    cmap="coolwarm",
                    vmin=-1, vmax=1,
                    square=True,
                    cbar=True,
                    ax=ax
                )
                ax.set_title("Sector Correlation Heat Map")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Not enough data to compute sector correlations.")

        # 5) sector vols
        if st.session_state.show_sec_vol:
            st.subheader("Sector volatilities (investment period)")
            sec_daily_vol = sector_rets.reindex(vami_df.index).dropna(how="any")

            if sec_daily_vol.empty:
                st.info("Not enough sector data to compute volatilities.")
            else:
                sec_vol = (sec_daily_vol.std(ddof=1) * np.sqrt(252)).sort_values(ascending=False)
                st.dataframe(
                    sec_vol.to_frame("Volatility (ann.)").style.format("{:.2%}"),
                    use_container_width=True
                )

        # 6) risk contributions
        if st.session_state.show_risk_contrib:
            st.subheader("Risk contributions (last lookback window)")

            if t0_last is None:
                st.info("Not enough data to compute risk contributions.")
            else:
                st.caption(
                    f"Computed at last rebalance date: **{t0_last.date()}**, "
                    f"lookback ≈ 252 trading days."
                )

                show_blocks = []

                if "ERC" in methods_to_show and cov_last is not None:
                    w_erc = risk_parity_weights(cov_last)
                    show_blocks.append(("ERC", risk_contrib_table(cov_last, cols_last, w_erc)))

                if "MDP" in methods_to_show and cov_last is not None:
                    w_mdp = mdp_weights(cov_last)
                    show_blocks.append(("MDP", risk_contrib_table(cov_last, cols_last, w_mdp)))

                if cov_last is not None:
                    w_ew = np.ones(len(cols_last)) / len(cols_last)
                    show_blocks.append(("Equal Weight", risk_contrib_table(cov_last, cols_last, w_ew)))

                    win_idx = trading_lookback_window(sector_ret_for_port, t0_last, 252).index
                    X_win = sector_ret_for_port.loc[win_idx]
                    spx_ret_full = prices["S&P 500"].pct_change()
                    y_win = spx_ret_full.loc[win_idx].dropna()
                    X_win = X_win.loc[y_win.index]

                    import statsmodels.api as sm
                    model = sm.OLS(y_win, sm.add_constant(X_win)).fit()
                    w_sp = np.clip(model.params[1:].values, 0, None)
                    w_sp = w_sp / (w_sp.sum() + 1e-18)
                    show_blocks.append(("S&P 500 (proxy)", risk_contrib_table(cov_last, cols_last, w_sp)))

                if show_blocks:
                    st.subheader("Risk contributions (relative)")
                    for i in range(0, len(show_blocks), 2):
                        cols_ui = st.columns(2)
                        for (name, rc_df), col in zip(show_blocks[i:i + 2], cols_ui):
                            with col:
                                st.markdown(f"#### {name}")
                                st.bar_chart(rc_df["Rel_RC"])

                    st.subheader("Detailed tables")
                    for name, rc_df in show_blocks:
                        st.markdown(f"**{name} — Weights & risk contributions**")
                        df_formatted = rc_df.copy()
                        df_formatted["Weight"] = df_formatted["Weight"].map("{:.2%}".format)
                        df_formatted["Abs_RC"] = df_formatted["Abs_RC"].map("{:.6f}".format)
                        df_formatted["Rel_RC"] = df_formatted["Rel_RC"].map("{:.2%}".format)
                        st.dataframe(df_formatted, use_container_width=True)
                        st.divider()

        # --- Portfolio weight evolution ---
        if st.session_state.show_weight_evolution:
            st.subheader("Portfolio weight evolution (rebalance dates)")
            st.caption(
                f"Weights recomputed using ~252 trading days of history at each {freq_label.lower()} rebalance."
            )

            # Only show ERC and MDP if available
            forced_methods = ["ERC", "MDP"]
            available_methods = [
                m for m in forced_methods
                if m in weight_history and not weight_history[m].empty
            ]

            if not available_methods:
                st.info("Not enough data to compute weight evolution for the selected period.")
            else:
                for method in available_methods:
                    st.markdown(f"### {method}")

                    wdf = weight_history[method]

                    # restrict to selected sectors only
                    wdf = wdf[[c for c in wdf.columns if c in picked]]

                    if wdf.empty:
                        st.info(f"No weight data available for {method}.")
                    else:
                        st.line_chart(wdf)

    # ----------------------------
    # Tab 3: Technical details / formulas
    # ----------------------------
    with tab_tech:
        st.title("Technical details")

        st.markdown("""
        This tab summarizes the main mathematical definitions and conventions used in the dashboard.
        Below, $r_t$ denotes **daily portfolio return** on day $t$, and there are $N$ assets.
        The annual risk-free rate $r_f$ is the value you set in the sidebar; we use
        the daily rate $r_f / 252$ when computing Sharpe and Sortino ratios.
        """)

        # ========================================================
        # PERFORMANCE METRICS
        # ========================================================
        st.header("Performance metrics")

        # ---- Growth of 1 unit invested ----
        st.subheader("Growth of 1 unit invested")
        st.markdown("""
        Let $V_t$ denote the value of a portfolio at time $t$ when **1 unit of capital** is invested
        at $t = 0$. If the daily portfolio returns are $r_1, r_2, \\dots, r_t$, and $V_0 = 1$, then
        """)
        st.latex(r"V_t = \prod_{u=1}^{t} (1 + r_u)")
        st.markdown("This is what the dashboard plots in the *Growth of 1 unit invested* chart.")

        # ---- CAGR ----
        st.subheader("Annualized return (CAGR)")
        st.markdown("""
        Let $T$ be the number of trading days and $V_T$ the final portfolio value starting from
        $V_0 = 1$. Assuming 252 trading days per year, the annualized return (CAGR) is
        """)
        st.latex(r"\text{CAGR} = V_T^{\,252/T} - 1")

        # ---- Ann vol ----
        st.subheader("Annualized volatility")
        st.markdown("""
        Let $r_t$ be daily returns and $\sigma_{\text{daily}}$ their sample standard deviation.
        The annualized volatility is
        """)
        st.latex(r"\sigma_{\text{ann}} = \sigma_{\text{daily}} \sqrt{252}")

        # ---- Sharpe ----
        st.subheader("Sharpe ratio")
        st.markdown("""
        With annual risk-free rate $r_f$, we define the **daily** risk-free rate as $r_f / 252$.
        The daily **excess** returns are
        """)
        st.latex(r"x_t = r_t - \frac{r_f}{252}")
        st.markdown("and the Sharpe ratio is")
        st.latex(
            r"\text{Sharpe}"
            r" = \frac{\mathbb{E}[x_t]}{\sqrt{\text{Var}(x_t)}} \sqrt{252}"
        )

        # ---- Sortino ----
        st.subheader("Sortino ratio")
        st.markdown("""
        The Sortino ratio only penalizes **downside volatility** of the excess returns.
        Let
        """)
        st.latex(r"D_t = \min(x_t, 0)")
        st.markdown("and define the downside deviation")
        st.latex(r"\sigma_{\text{down}} = \sqrt{\mathbb{E}[D_t^2]}")
        st.markdown("Then the Sortino ratio is")
        st.latex(
            r"\text{Sortino}"
            r" = \frac{\mathbb{E}[x_t]}{\sigma_{\text{down}}} \sqrt{252}"
        )

        # ---- Max drawdown ----
        st.subheader("Maximum drawdown")
        st.markdown("""
        For a portfolio value process $V_t$, define the running maximum
        """)
        st.latex(r"M_t = \max_{u \le t} V_u")
        st.markdown("The drawdown at time $t$ is")
        st.latex(r"\text{DD}_t = \frac{V_t}{M_t} - 1")
        st.markdown("and the **maximum drawdown** is")
        st.latex(r"\text{MaxDD} = \min_t \text{DD}_t")
        st.markdown("""
        The dashboard also reports the start and end dates of this worst drawdown.
        """)

        # ========================================================
        # PORTFOLIO CONSTRUCTIONS
        # ========================================================
        st.header("Portfolio constructions")

        st.subheader("Notation")
        st.markdown("""
        - $w \\in \\mathbb{R}^N$: portfolio weights, with $w_i \\ge 0$ and $\\sum_i w_i = 1$  
        - $\\Sigma \\in \\mathbb{R}^{N \\times N}$: covariance matrix of asset returns  
        - $\\sigma \\in \\mathbb{R}^N$: vector of asset volatilities, $\\sigma_i = \\sqrt{\\Sigma_{ii}}$
        """)

        # ---- Equal weight ----
        st.subheader("Equal-weight portfolio")
        st.markdown("""
        The equal-weight portfolio simply sets
        """)
        st.latex(r"w_i = \frac{1}{N}, \quad i = 1,\dots,N")
        st.markdown("""
        The dashboard rebalances to $1/N$ on the chosen frequency (monthly, quarterly, yearly).
        """)

        # ---- ERC ----
        st.subheader("Equal Risk Contribution (ERC)")
        st.markdown("""
        For a covariance matrix $\\Sigma$ and weights $w$, the **marginal contribution to risk**
        of asset $i$ is
        """)
        st.latex(r"\text{MRC}_i = (\Sigma w)_i")
        st.markdown("and the total portfolio variance is")
        st.latex(r"\sigma_p^2 = w^\top \Sigma w")
        st.markdown("The (absolute) **risk contribution** of asset $i$ is")
        st.latex(r"\text{RC}_i = w_i \cdot \text{MRC}_i = w_i (\Sigma w)_i")
        st.markdown("""
        ERC aims to choose weights $w$ such that all risk contributions are equal:
        """)
        st.latex(r"\text{RC}_1 = \text{RC}_2 = \cdots = \text{RC}_N")
        st.markdown("""
        under the constraints $w_i \\ge 0$ and $\\sum_i w_i = 1$.
        In the code, this is solved by an iterative fixed-point algorithm that updates
        weights until the risk contributions are approximately equal.
        """)

        # ---- MDP ----
        st.subheader("Maximum Diversification Portfolio (MDP)")
        st.markdown("""
        The MDP maximizes the **diversification ratio**
        """)
        st.latex(r"\text{DR}(w) = \frac{w^\top \sigma}{\sqrt{w^\top \Sigma w}}")
        st.markdown("""
        subject to $w_i \\ge 0$ and $\\sum_i w_i = 1$.

        Intuitively, the numerator is the weighted average of individual volatilities,
        while the denominator is the portfolio volatility; maximizing this ratio favours
        portfolios where the assets diversify each other strongly.
        """)
        st.markdown("""
        In the code:

        1. We compute $\\Sigma$ from the last 252 trading days (approx. 1 year).  
        2. We start from an unconstrained solution $w \\propto \\Sigma^{-1} \\sigma$.  
        3. We then perform **projected gradient ascent** on $\\text{DR}(w)$:  
           - take a gradient step to increase $\\text{DR}(w)$;  
           - project back onto the simplex $\\{ w \\ge 0, \\sum_i w_i = 1 \\}$.  
        4. The final weights are used until the next rebalance date.
        """)

        # ========================================================
        # REBALANCING AND LOOKBACK WINDOWS
        # ========================================================
        st.header("Rebalancing and lookback windows")

        st.markdown("""
        - The covariance matrix $\\Sigma$ for ERC and MDP is estimated on a **rolling window**
          of approximately 252 trading days (about 1 year).  
        - Rebalancing dates are the **last trading day** of each month / quarter / year,
          depending on the selected frequency.  
        - Between two rebalancing dates, portfolio weights are kept constant, and the portfolio
          value evolves as
        """)
        st.latex(r"V_{t+1} = V_t \bigl(1 + r_{p,t+1}\bigr)")
        st.latex(r"r_{p,t+1} = w^\top r_{t+1}")
        st.markdown("""
        where $r_{t+1}$ is the vector of asset returns on day $t+1$.
        """)

        # ========================================================
        # RISK CONTRIBUTIONS IN THE DASHBOARD
        # ========================================================
        st.header("Risk contributions in the dashboard")

        st.markdown("""
        On the **last valid lookback window** (≈ 252 trading days):

        - We compute $\\Sigma$ from sector returns.  
        - For each portfolio (ERC, MDP, Equal Weight, S&P 500 proxy) we compute weights $w$.  
        - The risk contributions are
        """)
        st.latex(r"\text{RC}_i = w_i (\Sigma w)_i")
        st.latex(r"\text{RelRC}_i = \frac{\text{RC}_i}{\sum_j \text{RC}_j}")
        st.markdown("""
        - The bar charts show $\\text{RelRC}_i$ (relative risk contribution in %).  
        - The tables show the weights $w_i$, absolute contributions $\\text{RC}_i$,
          and relative contributions $\\text{RelRC}_i$.
        """)


if __name__ == "__main__":
    main()
