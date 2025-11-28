import base64
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import requests
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st




# ---------------------------------------------------------
# News API – sector queries
# ---------------------------------------------------------
SECTOR_KEYWORDS = {
    "Communication Services": "telecom OR media OR communication services",
    "Consumer Discretionary": "\"consumer discretionary\" OR retail OR autos",
    "Consumer Staples": "\"consumer staples\" OR food OR beverages OR household products",
    "Energy": "energy sector OR oil OR gas OR renewables",
    "Financials": "banks OR banking OR insurance OR \"financial sector\"",
    "Health Care": "\"health care\" OR pharma OR biotech OR hospitals",
    "Industrials": "industrials OR manufacturing OR machinery OR aerospace",
    "Information Technology": "\"information technology\" OR software OR semiconductors OR tech stocks",
    "Materials": "materials sector OR metals OR mining OR chemicals",
    "Real Estate": "\"real estate\" OR REIT OR housing market",
    "Utilities": "utilities sector OR electricity OR water companies",
}

BASE_FILTER = '("stock market" OR stocks OR equities OR ETF OR "sector ETF" OR index)'


def get_sector_news(sector: str, api_key: str, n_articles: int = 5):
    """Fetch sector-related news from NewsAPI."""
    raw_query = SECTOR_KEYWORDS.get(sector, sector)
    query = f"({raw_query}) AND {BASE_FILTER}"

    url = (
        "https://newsapi.org/v2/everything?"
        f"q={query}&"
        "language=en&"
        "sortBy=publishedAt&"
        f"pageSize={n_articles}&"
        f"apiKey={api_key}"
    )

    resp = requests.get(url)
    try:
        data = resp.json()
    except Exception:
        st.error(f"NewsAPI: invalid JSON response (status {resp.status_code})")
        return []

    if data.get("status") != "ok":
        st.error(f"NewsAPI error: {data.get('code')} - {data.get('message')}")
        return []

    articles = data.get("articles", [])
    if not articles:
        st.warning("NewsAPI returned no articles for this query.")
        return []

    return [
        {
            "title": a["title"],
            "description": a["description"],
            "url": a["url"],
            "source": a["source"]["name"],
            "date": a["publishedAt"][:10],
        }
        for a in articles
    ]


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="SectorLab",
    page_icon="logo1.png",
    layout="wide",
)


# ---------------------------------------------------------
# Header (logo + app name in blue bar)
# ---------------------------------------------------------
def load_header():
    with open("logo.png", "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode("utf-8")

    st.markdown(
        f"""
        <style>
            .header-bar {{
                background: linear-gradient(120deg, #e3e9f3, #f7f9ff);
                padding: 5px 5px;
                margin: 10px 0 15px 0;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 16px;
            }}

            /* AUGMENTE SEULEMENT LA TAILLE DU LOGO */
            .header-logo {{
                height: 250px;   
            }}
        </style>

        <div class="header-bar">
            <img class="header-logo" src="data:image/png;base64,{logo_base64}">
        </div>
        """,
        unsafe_allow_html=True,
    )



# ---------------------------------------------------------
# Top navigation bar (Alpine-style buttons)
# ---------------------------------------------------------
def nav_bar():
    pages = ["Introduction", "Dashboard", "News", "Technical details"]

    if "page" not in st.session_state:
        st.session_state.page = "Introduction"

    st.markdown("")  # spacing under header
    cols = st.columns(len(pages))

    for i, p in enumerate(pages):
        with cols[i]:
            is_active = st.session_state.page == p
            if st.button(
                p,
                key=f"nav_{p}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.page = p


# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------
@st.cache_data
def load_data(parquet_path: str):
    df = pl.read_parquet(parquet_path)
    df = df.with_columns(
        [
            pl.col("Date").cast(pl.Datetime),
            pl.col("asset").cast(pl.Utf8),
            pl.col("price").cast(pl.Float64),
            pl.col("ret").cast(pl.Float64),
        ]
    )
    pdf = df.to_pandas()
    pdf["Date"] = pd.to_datetime(pdf["Date"])
    return pdf


# ---------------------------------------------------------
# Helper functions: lookback windows, rebalancing, segments
# ---------------------------------------------------------
def trading_lookback_window(df: pd.DataFrame, end_dt: pd.Timestamp, n: int = 252):
    idx = df.index[df.index < end_dt]
    if len(idx) < n:
        return None
    return df.loc[idx[-n:]]


def schedule_rebalances(index: pd.DatetimeIndex, freq_code: str) -> pd.DatetimeIndex:
    return index.to_series().resample(freq_code).last().dropna().index


def segment_returns(ret_wide: pd.DataFrame, dates: pd.DatetimeIndex):
    dlist = list(dates)
    for i in range(len(dlist)):
        t0 = dlist[i]
        t1 = dlist[i + 1] if i + 1 < len(dlist) else ret_wide.index.max()
        seg = ret_wide.loc[(ret_wide.index > t0) & (ret_wide.index <= t1)]
        if len(seg):
            yield t0, seg.index[0], seg.index[-1], seg


# ---------------------------------------------------------
# ERC (Equal Risk Contribution)
# ---------------------------------------------------------
def risk_parity_weights(cov: np.ndarray, tol: float = 1e-10, max_iter: int = 10_000):
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
    for t0, start_incl, end_incl, seg in segment_returns(
        ret_wide, pd.DatetimeIndex(valid_rebals)
    ):
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
        seg_ret = seg.values @ w
        for dt, r in zip(seg.index, seg_ret):
            nav *= 1.0 + (0.0 if np.isnan(r) else r)
            vami.append(nav)
            times.append(dt)
    return pd.Series(vami, index=pd.DatetimeIndex(times), name="ERC")


# ---------------------------------------------------------
# Equal Weight portfolio
# ---------------------------------------------------------
def portfolio_path_equal_weight(ret_wide: pd.DataFrame, freq_code: str) -> pd.Series:
    idx = ret_wide.dropna(how="all").index
    if len(idx) == 0:
        return pd.Series(dtype=float)
    rebal_dates = schedule_rebalances(idx, freq_code)
    n = ret_wide.shape[1]
    w = np.ones(n) / n
    vami, times, nav = [], [], 1.0
    for t0, start_incl, end_incl, seg in segment_returns(ret_wide, rebal_dates):
        seg_ret = seg.values @ w
        for dt, r in zip(seg.index, seg_ret):
            nav *= 1.0 + (0.0 if np.isnan(r) else r)
            vami.append(nav)
            times.append(dt)
    return pd.Series(vami, index=pd.DatetimeIndex(times), name="Equal Weight")


# ---------------------------------------------------------
# MDP (Maximum Diversification Portfolio)
# ---------------------------------------------------------
def project_to_simplex(v: np.ndarray) -> np.ndarray:
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1)
    w = np.clip(v - theta, 0.0, None)
    s = w.sum()
    return w if s == 0 else w / s


def mdp_unconstrained(cov: np.ndarray) -> np.ndarray:
    sigma = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
    try:
        w = np.linalg.solve(cov, sigma)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(cov) @ sigma
    w = np.maximum(w, 0)
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w) / w.size


def mdp_weights(cov: np.ndarray, tol: float = 1e-10, max_iter: int = 5000) -> np.ndarray:
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
    for t0, start_incl, end_incl, seg in segment_returns(
        ret_wide, pd.DatetimeIndex(valid_rebals)
    ):
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
        seg_ret = seg.values @ w
        for dt, r in zip(seg.index, seg_ret):
            nav *= 1.0 + (0.0 if np.isnan(r) else r)
            vami.append(nav)
            times.append(dt)
    return pd.Series(vami, index=pd.DatetimeIndex(times), name="MDP")


# ---------------------------------------------------------
# Risk contributions & weight history
# ---------------------------------------------------------
def last_valid_rebalance_and_cov(
    ret_wide: pd.DataFrame, freq_code: str, lookback: int = 252
):
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
    m = cov @ w
    abs_rc = w * m
    tot_var = float(w @ m) + 1e-18
    rel_rc = abs_rc / tot_var
    return pd.DataFrame({"Weight": w, "Abs_RC": abs_rc, "Rel_RC": rel_rc}, index=cols)


def compute_weight_history(
    ret_wide: pd.DataFrame, freq_code: str, method: str
) -> pd.DataFrame:
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


# =========================================================
# MAIN APP
# =========================================================
def main():
    load_header()

    DATA_PATH = "Data.parquet"
    if not Path(DATA_PATH).exists():
        st.error("Data.parquet not found. Please run the conversion script first.")
        st.stop()

    data = load_data(DATA_PATH)
    all_assets = sorted([a for a in data["asset"].unique() if a != "S&P 500"])
    benchmark_name = "S&P 500"

    # Top navigation buttons
    nav_bar()
    page = st.session_state.page

    # =====================================================
    # PAGE 1: INTRODUCTION
    # =====================================================
    if page == "Introduction":
        st.title("SectorLab – User Guide")

        st.markdown(
            f"""
        ## 1. What is SectorLab?

        **SectorLab** is a teaching tool that helps you understand:

        - How **11 equity sectors** (via sector ETFs) behave over time  
        - How different **portfolio construction methods** allocate risk and capital  
        - How those portfolios compare to a **benchmark** ({benchmark_name})

        The app is designed for beginners in portfolio management and risk budgeting.

        

        ## 2. Portfolios you can compare

        SectorLab builds and tracks the following portfolios:

        - **ERC – Equal Risk Contribution**  
        Each sector contributes (approximately) the same amount of **risk** to the portfolio.

        - **MDP – Maximum Diversification Portfolio**  
        Maximises the **diversification ratio** so that the portfolio benefits as much as possible from imperfect correlations between sectors.

        - **Equal Weight (1/N)**  
        Same capital weight in each selected sector. Simple reference portfolio.

        - **Benchmark – {benchmark_name}**  
        Used as a market reference to see whether the sector-based portfolios add value.

        

        ## 3. How to use the controls (left sidebar)

        1. **Sector selection**  
        - Choose which sectors to include in the portfolio.  
        - Use **“Select all sectors”** to start with the full universe.

        2. **Time period & rebalancing**  
        - Pick the **Start date** and **End date** for your analysis.  
        - Choose the **Rebalancing frequency** (Monthly / Quarterly / Yearly).  
        - Remember: the model needs about **one year of data** before the first rebalance.

        3. **Portfolios shown**  
        - Decide which risk-budgeting portfolios (**ERC** and/or **MDP**) you want to display, in addition to **Equal Weight** and the **benchmark**.

        4. **Risk-free rate**  
        - Set an **annual risk-free rate**.  
        - This rate is used to compute **Sharpe** and **Sortino** ratios (excess returns over the risk-free rate).

        

        ## 4. Reading the Dashboard

        On the **Dashboard** tab, you can turn sections on/off with checkboxes at the top.

        - **Growth of 1 unit invested**  
        Shows how 1 unit invested in each portfolio (ERC, MDP, Equal Weight, benchmark) evolves over time.

        - **Performance metrics**  
        For each portfolio:  
        - Annualised return and volatility  
        - Sharpe and Sortino ratios  
        - Maximum drawdown and its period

        - **Risk–return scatter**  
        Plots **annualised return vs annualised volatility** so you can quickly see which portfolios offer better risk–reward trade-offs.

        - **Rolling metrics (252-day window)**  
        - Rolling annualised volatility  
        - Rolling Sharpe ratio (using excess returns)  
        This shows how risk and risk-adjusted performance change through time.

        - **Portfolio weight evolution**  
        Shows how ERC and MDP rebalance across sectors at each rebalancing date.

        - **Correlations & sector risk**  
        - Correlation between each sector and each portfolio / benchmark  
        - Correlation matrix between sectors  
        - Annualised volatility by sector  
        - Risk contributions of each sector to each portfolio

        

        ## 5. Other tabs

        - **Dashboard** – main interactive visualisation of portfolios and risk.  
        - **News** – recent market news related to the selected sectors (via NewsAPI).  
        - **Technical details** – mathematical formulas and methodology used in the app
        (ERC, MDP, performance and risk metrics).

        Use this Introduction as a map:  
        start with **Dashboard** to explore the results, then use **Technical details** to
        connect the output to the theory from your quantitative asset and risk management course.
        """
        )


    # =====================================================
    # PAGE 2: NEWS
    # =====================================================
    elif page == "News":
        st.title("Sector News")
        st.write("Select a sector to view recent news articles:")

        sector_choice = st.selectbox("Sector", all_assets)
        api_key = st.secrets["NEWSAPI_KEY"]
        articles = get_sector_news(sector_choice, api_key)

        if not articles:
            st.warning("No news available or API limit reached.")
        else:
            for a in articles:
                with st.container():
                    st.subheader(a["title"])
                    if a["description"]:
                        st.write(a["description"])
                    st.write(f"📅 {a['date']} — 📰 {a['source']}")
                    st.markdown(f"[Read article]({a['url']})")
                    st.markdown("---")

    # =====================================================
    # PAGE 3: DASHBOARD (with sidebar)
    # =====================================================
    elif page == "Dashboard":
        # ---------- SIDEBAR (only here) ----------
        st.sidebar.header("Controls")

        # Sector selection
        with st.sidebar.expander("Sector selection", expanded=True):
            if "picked" not in st.session_state:
                st.session_state.picked = all_assets.copy()
            if "select_all" not in st.session_state:
                st.session_state.select_all = True

            def _on_select_all_change():
                if st.session_state.select_all:
                    st.session_state.picked = all_assets.copy()

            def _on_picked_change():
                st.session_state.select_all = set(st.session_state.picked) == set(
                    all_assets
                )

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

        # Time period & rebalancing
        with st.sidebar.expander("Time period & rebalancing", expanded=True):
            min_date = data["Date"].min().date()
            max_date = data["Date"].max().date()
            start_date = st.date_input(
                "Start date", min_value=min_date, max_value=max_date, value=min_date
            )
            end_date = st.date_input(
                "End date", min_value=min_date, max_value=max_date, value=max_date
            )

            freq_label = st.selectbox(
                "Rebalancing frequency", ["Monthly", "Quarterly", "Yearly"], index=0
            )
            freq_map = {"Monthly": "M", "Quarterly": "Q", "Yearly": "A"}
            freq = freq_map[freq_label]

        # Portfolios shown
        with st.sidebar.expander("Portfolios shown", expanded=True):
            methods_to_show = st.multiselect(
                "Risk-budgeting portfolios", ["ERC", "MDP"], default=["ERC", "MDP"]
            )

        # Risk-free rate
        with st.sidebar.expander("Risk-free rate", expanded=False):
            rf = (
                st.number_input(
                    "Annual risk-free rate (%)",
                    min_value=-5.0,
                    max_value=20.0,
                    value=0.0,
                    step=0.25,
                )
                / 100.0
            )
            st.caption("Used in Sharpe & Sortino ratios as excess return.")

        # ---------- Data selection ----------
        df_sel = data[
            (data["asset"].isin(picked + [benchmark_name]))
            & (
                data["Date"].between(
                    pd.to_datetime(start_date), pd.to_datetime(end_date)
                )
            )
        ].copy()

        prices = (
            df_sel.pivot_table(index="Date", columns="asset", values="price")
            .sort_index()
        )
        rets = df_sel.pivot_table(index="Date", columns="asset", values="ret").sort_index()

        rets = rets.dropna(how="all")
        prices = prices.reindex(rets.index)

        bench_ret = rets[benchmark_name].dropna()
        sector_rets = rets[picked].dropna(how="all")
        sector_prices = prices[picked].loc[sector_rets.index]

        sector_ret_for_port = sector_rets.dropna(axis=0, how="any")

        erc_curve = portfolio_path_erc(sector_ret_for_port, freq)
        mdp_curve = portfolio_path_mdp(sector_ret_for_port, freq)
        ew_curve = portfolio_path_equal_weight(sector_ret_for_port, freq)
        bench_curve_full = (1 + bench_ret).cumprod().rename("S&P 500")

        empty_selected = [
            m
            for m in methods_to_show
            if (m == "ERC" and len(erc_curve) == 0)
            or (m == "MDP" and len(mdp_curve) == 0)
        ]
        if empty_selected:
            st.warning(
                "Insufficient lookback for: "
                + ", ".join(empty_selected)
                + ". The dashboard will still show Equal Weight and the benchmark."
            )

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

        daily = {k: curves[k].pct_change().dropna() for k in curves}

        # ---------- Performance metrics ----------
        def metrics_from_curve(
            curve: pd.Series, daily_ret: pd.Series, rf_annual: float
        ):
            curve = curve.dropna()
            daily_ret = daily_ret.reindex(curve.index).dropna()
            if len(curve) < 2 or len(daily_ret) < 2:
                return dict(
                    Sharpe=np.nan,
                    Sortino=np.nan,
                    AnnRet=np.nan,
                    AnnVol=np.nan,
                    MaxDD=np.nan,
                    DD_Start=pd.NaT,
                    DD_End=pd.NaT,
                )

            cagr = curve.iloc[-1] ** (252 / len(curve)) - 1

            rf_daily = rf_annual / 252.0
            excess = daily_ret - rf_daily

            ann_vol = daily_ret.std(ddof=1) * np.sqrt(252)
            sharpe = (
                excess.mean() / (excess.std(ddof=1) + 1e-18)
            ) * np.sqrt(252)

            downside = excess[excess < 0.0]
            dd = np.sqrt((downside.pow(2)).mean())
            sortino = (
                excess.mean() / (dd + 1e-18)
            ) * np.sqrt(252)

            roll_max = curve.cummax()
            drawdown = curve / roll_max - 1.0
            max_dd = drawdown.min()
            end = drawdown.idxmin()
            start = (curve.loc[:end]).idxmax()
            return dict(
                Sharpe=sharpe,
                Sortino=sortino,
                AnnRet=cagr,
                AnnVol=ann_vol,
                MaxDD=max_dd,
                DD_Start=start,
                DD_End=end,
            )

        metrics_df = pd.DataFrame(
            {k: metrics_from_curve(curves[k], daily[k], rf) for k in curves}
        ).T

        fmt = {
            "Sharpe": "{:.2f}".format,
            "Sortino": "{:.2f}".format,
            "AnnRet": "{:.2%}".format,
            "AnnVol": "{:.2%}".format,
            "MaxDD": "{:.2%}".format,
            "DD_Start": lambda x: x.strftime("%Y-%m-%d") if pd.notnull(x) else "",
            "DD_End": lambda x: x.strftime("%Y-%m-%d") if pd.notnull(x) else "",
        }

        weight_history = {}
        methods_for_weights = ["Equal Weight"]
        if "ERC" in methods_to_show:
            methods_for_weights.append("ERC")
        if "MDP" in methods_to_show:
            methods_for_weights.append("MDP")

        for m in methods_for_weights:
            weight_history[m] = compute_weight_history(sector_ret_for_port, freq, m)

        t0_last, cols_last, cov_last = last_valid_rebalance_and_cov(
            sector_ret_for_port, freq, lookback=252
        )

        # ---------- Dashboard content ----------
        st.title("Dashboard")

        vami_df = pd.DataFrame({k: curves[k] for k in curves}).dropna()

        st.subheader("Portfolio performance & risk overview")
        st.caption(
            f"{len(picked)} sectors · "
            f"{vami_df.index.min().date()} to {vami_df.index.max().date()} · "
            f"Rebalancing: {freq_label}"
        )

        # Section selector
        with st.expander(
            "What would you like to see on the dashboard?", expanded=True
        ):
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

            st.checkbox(
                "Select all sections",
                key="select_all_sections",
                on_change=toggle_all,
            )

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
                    st.checkbox(
                        section_labels[k], key=k, on_change=update_select_all
                    )

            with col_right:
                st.markdown("**Risk & correlations**")
                for k in risk_keys:
                    st.checkbox(
                        section_labels[k], key=k, on_change=update_select_all
                    )

        # 1) Growth of 1 unit invested
        if st.session_state.show_vami:
            st.subheader("Growth of 1 unit invested")
            scale = st.radio(
                "Scale for growth chart:",
                ["Linear", "Log"],
                index=0,
                horizontal=True,
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
                mime="text/csv",
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
                mime="text/csv",
            )

        # Sector daily returns (for correlations)
        sec_daily = sector_rets.reindex(vami_df.index).dropna(how="any")
        corr_cols = {k: daily[k].reindex(sec_daily.index) for k in curves}
        corr_df = pd.DataFrame(
            {col: sec_daily.corrwith(series) for col, series in corr_cols.items()}
        )

        
        # 3) Risk–return scatter
        if st.session_state.show_risk_return_scatter:
            st.subheader("Risk–return scatter (annualized return vs volatility)")

            rr_df = metrics_df[["AnnVol", "AnnRet", "MaxDD"]].dropna()

            if rr_df.empty:
                st.info("Not enough data to build the risk–return scatter.")
            else:

                import matplotlib.pyplot as plt

                # --- Professional dark theme ---
                plt.style.use("dark_background")

            

                fig, ax = plt.subplots(figsize=(6, 4))

                # --- Color scheme ---
                color_map = {
                    "S&P 500": "#1f77b4",
                    "Equal Weight": "#7f7f7f",
                    "ERC": "#2ca02c",
                    "MDP": "#ff7f0e",
                }

                # Marker size based on drawdown
                dd_abs = rr_df["MaxDD"].abs()
                scale = dd_abs.max() if dd_abs.max() > 0 else 1.0
                sizes = 40 + 260 * (dd_abs / scale)

                # Plot each portfolio
                for name, row in rr_df.iterrows():
                    ax.scatter(
                        row["AnnVol"],
                        row["AnnRet"],
                        s=sizes.loc[name],
                        color=color_map.get(name, "#aaaaaa"),
                        edgecolors="white",
                        linewidths=0.8,
                        label=name,
                    )

                # Sharpe ratio reference line
                sharpe_series = rr_df["AnnRet"] / rr_df["AnnVol"]
                sharpe_series = sharpe_series.replace([np.inf, -np.inf], np.nan).dropna()
                if not sharpe_series.empty:
                    slope = sharpe_series.mean()
                    x_min = 0.10
                    x_max = rr_df["AnnVol"].max() * 1.15
                    x_line = np.linspace(x_min, x_max, 100)
                    ax.plot(
                        x_line,
                        slope * x_line,
                        "--",
                        color="grey",
                        linewidth=1,
                        label="Sharpe reference",
                    )

                # --- Axis settings ---
                ax.set_xlim(0.10, rr_df["AnnVol"].max() * 1.15)
                ax.set_ylim(rr_df["AnnRet"].min() * 0.9, rr_df["AnnRet"].max() * 1.15)

                ax.set_xlabel("Annualized volatility", fontsize=12)
                ax.set_ylabel("Annualized return (CAGR)", fontsize=12)

                # --- Grid & aesthetics ---
                ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.25)

                # Nice clean legend
                h, lbl = ax.get_legend_handles_labels()
                ax.legend(h, lbl, frameon=True, facecolor="#222222", edgecolor="#444444")

                st.pyplot(fig)


        # 4) Rolling volatility & Sharpe
        if st.session_state.show_rolling_metrics:
            st.subheader("Rolling volatility & Sharpe (252-day window)")
            rf_daily = rf / 252.0
            window = 252
            rolling_vol = {}
            rolling_sharpe = {}

            for name, ret in daily.items():
                r = ret.reindex(vami_df.index).dropna()
                if len(r) < window + 5:
                    continue

                vol = r.rolling(window).std(ddof=1) * np.sqrt(252)
                rolling_vol[name] = vol

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
                st.info(
                    "Not enough data to compute rolling metrics for the selected period."
                )

        # 5) Sector vs portfolios correlations
        if st.session_state.show_sec_vs_port_corr:
            st.subheader(
                "Sector correlations vs portfolios/benchmark (investment period)"
            )
            st.dataframe(corr_df.round(3))

        # 6) Sector correlation matrix
        if st.session_state.show_sec_corr_matrix:
            st.subheader(
                "Correlation matrix between selected sectors (investment period)"
            )
            sector_corr_matrix = sec_daily.corr()
            if not sector_corr_matrix.empty:
                fig, ax = plt.subplots(
                    figsize=(
                        0.6 * len(sector_corr_matrix.columns) + 4,
                        0.6 * len(sector_corr_matrix.index) + 4,
                    )
                )
                sns.heatmap(
                    sector_corr_matrix,
                    annot=True,
                    fmt=".2f",
                    cmap="coolwarm",
                    vmin=-1,
                    vmax=1,
                    square=True,
                    cbar=True,
                    ax=ax,
                )
                ax.set_title("Sector Correlation Heat Map")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Not enough data to compute sector correlations.")

        # 7) Sector volatilities
        if st.session_state.show_sec_vol:
            st.subheader("Sector volatilities (investment period)")
            sec_daily_vol = sector_rets.reindex(vami_df.index).dropna(how="any")
            if sec_daily_vol.empty:
                st.info("Not enough sector data to compute volatilities.")
            else:
                sec_vol = (
                    sec_daily_vol.std(ddof=1) * np.sqrt(252)
                ).sort_values(ascending=False)
                st.dataframe(
                    sec_vol.to_frame("Volatility (ann.)").style.format("{:.2%}"),
                    use_container_width=True,
                )

        # 8) Risk contributions
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
                    show_blocks.append(
                        ("ERC", risk_contrib_table(cov_last, cols_last, w_erc))
                    )

                if "MDP" in methods_to_show and cov_last is not None:
                    w_mdp = mdp_weights(cov_last)
                    show_blocks.append(
                        ("MDP", risk_contrib_table(cov_last, cols_last, w_mdp))
                    )

                if cov_last is not None:
                    w_ew = np.ones(len(cols_last)) / len(cols_last)
                    show_blocks.append(
                        ("Equal Weight", risk_contrib_table(cov_last, cols_last, w_ew))
                    )

                    # Approximate S&P 500 exposures via regression
                    win_idx = trading_lookback_window(
                        sector_ret_for_port, t0_last, 252
                    ).index
                    X_win = sector_ret_for_port.loc[win_idx]
                    spx_ret_full = prices["S&P 500"].pct_change()
                    y_win = spx_ret_full.loc[win_idx].dropna()
                    X_win = X_win.loc[y_win.index]
                    model = sm.OLS(y_win, sm.add_constant(X_win)).fit()
                    w_sp = np.clip(model.params[1:].values, 0, None)
                    w_sp = w_sp / (w_sp.sum() + 1e-18)
                    show_blocks.append(
                        ("S&P 500", risk_contrib_table(cov_last, cols_last, w_sp))
                    )

                if show_blocks:
                    st.subheader("Risk contributions (relative)")
                    for i in range(0, len(show_blocks), 2):
                        cols_ui = st.columns(2)
                        for (name, rc_df), col in zip(show_blocks[i : i + 2], cols_ui):
                            with col:
                                st.markdown(f"#### {name}")
                                st.bar_chart(rc_df["Rel_RC"])
                


                    st.subheader("Detailed tables")
                    for name, rc_df in show_blocks:
                        st.markdown(f"**{name} — Weights & risk contributions**")
                        df_formatted = rc_df.copy()
                        df_formatted["Weight"] = df_formatted["Weight"].map(
                            "{:.2%}".format
                        )
                        df_formatted["Abs_RC"] = df_formatted["Abs_RC"].map(
                            "{:.6f}".format
                        )
                        df_formatted["Rel_RC"] = df_formatted["Rel_RC"].map(
                            "{:.2%}".format
                        )
                        st.dataframe(df_formatted, use_container_width=True)
                        st.divider()

        # 9) Portfolio weight evolution
        if st.session_state.show_weight_evolution:
            st.subheader("Portfolio weight evolution (rebalance dates)")
            st.caption(
                f"Weights recomputed using ~252 trading days of history at each {freq_label.lower()} rebalance."
            )

            forced_methods = ["ERC", "MDP"]
            available_methods = [
                m
                for m in forced_methods
                if m in weight_history and not weight_history[m].empty
            ]

            if not available_methods:
                st.info(
                    "Not enough data to compute weight evolution for the selected period."
                )
            else:
                for method in available_methods:
                    st.markdown(f"### {method}")
                    wdf = weight_history[method]
                    wdf = wdf[[c for c in wdf.columns if c in picked]]
                    if wdf.empty:
                        st.info(f"No weight data available for {method}.")
                    else:
                        st.line_chart(wdf)

    # =====================================================
    # PAGE 4: TECHNICAL DETAILS (LaTeX)
    # =====================================================
    elif page == "Technical details":
        st.title("Technical details")

        st.markdown(
            """
        This tab explains the main formulas used in SectorLab.
        You can select a specific concept from the menu below to see its definition,
        formula, and the meaning of each variable.
        """
        )

        topic = st.selectbox(
            "Choose a topic to explore:",
            [
                "Overview (all topics)",
                "Growth of 1 unit invested",
                "Annualized return (CAGR)",
                "Annualized volatility",
                "Sharpe ratio",
                "Sortino ratio",
                "Maximum drawdown",
                "Equal-weight portfolio",
                "Equal Risk Contribution (ERC)",
                "Maximum Diversification Portfolio (MDP)",
                "Rebalancing & lookback windows",
                "Risk contributions",
            ],
        )

        # ---------- Helper: section title separator ----------
        def section_title(label: str):
            st.markdown(f"---\n### {label}")

        # ---------- GROWTH OF 1 UNIT ----------
        def show_growth():
            section_title("Growth of 1 unit invested")

            st.markdown(
                """
                The *Growth of 1 unit invested* represents the value of a portfolio over time
                assuming an initial investment of 1 unit of capital. It is computed through the
                compounding of daily returns. If $r_1, r_2, \dots, r_t$ denote daily portfolio
                returns and $V_0 = 1$, the value at time $t$ is
                """
            )
            st.latex(r"V_t = \prod_{u=1}^{t} (1 + r_u)")

            st.markdown(
                """
                This measure captures the cumulative effect of gains and losses through time.
                Because each day's return applies to the portfolio value achieved the previous day,
                the process reflects true compounding: positive returns generate growth on an
                increasingly larger base, while negative returns reduce the base on which future
                returns apply. Even small differences in daily performance therefore lead to
                meaningful divergences over longer horizons.

                Academically, the series $V_t$ provides the foundation for evaluating long-term
                performance, since it allows for direct comparisons between portfolio construction
                methods independently of their weighting schemes or risk profiles. It is the
                basis for annualized return (CAGR), maximum drawdown, and other performance metrics.

                Intuitively, it can be interpreted as the answer to a simple question: if an
                investor had allocated one unit of capital to this portfolio on the first day of
                the sample, how would that investment have evolved over time? The resulting curve
                illustrates both the growth potential and the path dependency of the portfolio's
                performance.

                **Variables**  
                $V_t$ — portfolio value at time $t$  
                $r_u$ — daily portfolio return on day $u$
                """
            )


        # ---------- CAGR ----------
        def show_cagr():
            section_title("Annualized return (CAGR)")

            st.markdown(
                """
                The **Compound Annual Growth Rate (CAGR)** represents the constant annual rate of
                return that would produce the same final value as the actual sequence of daily
                returns. It translates the total growth of a portfolio over a given period into an
                equivalent yearly growth rate, allowing performance to be compared across samples
                of different lengths. If the portfolio starts at 1 and reaches a value of $V_T$
                after $T$ trading days, assuming 252 trading days per year, the annualized return is
                """
            )
            st.latex(r"\text{CAGR} = V_T^{\,252/T} - 1")

            st.markdown(
                """
                CAGR is useful because it removes the effects of short-term fluctuations and
                expresses growth on a yearly scale under the assumption of smooth compounding. It
                summarises long-term performance in a single figure and facilitates comparisons
                between portfolios that may have experienced different paths of volatility and
                interim drawdowns.

                **Variables**  
                $V_T$ — final portfolio value over the sample (starting from 1)  
                $T$ — number of trading days in the sample  
                $252$ — assumed number of trading days per year
                """
            )


        # ---------- ANNUALIZED VOL ----------
        def show_vol():
            section_title("Annualized volatility")

            st.markdown(
                """
                Annualized volatility measures the variability of daily portfolio returns scaled
                to a yearly horizon. It captures the extent to which returns fluctuate around
                their mean and is commonly interpreted as a measure of total risk. If
                $\\sigma_{\\text{daily}}$ denotes the standard deviation of daily returns, the
                annualized volatility is obtained by
                """
            )
            st.latex(r"\sigma_{\text{ann}} = \sigma_{\text{daily}} \sqrt{252}")

            st.markdown(
                """
                This scaling reflects the assumption that daily returns are independent and
                identically distributed, so that variance grows proportionally with time. The
                metric provides a consistent way to compare risk across portfolios and aligns the
                daily return distribution with the annual horizon on which most investment
                decisions are evaluated.

                **Variables**  
                $\\sigma_{\\text{daily}}$ — standard deviation of daily returns  
                $\\sigma_{\\text{ann}}$ — annualized volatility  
                $252$ — assumed number of trading days per year
                """
            )


        # ---------- SHARPE ----------
        def show_sharpe():
            section_title("Sharpe ratio")

            st.markdown(
                """
                The Sharpe ratio evaluates a portfolio’s performance by measuring the average
                excess return earned per unit of total risk. It compares the portfolio’s daily
                returns to the daily risk-free rate and relates this difference to the volatility
                of excess returns. With an annual risk-free rate $r_f$, the daily excess return is
                defined as
                """
            )
            st.latex(r"x_t = r_t - \frac{r_f}{252}")

            st.markdown(
                """
                The annualized Sharpe ratio is then computed as
                """
            )
            st.latex(
                r"\text{Sharpe} = \frac{\mathbb{E}[x_t]}{\sqrt{\text{Var}(x_t)}} \sqrt{252}"
            )

            st.markdown(
                """
                This metric provides a normalized measure of risk-adjusted performance: a higher
                Sharpe ratio indicates that the portfolio generates more excess return relative to
                the amount of risk taken. Conceptually, it allows different strategies to be
                compared on a common scale, even if they exhibit different volatilities or
                return profiles. It is widely used in asset management because it summarizes both
                reward and risk in a single statistic and relies on the idea that volatility is an
                appropriate measure of uncertainty in returns.

                **Variables**  
                $r_t$ — daily portfolio return  
                $r_f$ — annual risk-free rate  
                $x_t$ — daily excess return ($r_t - r_f/252$)  
                $\\mathbb{E}[x_t]$ — mean of daily excess returns  
                $\\text{Var}(x_t)$ — variance of daily excess returns  
                $252$ — assumed number of trading days per year
                """
            )


        # ---------- SORTINO ----------
        def show_sortino():
            section_title("Sortino ratio")

            st.markdown(
                """
                The Sortino ratio measures a portfolio’s risk-adjusted performance while
                distinguishing between desirable and undesirable volatility. Unlike the Sharpe
                ratio, which treats all return fluctuations symmetrically, the Sortino ratio
                focuses exclusively on downside movements, reflecting the idea that negative
                deviations from the target return are more relevant for investors than positive
                ones. With daily excess returns $x_t$, the downside return is defined as
                """
            )
            st.latex(r"D_t = \min(x_t, 0)")

            st.markdown(
                """
                and the corresponding downside deviation is
                """
            )
            st.latex(r"\sigma_{\text{down}} = \sqrt{\mathbb{E}[D_t^2]}")

            st.markdown(
                """
                The annualized Sortino ratio is then
                """
            )
            st.latex(
                r"\text{Sortino} = \frac{\mathbb{E}[x_t]}{\sigma_{\text{down}}} \sqrt{252}"
            )

            st.markdown(
                """
                Conceptually, the Sortino ratio isolates the component of risk that reflects
                underperformance relative to the risk-free rate. By ignoring upside variation,
                it provides a more targeted assessment of downside risk and is often considered a
                more accurate measure of the quality of returns for portfolios with asymmetric or
                skewed distributions. A higher Sortino ratio indicates that the portfolio generates
                stronger excess returns relative to the magnitude of its negative fluctuations.

                **Variables**  
                $x_t$ — daily excess return  
                $D_t$ — downside return ($\min(x_t, 0)$)  
                $\sigma_{\text{down}}$ — downside deviation  
                $252$ — assumed number of trading days per year
                """
            )

        def show_maxdd():
            section_title("Maximum drawdown")

            st.markdown(
                """
                Maximum drawdown quantifies the largest decline a portfolio experiences from any
                historical peak to a subsequent trough. It captures the worst cumulative loss
                that an investor would have encountered over the sample period and provides a
                direct measure of downside risk. If $V_t$ denotes the portfolio value at time $t$,
                the running maximum is defined as
                """
            )
            st.latex(r"M_t = \max_{u \le t} V_u")

            st.markdown(
                """
                and the drawdown at time $t$ is
                """
            )
            st.latex(r"\text{DD}_t = \frac{V_t}{M_t} - 1")

            st.markdown(
                """
                The maximum drawdown is the smallest value of $\\text{DD}_t$ over the sample,
                representing the deepest loss experienced before a recovery occurred. This metric
                is widely used to evaluate downside risk because it reflects both the severity and
                duration of adverse market conditions.
                """
            )
            st.latex(r"\text{MaxDD} = \min_t \text{DD}_t")

            st.markdown(
                """
                **Variables**  
                $V_t$ — portfolio value at time $t$  
                $M_t$ — running maximum of portfolio value  
                $\\text{DD}_t$ — drawdown at time $t$  
                $\\text{MaxDD}$ — maximum drawdown over the sample
                """
            )
         
        # ---------- Equal-Weight Portfolio ----------

        def show_ew():
            section_title("Equal-weight portfolio")

            st.markdown(
                """
                The equal-weight portfolio allocates the same proportion of capital to each of the
                $N$ selected sectors. This specification does not depend on historical returns,
                volatilities, or covariances and therefore serves as a simple benchmark for
                comparing more sophisticated allocation strategies. The weight assigned to each
                sector is
                """
            )
            st.latex(r"w_i = \frac{1}{N}, \quad i = 1,\dots,N")

            st.markdown(
                """
                **Variables**  
                $N$ — number of selected sectors  
                $w_i$ — weight of sector $i$
                """
            )
        
        # ---------- Equal Risk Contribution (ERC) ----------

        def show_erc():
            section_title("Equal Risk Contribution (ERC)")

            st.markdown(
                """
                The Equal Risk Contribution portfolio seeks a set of weights such that each sector
                contributes the same amount of risk to the overall portfolio. Let $\\Sigma$ denote
                the covariance matrix of sector returns and $w$ the vector of weights. The marginal
                contribution to risk of sector $i$ is
                """
            )
            st.latex(r"\text{MRC}_i = (\Sigma w)_i")

            st.markdown(
                """
                The portfolio variance is
                """
            )
            st.latex(r"\sigma_p^2 = w^\top \Sigma w")

            st.markdown(
                """
                and the absolute risk contribution of sector $i$ is
                """
            )
            st.latex(r"\text{RC}_i = w_i (\Sigma w)_i")

            st.markdown(
                """
                The ERC objective is to equalize these contributions across sectors under the
                constraints that weights are non-negative and sum to one:
                """
            )
            st.latex(r"\text{RC}_1 = \text{RC}_2 = \cdots = \text{RC}_N")

            st.markdown(
                """
                In practice, the ERC portfolio is obtained through an iterative fixed-point
                procedure that adjusts the weights until the risk contributions converge to a
                common value. This approach produces a diversified allocation that balances risk
                rather than capital.
                """
            )

            st.markdown(
                """
                **Variables**  
                $w$ — vector of portfolio weights  
                $\\Sigma$ — covariance matrix  
                $\\text{MRC}_i$ — marginal risk contribution  
                $\\text{RC}_i$ — absolute risk contribution  
                $\\sigma_p^2$ — total portfolio variance
                """
            )

       # ---------- Maximum Diversification Portfolio (MDP) ----------
        def show_mdp():
            section_title("Maximum Diversification Portfolio (MDP)")

            st.markdown(
                """
                The Maximum Diversification Portfolio maximizes the diversification ratio, which
                compares the weighted average of individual volatilities to the total portfolio
                volatility. For weights $w$, volatilities $\\sigma$, and covariance matrix
                $\\Sigma$, the ratio is
                """
            )
            st.latex(r"\text{DR}(w) = \frac{w^\top \sigma}{\sqrt{w^\top \Sigma w}}")

            st.markdown(
                """
                Maximizing this quantity under long-only, fully invested constraints
                ($w_i \\ge 0$, $\\sum_i w_i = 1$) leads to a portfolio that benefits as much as
                possible from imperfect correlations across sectors. The numerator represents the
                average standalone risk of the assets, while the denominator represents the total
                risk of the combined portfolio. A higher ratio indicates greater diversification
                benefits.
                """
            )

            st.markdown(
                """
                In the implementation, $\\Sigma$ is estimated using the most recent 252 trading
                days. The initial weight vector is obtained from the unconstrained solution
                $w \\propto \\Sigma^{-1} \\sigma$, and projected gradient ascent is then applied
                to optimize the ratio while maintaining feasibility through projection onto the
                simplex.
                """
            )

            st.markdown(
                """
                **Variables**  
                $w$ — vector of portfolio weights  
                $\\sigma$ — vector of sector volatilities  
                $\\Sigma$ — covariance matrix  
                $\\text{DR}(w)$ — diversification ratio
                """
            )

        # ---------- Rebalancing & Lookback Windows ----------

        def show_rebalancing():
            section_title("Rebalancing and lookback windows")

            st.markdown(
                """
                Portfolio weights are updated periodically according to the rebalancing frequency
                selected in the sidebar (monthly, quarterly, or yearly). For ERC and MDP, each
                rebalance uses a rolling estimation window of approximately 252 trading days to
                compute the covariance matrix and other required statistics. Between rebalancing
                dates, weights remain fixed and the portfolio value evolves according to
                """
            )
            st.latex(r"V_{t+1} = V_t (1 + r_{p,t+1}), \quad r_{p,t+1} = w^\top r_{t+1}")

            st.markdown(
                """
                This structure separates the estimation of risk parameters from the actual growth
                of the portfolio, aligning the simulations with typical industry practices in
                quantitative asset management.
                """
            )

            st.markdown(
                """
                **Variables**  
                $r_{p,t+1}$ — portfolio return between $t$ and $t+1$  
                $r_{t+1}$ — vector of asset returns  
                $w$ — portfolio weights between rebalancing dates  
                $V_t$ — portfolio value at time $t$
                """
            )

        # ---------- Risk Contributions ----------

        def show_risk_contrib():
            section_title("Risk contributions in the dashboard")

            st.markdown(
                """
                On the most recent valid lookback window, the dashboard reports both absolute and
                relative risk contributions of each sector. Given weights $w$ and covariance
                matrix $\\Sigma$, the absolute contribution of sector $i$ is
                """
            )
            st.latex(r"\text{RC}_i = w_i (\Sigma w)_i")

            st.markdown(
                """
                and the relative contribution is
                """
            )
            st.latex(r"\text{RelRC}_i = \frac{\text{RC}_i}{\sum_j \text{RC}_j}")

            st.markdown(
                """
                These quantities illustrate how total portfolio risk is distributed across sectors.
                The dashboard visualizes relative contributions through bar charts, while tables
                display both weights and risk contributions. This decomposition provides insight
                into which sectors dominate portfolio risk and how risk allocation differs across
                ERC, MDP, Equal-Weight, and the benchmark.
                """
            )

            st.markdown(
                """
                **Variables**  
                $w_i$ — weight of sector $i$  
                $\\Sigma$ — covariance matrix  
                $\\text{RC}_i$ — absolute risk contribution  
                $\\text{RelRC}_i$ — relative contribution to total portfolio risk
                """
            )



        # ---------- ROUTING BASED ON TOPIC ----------
        if topic == "Overview (all topics)":
            # Show everything in a logical order
            st.header("Performance metrics")
            show_growth()
            show_cagr()
            show_vol()
            show_sharpe()
            show_sortino()
            show_maxdd()

            st.header("Portfolio constructions")
            show_ew()
            show_erc()
            show_mdp()

            st.header("Rebalancing & risk contributions")
            show_rebalancing()
            show_risk_contrib()

        else:
            # Show only the selected concept
            if topic == "Growth of 1 unit invested":
                show_growth()
            elif topic == "Annualized return (CAGR)":
                show_cagr()
            elif topic == "Annualized volatility":
                show_vol()
            elif topic == "Sharpe ratio":
                show_sharpe()
            elif topic == "Sortino ratio":
                show_sortino()
            elif topic == "Maximum drawdown":
                show_maxdd()
            elif topic == "Equal-weight portfolio":
                show_ew()
            elif topic == "Equal Risk Contribution (ERC)":
                show_erc()
            elif topic == "Maximum Diversification Portfolio (MDP)":
                show_mdp()
            elif topic == "Rebalancing & lookback windows":
                show_rebalancing()
            elif topic == "Risk contributions":
                show_risk_contrib()



# ---------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
