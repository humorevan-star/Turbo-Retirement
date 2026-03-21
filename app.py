import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="VOO vs SPXL — Springboard Simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .main { background-color: #0b0e14; color: #e1e1e1; }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }
    </style>
""", unsafe_allow_html=True)


# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.markdown("## Strategy Controls")
initial_inv = st.sidebar.number_input("Initial Investment ($)", value=0, min_value=0, step=500,
    help="Lump sum invested on day 1, split at the same ratio as your monthly DCA.")
monthly_inv = st.sidebar.number_input("Monthly Contribution ($)", value=500, min_value=10, step=50)

st.sidebar.markdown("---")
st.sidebar.markdown("### Market Assumptions")
ann_ret = st.sidebar.slider("S&P 500 Annual Return (%)", 0.0, 20.0, 10.0, 0.5) / 100
ann_vol = st.sidebar.slider("Annual Volatility (%)", 5.0, 45.0, 18.0, 0.5) / 100

st.sidebar.markdown("---")
st.sidebar.markdown("### Projection Settings")
proj_years = st.sidebar.slider("Years to Project Forward", 1, 40, 20)
n_sim = st.sidebar.select_slider("Monte Carlo Paths", options=[100, 300, 500, 1000], value=300)

st.sidebar.markdown("---")
st.sidebar.markdown("### Instrument Parameters")
voo_expense = st.sidebar.number_input("VOO Expense Ratio (%)", value=0.03, step=0.01, format="%.2f") / 100
spxl_expense = st.sidebar.number_input("SPXL Expense Ratio (%)", value=0.91, step=0.01, format="%.2f") / 100
leverage = st.sidebar.number_input("SPXL Leverage Factor", value=3.0, step=0.5, format="%.1f")


# =============================================================================
# MATH CORE
# =============================================================================
def vol_drag(lev: float, sigma: float) -> float:
    return lev * (lev - 1) / 2 * sigma ** 2


def net_annual(lev: float, ann_return: float, sigma: float, expense: float) -> float:
    return lev * ann_return - vol_drag(lev, sigma) - expense


def dca_gbm_paths(years, monthly, ann_r, ann_s, expense, lev, n_paths, seed=42, initial=0):
    rng = np.random.default_rng(seed)
    days = years * 252
    dt = 1 / 252
    drag = vol_drag(lev, ann_s)
    drift_daily = (lev * ann_r - drag - expense - 0.5 * (lev * ann_s) ** 2) * dt
    sigma_daily = lev * ann_s * np.sqrt(dt)

    Z = rng.standard_normal((days, n_paths))
    log_rets = drift_daily + sigma_daily * Z

    portfolio = np.zeros((days + 1, n_paths))
    portfolio[0] = initial  # lump sum on day 1
    for i in range(days):
        if i % 21 == 0:
            portfolio[i] += monthly
        portfolio[i + 1] = portfolio[i] * np.exp(log_rets[i])

    invested = initial + monthly * (days // 21)
    return portfolio, invested


@st.cache_data(ttl=3600)
def load_historical() -> pd.DataFrame:
    """
    Try multiple data sources in order until one works.
    1. yfinance with browser-like headers (works on most hosts)
    2. yfinance Ticker fallback (avoids bulk-download rate limits)
    3. pandas-datareader stooq (works on Streamlit Cloud)
    """
    def _clean(df):
        df.columns = [str(c) for c in df.columns]
        for col in ["VOO", "SPXL"]:
            if col not in df.columns:
                return pd.DataFrame()
        return df[["VOO", "SPXL"]].dropna()

    # ── attempt 1: yfinance with session headers ──────────────────────────
    try:
        import requests
        from requests import Session
        session = Session()
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        })
        raw = yf.download(
            ["VOO", "SPXL"], start="2010-01-01",
            auto_adjust=True, progress=False,
            session=session,
        )
        if not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                level0 = raw.columns.get_level_values(0).unique().tolist()
                field = "Close" if "Close" in level0 else level0[0]
                df = raw[field].copy()
            else:
                df = raw.copy()
            result = _clean(df)
            if not result.empty:
                return result
    except Exception:
        pass

    # ── attempt 2: yfinance Ticker one-by-one ────────────────────────────
    try:
        frames = {}
        for ticker in ["VOO", "SPXL"]:
            t = yf.Ticker(ticker)
            hist = t.history(start="2010-01-01", auto_adjust=True)
            if not hist.empty:
                frames[ticker] = hist["Close"]
        if len(frames) == 2:
            df = pd.DataFrame(frames).dropna()
            if not df.empty:
                return df
    except Exception:
        pass

    # ── attempt 3: pandas-datareader stooq ───────────────────────────────
    try:
        import pandas_datareader.data as web
        frames = {}
        # stooq uses US tickers with .US suffix
        for ticker, stooq_sym in [("VOO", "VOO.US"), ("SPXL", "SPXL.US")]:
            df_t = web.DataReader(stooq_sym, "stooq", start="2010-01-01")
            df_t = df_t.sort_index()
            frames[ticker] = df_t["Close"]
        if len(frames) == 2:
            df = pd.DataFrame(frames).dropna()
            if not df.empty:
                return df
    except Exception:
        pass

    return pd.DataFrame()


def run_historical_dca(prices: pd.DataFrame, monthly: float, initial: float = 0) -> pd.DataFrame:
    results = {"Date": [], "VOO": [], "SPXL": [], "Invested": []}
    voo_shares = spxl_shares = invested = 0.0
    last_month = -1
    initial_deployed = False

    for date, row in prices.iterrows():
        m = date.month
        vp = float(row.get("VOO", 0))
        sp = float(row.get("SPXL", 0))

        # Deploy lump sum on first valid day
        if not initial_deployed and vp > 0 and sp > 0 and initial > 0:
            voo_shares += (initial / 2) / vp
            spxl_shares += (initial / 2) / sp
            invested += initial
            initial_deployed = True

        if m != last_month:
            if vp > 0:
                voo_shares += monthly / vp
            if sp > 0:
                spxl_shares += monthly / sp
            invested += monthly * 2
            last_month = m

        results["Date"].append(date)
        results["VOO"].append(round(voo_shares * float(row.get("VOO", 0)), 2))
        results["SPXL"].append(round(spxl_shares * float(row.get("SPXL", 0)), 2))
        results["Invested"].append(round(invested / 2, 2))

    return pd.DataFrame(results).set_index("Date")


def compute_drawdown(series: pd.Series) -> pd.Series:
    return ((series - series.cummax()) / series.cummax() * 100).clip(upper=0)


def generate_signals(prices: pd.DataFrame, ma_days=200) -> pd.DataFrame:
    df = prices.copy()
    df["MA"] = df["SPXL"].rolling(ma_days, min_periods=1).mean()
    df["Signal"] = 0
    in_lev = False
    signals = []
    for i, (date, row) in enumerate(df.iterrows()):
        above = row["SPXL"] > row["MA"]
        if not in_lev and above:
            in_lev = True
            signals.append((date, "BUY", row["SPXL"]))
        elif in_lev and not above:
            in_lev = False
            signals.append((date, "SELL", row["SPXL"]))
    return pd.DataFrame(signals, columns=["Date", "Type", "Price"]).set_index("Date")




def fmt_currency(v: float) -> str:
    if v >= 1_000_000:
        return f"${v / 1_000_000:.2f}M"
    return f"${v:,.0f}"


def kpi(col, label, val, sub=None, color="#e8eaf0"):
    sub_html = (
        f'<p style="margin:2px 0 0;font-size:11px;color:#4a5568;font-family:monospace;">{sub}</p>'
        if sub else ""
    )
    col.markdown(
        f'<div style="background:#111318;padding:14px 16px;border-radius:4px;'
        f'border:1px solid rgba(255,255,255,0.07);text-align:center;">'
        f'<p style="margin:0 0 3px;font-size:9px;color:#4a5568;font-family:monospace;'
        f'text-transform:uppercase;letter-spacing:0.1em;">{label}</p>'
        f'<p style="margin:0;font-family:monospace;font-size:20px;font-weight:600;color:{color};">{val}</p>'
        f'{sub_html}</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# SPRINGBOARD ENGINE v4 — "Opportunistic Predator" with Profit Lock
# =============================================================================
# FULL CYCLE:
#   BASE STATE:  100% VOO — accumulate war chest, zero vol decay, minimal fees
#   TIER ENTRY:  As market drops from 50-day high, shift VOO → SPXL in chunks
#     -10%  → 20% of current VOO bucket moves to SPXL
#     -20%  → 40% of remaining VOO moves to SPXL
#     -30%  → 100% remaining VOO moves to SPXL (full coiled spring)
#   PROFIT LOCK: When market hits new ALL-TIME HIGH → exit SPXL → back to 100% VOO
#                Whipsaw guard: must be >5 days since last tier entry before locking
#
# Key difference from v3: tiers are based on DROP FROM 50-DAY HIGH (not ATH)
# so the system reacts faster to corrections, and the ATH exit creates a clean cycle.
# =============================================================================

DEFAULT_TIERS = [
    (0.00,   0),   # at 50d high → 0% SPXL (100% VOO)
    (0.10,  20),   # -10% → 20% of VOO shifted to SPXL
    (0.20,  52),   # -20% → 40% of remaining (0.8 * 0.4 + 0.2 = 52% total SPXL)
    (0.30, 100),   # -30% → all-in SPXL
]

def _tier_spxl(drawdown: float, tiers: list) -> float:
    if drawdown <= tiers[0][0]:
        return float(tiers[0][1])
    if drawdown >= tiers[-1][0]:
        return float(tiers[-1][1])
    for i in range(len(tiers) - 1):
        dd0, pct0 = tiers[i]
        dd1, pct1 = tiers[i + 1]
        if dd0 <= drawdown <= dd1:
            t = (drawdown - dd0) / (dd1 - dd0)
            return float(pct0 + t * (pct1 - pct0))
    return float(tiers[-1][1])


def run_springboard(
    years, monthly,
    ann_r, ann_s,
    voo_exp, spxl_exp, lev,
    tiers=None,
    rebal_speed=0.30,
    n_paths=200, seed=42,
    initial=0,
    whipsaw_guard=5,   # min days in tier before profit lock can trigger
):
    if tiers is None:
        tiers = DEFAULT_TIERS

    rng  = np.random.default_rng(seed)
    days = years * 252
    dt   = 1 / 252

    drag_spxl  = vol_drag(lev, ann_s)
    drift_voo  = (ann_r - voo_exp  - 0.5 * ann_s**2) * dt
    sig_voo    = ann_s * np.sqrt(dt)
    drift_spxl = (lev*ann_r - drag_spxl - spxl_exp - 0.5*(lev*ann_s)**2) * dt
    sig_spxl   = lev * ann_s * np.sqrt(dt)
    drift_idx  = (ann_r - 0.5 * ann_s**2) * dt
    sig_idx    = ann_s * np.sqrt(dt)

    all_spring     = np.zeros((days+1, n_paths))
    all_voo_base   = np.zeros((days+1, n_paths))  # pure VOO DCA benchmark
    all_spxl_base  = np.zeros((days+1, n_paths))  # pure SPXL DCA benchmark
    all_spxl_alloc = np.zeros((days+1, n_paths))

    Z = rng.standard_normal((days, n_paths))

    for p in range(n_paths):
        # strategy buckets
        voo_b = initial * 1.0   # start 100% VOO
        spxl_b = 0.0
        # pure benchmarks
        voo_base  = initial
        spxl_base = initial

        idx = 1.0; idx_ath = 1.0
        high_50d = np.ones(51)   # rolling 50-day high buffer
        buf_ptr  = 0
        days_since_entry = whipsaw_guard + 1
        locked = False  # profit-lock state

        for i in range(days):
            r_voo  = drift_voo  + sig_voo  * Z[i, p]
            r_spxl = drift_spxl + sig_spxl * Z[i, p]
            r_idx  = drift_idx  + sig_idx  * Z[i, p]

            voo_b     = max(voo_b    * np.exp(r_voo),  0.0)
            spxl_b    = max(spxl_b   * np.exp(r_spxl), 0.0)
            voo_base  = max(voo_base  * np.exp(r_voo),  0.0)
            spxl_base = max(spxl_base * np.exp(r_spxl), 0.0)

            idx *= np.exp(r_idx)
            idx_ath = max(idx_ath, idx)

            # update 50-day rolling high
            high_50d[buf_ptr % 51] = idx
            buf_ptr += 1
            h50 = np.max(high_50d)

            dd_from_50d = max(0.0, (h50 - idx) / h50)
            at_new_ath  = (idx >= idx_ath * 0.999)  # within 0.1% of ATH = new high

            days_since_entry += 1

            # ── PROFIT LOCK: new ATH → exit all SPXL → 100% VOO ───────────────
            if at_new_ath and spxl_b > 0 and days_since_entry > whipsaw_guard:
                voo_b  += spxl_b
                spxl_b  = 0.0
                locked  = True

            # ── TIER ENTRY: drawdown from 50d high ────────────────────────────
            target_spxl_pct = _tier_spxl(dd_from_50d, tiers) / 100.0
            total = voo_b + spxl_b
            if total > 0:
                current_spxl_pct = spxl_b / total
                if target_spxl_pct > current_spxl_pct + 0.01:  # entering deeper tier
                    gap = target_spxl_pct - current_spxl_pct
                    move = total * gap * rebal_speed
                    if voo_b >= move:
                        spxl_b += move
                        voo_b  -= move
                        days_since_entry = 0
                        locked = False

            # ── MONTHLY DCA: contribute at current allocation ─────────────────
            if i > 0 and i % 21 == 0:
                total = voo_b + spxl_b
                cur_spxl = spxl_b / total if total > 0 else 0.0
                spxl_b   += monthly * cur_spxl
                voo_b    += monthly * (1.0 - cur_spxl)
                voo_base  += monthly
                spxl_base += monthly

            total = voo_b + spxl_b
            all_spring[i+1, p]     = total
            all_voo_base[i+1, p]   = voo_base
            all_spxl_base[i+1, p]  = spxl_base
            all_spxl_alloc[i+1, p] = spxl_b / total if total > 0 else 0.0

    return {
        "voo_p50":        np.percentile(all_voo_base,    50, axis=1),
        "spxl_p50":       np.percentile(all_spxl_base,   50, axis=1),
        "spring_p10":     np.percentile(all_spring,      10, axis=1),
        "spring_p25":     np.percentile(all_spring,      25, axis=1),
        "spring_p50":     np.percentile(all_spring,      50, axis=1),
        "spring_p75":     np.percentile(all_spring,      75, axis=1),
        "spring_p90":     np.percentile(all_spring,      90, axis=1),
        "spxl_alloc_p50": np.percentile(all_spxl_alloc,  50, axis=1),
        "avg_spxl":       float(np.mean(np.percentile(all_spxl_alloc, 50, axis=1)) * 100),
    }


def _days_to_target(series: np.ndarray, target: float) -> int:
    """Return first index where series >= target, or -1 if never reached."""
    idx = np.where(series >= target)[0]
    return int(idx[0]) if len(idx) > 0 else -1


def run_ath_rotation_backtest(
    prices: pd.DataFrame,
    monthly: float,
    tiers=None,
    rebal_speed: float = 0.30,
    initial: float = 0,
    whipsaw_guard: int = 5,
) -> pd.DataFrame:
    """Real historical backtest — full Profit Lock cycle on actual prices."""
    if tiers is None:
        tiers = DEFAULT_TIERS

    voo_b = initial; spxl_b = 0.0
    voo_ath = None; lm = -1; invested = initial
    voo_shares_pure = 0.0; spxl_shares_pure = 0.0
    initial_deployed = (initial == 0)
    days_since_entry = whipsaw_guard + 1

    voo_prices   = prices["VOO"].values
    spxl_prices  = prices["SPXL"].values
    dates        = prices.index
    n            = len(dates)

    out_strategy  = np.zeros(n)
    out_alloc     = np.zeros(n)
    out_invested  = np.zeros(n)
    out_voo_pure  = np.zeros(n)
    out_spxl_pure = np.zeros(n)

    high_50d = None

    for i, date in enumerate(dates):
        vp = voo_prices[i]; sp = spxl_prices[i]
        if vp <= 0 or sp <= 0:
            out_strategy[i]  = voo_b + spxl_b
            out_alloc[i]     = spxl_b / max(voo_b + spxl_b, 1)
            out_invested[i]  = invested
            out_voo_pure[i]  = voo_shares_pure * vp
            out_spxl_pure[i] = spxl_shares_pure * sp
            continue

        # deploy lump sum day 1
        if not initial_deployed and initial > 0:
            voo_b = initial; spxl_b = 0.0
            voo_shares_pure  += (initial / 2) / vp
            spxl_shares_pure += (initial / 2) / sp
            initial_deployed = True

        # daily price-return compounding
        if i > 0:
            pv = voo_prices[i-1]; ps = spxl_prices[i-1]
            if pv > 0: voo_b  *= vp / pv
            if ps > 0: spxl_b *= sp / ps

        voo_ath = max(voo_ath, vp) if voo_ath else vp

        # 50-day rolling high on VOO price
        lo = max(0, i - 49)
        h50 = float(np.max(voo_prices[lo:i+1]))
        dd_from_50d = max(0.0, (h50 - vp) / h50)
        at_new_ath  = (vp >= voo_ath * 0.999)

        days_since_entry += 1

        # PROFIT LOCK
        if at_new_ath and spxl_b > 0 and days_since_entry > whipsaw_guard:
            voo_b  += spxl_b
            spxl_b  = 0.0

        # TIER ENTRY
        target_spxl_pct = _tier_spxl(dd_from_50d, tiers) / 100.0
        total = voo_b + spxl_b
        if total > 0:
            cur_spxl = spxl_b / total
            if target_spxl_pct > cur_spxl + 0.01:
                gap  = target_spxl_pct - cur_spxl
                move = total * gap * rebal_speed
                if voo_b >= move:
                    spxl_b += move; voo_b -= move
                    # convert dollars to fractional shares for tracking
                    days_since_entry = 0

        # MONTHLY DCA
        if date.month != lm:
            total = voo_b + spxl_b
            cur_spxl = spxl_b / total if total > 0 else 0.0
            spxl_b   += monthly * cur_spxl
            voo_b    += monthly * (1.0 - cur_spxl)
            invested += monthly
            if vp > 0: voo_shares_pure  += monthly / vp
            if sp > 0: spxl_shares_pure += monthly / sp
            lm = date.month

        total = voo_b + spxl_b
        out_strategy[i]  = total
        out_alloc[i]     = spxl_b / total if total > 0 else 0.0
        out_invested[i]  = invested
        out_voo_pure[i]  = voo_shares_pure * vp
        out_spxl_pure[i] = spxl_shares_pure * sp

    return pd.DataFrame({
        "Strategy":   np.round(out_strategy,  2),
        "SPXL_alloc": np.round(out_alloc * 100, 1),
        "Invested":   np.round(out_invested,  2),
        "VOO_pure":   np.round(out_voo_pure,  2),
        "SPXL_pure":  np.round(out_spxl_pure, 2),
    }, index=dates)
# =============================================================================
# OMEGA LEAPS — Pure LEAPS DCA (no VOO core)
# =============================================================================
# Every dollar goes into deep-ITM calls.
# When no active position: cash sits in MMF at risk_free rate.
# Entry: deploy 100% of cash buffer into LEAPS when trigger fires.
# Exit (Profit Lock): sell all LEAPS → back to MMF when new ATH hit.
# Roll: every roll_months, extend to next 2-year cycle (~1% cost).
# Monthly DCA: goes into MMF immediately, deployed on next entry signal.
#
# LEAPS daily P&L model:
#   dV = V * (delta * r_idx - theta_daily)
#   where theta_daily = theta_mo / 21
#
# Effective leverage = delta * S / C  (Black-Scholes deep ITM)
# At delta=0.875, 15% ITM strike: effective leverage ≈ 3.5-5x on cash deployed
# =============================================================================

def _bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price, delta, effective leverage."""
    try:
        from scipy.stats import norm
        if T <= 0 or sigma <= 0:
            intrinsic = max(S - K, 0.01)
            return intrinsic, min((S - K) / S + 0.05, 0.99), S / max(S - K, 0.01)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        price  = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta  = float(norm.cdf(d1))
        eff_lev= delta * S / max(price, 0.01)
        return max(price, S - K + 0.001), delta, eff_lev
    except Exception:
        intrinsic = max(S - K, S * 0.10)
        delta = 0.875
        return intrinsic, delta, delta * S / intrinsic


def run_leaps_hybrid(
    years, monthly,
    ann_r, ann_s,
    voo_exp,                   # kept for API compat, not used
    voo_core_pct   = 0.0,      # ignored — pure LEAPS
    leaps_pct      = 1.0,      # ignored — 100% always
    trigger_dd     = 0.10,
    delta_target   = 0.875,
    leaps_tenor    = 1.75,
    roll_months    = 9,
    roll_cost_pct  = 0.01,
    theta_mo_pct   = 0.005,
    risk_free      = 0.04,
    n_paths        = 200,
    seed           = 42,
    initial        = 0,
):
    rng  = np.random.default_rng(seed)
    days = years * 252
    dt   = 1 / 252

    drift_idx  = (ann_r - 0.5 * ann_s**2) * dt
    sig_idx    = ann_s * np.sqrt(dt)
    drift_voo  = (ann_r - voo_exp - 0.5 * ann_s**2) * dt
    sig_voo    = ann_s * np.sqrt(dt)

    # pure SPXL benchmark
    drag_spxl  = vol_drag(3.0, ann_s)
    drift_spxl = (3.0*ann_r - drag_spxl - 0.0091 - 0.5*(3.0*ann_s)**2) * dt
    sig_spxl   = 3.0 * ann_s * np.sqrt(dt)

    all_leaps  = np.zeros((days+1, n_paths))
    all_voo    = np.zeros((days+1, n_paths))
    all_spxl   = np.zeros((days+1, n_paths))
    all_active = np.zeros((days+1, n_paths))

    Z = rng.standard_normal((days, n_paths))

    for p in range(n_paths):
        cash        = float(initial)   # MMF cash buffer
        leaps_val   = 0.0              # mark-to-market of open position
        leaps_active= False
        leaps_delta = delta_target
        days_since_entry = 999

        # pure benchmarks
        voo_b   = float(initial)
        spxl_b  = float(initial)

        idx = 1.0; idx_ath = 1.0

        for i in range(days):
            r_idx  = drift_idx  + sig_idx  * Z[i, p]
            r_voo  = drift_voo  + sig_voo  * Z[i, p]
            r_spxl = drift_spxl + sig_spxl * Z[i, p]

            # benchmarks compound daily
            voo_b  = max(voo_b  * np.exp(r_voo),  0.0)
            spxl_b = max(spxl_b * np.exp(r_spxl), 0.0)

            # LEAPS P&L
            if leaps_active:
                theta_d = theta_mo_pct / 21.0
                leaps_val = max(leaps_val * (np.exp(leaps_delta * r_idx) - theta_d), 0.0)
            else:
                # idle cash earns MMF
                cash *= np.exp(risk_free * dt)

            idx    *= np.exp(r_idx)
            idx_ath = max(idx_ath, idx)

            dd     = max(0.0, (idx_ath - idx) / idx_ath)
            at_ath = idx >= idx_ath * 0.999
            days_since_entry += 1

            # ── PROFIT LOCK ───────────────────────────────────────────────
            if leaps_active and at_ath and days_since_entry > 10:
                cash        += leaps_val
                leaps_val    = 0.0
                leaps_active = False

            # ── ROLL ──────────────────────────────────────────────────────
            if leaps_active and i > 0 and i % (roll_months * 21) == 0:
                leaps_val *= (1.0 - roll_cost_pct)

            # ── ENTRY ─────────────────────────────────────────────────────
            if not leaps_active and dd >= trigger_dd and cash > 0:
                S = idx; K = idx * (1.0 - 0.15); T = leaps_tenor
                _, leaps_delta, eff_lev = _bs_call_price(S, K, T, risk_free, ann_s)
                leaps_val        = cash
                cash             = 0.0
                leaps_active     = True
                days_since_entry = 0

            # ── MONTHLY DCA ───────────────────────────────────────────────
            if i > 0 and i % 21 == 0:
                voo_b  += monthly
                spxl_b += monthly
                if leaps_active:
                    # DCA directly into existing position
                    leaps_val += monthly
                else:
                    cash += monthly

            total = leaps_val + cash
            all_leaps[i+1, p]  = total
            all_voo[i+1, p]    = voo_b
            all_spxl[i+1, p]   = spxl_b
            all_active[i+1, p] = 1.0 if leaps_active else 0.0

    return {
        "hybrid_p10":   np.percentile(all_leaps, 10, axis=1),
        "hybrid_p25":   np.percentile(all_leaps, 25, axis=1),
        "hybrid_p50":   np.percentile(all_leaps, 50, axis=1),
        "hybrid_p75":   np.percentile(all_leaps, 75, axis=1),
        "hybrid_p90":   np.percentile(all_leaps, 90, axis=1),
        "voo_p50":      np.percentile(all_voo,   50, axis=1),
        "spxl_p50":     np.percentile(all_spxl,  50, axis=1),
        "leaps_active": np.mean(all_active,           axis=1),
    }


def run_leaps_historical(
    prices: pd.DataFrame,
    monthly: float,
    initial: float       = 0,
    voo_core_pct: float  = 0.0,   # ignored
    leaps_pct: float     = 1.0,   # ignored
    trigger_dd: float    = 0.10,
    delta_target: float  = 0.875,
    leaps_tenor: float   = 1.75,
    roll_months: int     = 9,
    roll_cost_pct: float = 0.01,
    theta_mo_pct: float  = 0.005,
    risk_free: float     = 0.04,
) -> pd.DataFrame:
    """Pure LEAPS DCA historical backtest on real VOO prices."""
    voo_prices  = prices["VOO"].values
    spxl_prices = prices["SPXL"].values
    dates       = prices.index
    n           = len(dates)

    log_rets_init = np.diff(np.log(np.maximum(voo_prices[:61], 1e-6)))
    hist_vol = float(np.std(log_rets_init) * np.sqrt(252)) if len(log_rets_init) > 5 else 0.18

    cash         = float(initial)
    leaps_val    = 0.0
    leaps_active = False
    leaps_delta  = delta_target
    days_since_entry = 999

    voo_ath = None; lm = -1; invested = initial
    voo_shares_pure  = (initial / max(voo_prices[0], 1)) if initial > 0 else 0.0
    spxl_shares_pure = (initial / max(spxl_prices[0], 1)) if initial > 0 else 0.0

    out_leaps    = np.zeros(n)
    out_alloc    = np.zeros(n)
    out_invested = np.zeros(n)
    out_voo_pure = np.zeros(n)
    out_spxl_pure= np.zeros(n)
    out_active   = np.zeros(n)

    for i, date in enumerate(dates):
        vp = voo_prices[i]; sp = spxl_prices[i]
        if vp <= 0 or sp <= 0:
            out_leaps[i]    = leaps_val + cash
            out_invested[i] = invested
            continue

        # daily P&L
        if i > 0 and voo_prices[i-1] > 0:
            r_idx = np.log(vp / voo_prices[i-1])
            if leaps_active:
                theta_d   = theta_mo_pct / 21.0
                leaps_val = max(leaps_val * (np.exp(leaps_delta * r_idx) - theta_d), 0.0)
            else:
                cash *= np.exp(risk_free / 252)

        voo_ath      = max(voo_ath, vp) if voo_ath else vp
        dd           = max(0.0, (voo_ath - vp) / voo_ath)
        at_ath       = vp >= voo_ath * 0.999
        days_since_entry += 1

        # PROFIT LOCK
        if leaps_active and at_ath and days_since_entry > 10:
            cash        += leaps_val
            leaps_val    = 0.0
            leaps_active = False

        # ROLL
        if leaps_active and i > 0 and i % (roll_months * 21) == 0:
            leaps_val *= (1.0 - roll_cost_pct)

        # ENTRY
        if not leaps_active and dd >= trigger_dd and cash > 0:
            try:
                _, leaps_delta, _ = _bs_call_price(vp, vp * 0.85, leaps_tenor, risk_free, hist_vol)
            except Exception:
                leaps_delta = delta_target
            leaps_val        = cash
            cash             = 0.0
            leaps_active     = True
            days_since_entry = 0

        # MONTHLY DCA
        if date.month != lm:
            invested += monthly
            if vp > 0: voo_shares_pure  += monthly / vp
            if sp > 0: spxl_shares_pure += monthly / sp
            if leaps_active:
                leaps_val += monthly
            else:
                cash += monthly
            lm = date.month

        total = leaps_val + cash
        out_leaps[i]    = total
        out_alloc[i]    = leaps_val / max(total, 1)
        out_invested[i] = invested
        out_voo_pure[i] = voo_shares_pure * vp
        out_spxl_pure[i]= spxl_shares_pure * sp
        out_active[i]   = 1.0 if leaps_active else 0.0

    return pd.DataFrame({
        "Strategy":    np.round(out_leaps,     2),
        "LEAPS_alloc": np.round(out_alloc * 100, 1),
        "LEAPS_active":out_active,
        "Invested":    np.round(out_invested,  2),
        "VOO_pure":    np.round(out_voo_pure,  2),
        "SPXL_pure":   np.round(out_spxl_pure, 2),
    }, index=dates)

# =============================================================================
# MAIN
# =============================================================================
def main():
    st.title("VOO vs SPXL — Springboard P200 + Volatility Edge")
    st.caption(
        f"${monthly_inv}/month · {proj_years}yr projection · "
        f"{ann_ret*100:.1f}% return · {ann_vol*100:.1f}% vol · {leverage:.1f}x leverage"
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Historical Backtest",
        "🔮 Forward Projection",
        "🚀 Springboard P200",
        "📐 The Math",
        "🔥 Omega LEAPS Hybrid",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: HISTORICAL BACKTEST
    # -------------------------------------------------------------------------
    with tab1:
        st.markdown("Real DCA into VOO and SPXL using actual prices from Jan 2010.")
        with st.spinner("Loading historical prices from Yahoo Finance..."):
            prices = load_historical()

        if prices.empty:
            st.error(
                "Could not load historical price data from Yahoo Finance. "
                "This can happen on Streamlit Cloud due to network restrictions or yfinance rate limits. "
                "The **Forward Projection** and **Springboard** tabs use pure simulation and work without live data."
            )
        else:
            hist = run_historical_dca(prices, monthly_inv, initial=initial_inv)
            if hist.empty or len(hist) == 0:
                st.error("Historical DCA returned no rows — price data may be malformed. Try refreshing.")
            else:
                signals = generate_signals(prices)
                voo_final = hist["VOO"].iloc[-1]
                spxl_final = hist["SPXL"].iloc[-1]
                invested_final = hist["Invested"].iloc[-1]

                k = st.columns(5)
                kpi(k[0], "VOO Final Value", fmt_currency(voo_final), f"{voo_final/invested_final:.2f}x invested", "#00ff9f")
                kpi(k[1], "SPXL Final Value", fmt_currency(spxl_final), f"{spxl_final/invested_final:.2f}x invested", "#00d4ff")
                kpi(k[2], "SPXL vs VOO", f"{spxl_final/voo_final:.2f}x", "leverage edge", "#ffcc00")
                kpi(k[3], "Total Invested", fmt_currency(invested_final), "per instrument", "#aaaaaa")
                kpi(k[4], "Vol Drag (SPXL)", f"{vol_drag(leverage, ann_vol)*100:.2f}%/yr", "Ito annual drag", "#ff4b4b")

                st.markdown("")

                buy_signals = signals[signals["Type"] == "BUY"]
                sell_signals = signals[signals["Type"] == "SELL"]

                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    row_heights=[0.65, 0.35],
                    vertical_spacing=0.04,
                )

                fig.add_trace(go.Scatter(x=hist.index, y=hist["VOO"], name="VOO", line=dict(color="#00ff9f", width=2)), row=1, col=1)
                fig.add_trace(go.Scatter(x=hist.index, y=hist["SPXL"], name="SPXL", line=dict(color="#00d4ff", width=2)), row=1, col=1)
                fig.add_trace(go.Scatter(x=hist.index, y=hist["Invested"], name="Invested", line=dict(color="#555", width=1, dash="dash")), row=1, col=1)

                fig.add_trace(go.Scatter(
                    x=hist.index, y=hist["SPXL"], name="SPXL Price", line=dict(color="#00d4ff", width=1.5)
                ), row=2, col=1)
                ma200 = hist["SPXL"].rolling(200, min_periods=1).mean()
                fig.add_trace(go.Scatter(
                    x=hist.index, y=ma200, name="MA200", line=dict(color="#f5a623", width=1, dash="dot")
                ), row=2, col=1)
                fig.add_trace(go.Scatter(
                    x=buy_signals.index, y=buy_signals["Price"],
                    mode="markers", name="BUY",
                    marker=dict(symbol="triangle-up", size=10, color="#00ff9f")
                ), row=2, col=1)
                fig.add_trace(go.Scatter(
                    x=sell_signals.index, y=sell_signals["Price"],
                    mode="markers", name="SELL",
                    marker=dict(symbol="triangle-down", size=10, color="#ff4b4b")
                ), row=2, col=1)

                fig.update_layout(
                    template="plotly_dark", height=520,
                    paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=10, r=10, t=20, b=10),
                )
                fig.update_yaxes(tickprefix="$", row=1, col=1)
                fig.update_yaxes(tickprefix="$", row=2, col=1)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### Drawdown")
                dd_fig = go.Figure()
                dd_fig.add_trace(go.Scatter(x=hist.index, y=compute_drawdown(hist["VOO"]), fill="tozeroy", name="VOO DD", line=dict(color="#00ff9f", width=1)))
                dd_fig.add_trace(go.Scatter(x=hist.index, y=compute_drawdown(hist["SPXL"]), fill="tozeroy", name="SPXL DD", line=dict(color="#ff4b4b", width=1)))
                dd_fig.update_layout(template="plotly_dark", height=200, paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14", margin=dict(l=10, r=10, t=10, b=10))
                dd_fig.update_yaxes(ticksuffix="%")
                st.plotly_chart(dd_fig, use_container_width=True)

                st.markdown("---")
                st.markdown("### ATH Rotation strategy — real historical backtest")
                st.caption("Same prices, different strategy: shift VOO value to SPXL during drawdowns, reduce SPXL when above ATH.")

                rot_c1, rot_c2 = st.columns(2)
                with rot_c1:
                    rot_base = st.slider("SPXL % at ATH (base)", 0, 100, 10, 5, key="rot_base",
                        help="Allocation when market is at all-time high.")
                with rot_c2:
                    rot_dd = st.slider("Drawdown % → 100% SPXL", 15, 60, 30, 5, key="rot_dd",
                        help="How far below ATH before you reach 100% SPXL.")

                # Build a simple 4-tier table from the sliders
                rot_tiers = [
                    (0.00, rot_base),
                    (0.10, min(100, rot_base + (100 - rot_base) * (0.10 / (rot_dd/100)))),
                    (0.20, min(100, rot_base + (100 - rot_base) * (0.20 / (rot_dd/100)))),
                    (rot_dd / 100, 100),
                ]
                rot_tiers = [(dd, int(round(pct))) for dd, pct in rot_tiers]
                rot = run_ath_rotation_backtest(
                    prices, monthly_inv,
                    tiers=rot_tiers,
                    rebal_speed=0.30,
                    initial=initial_inv,
                )

                rot_final    = float(rot["Strategy"].iloc[-1])
                voo_pur_fin  = float(rot["VOO_pure"].iloc[-1])
                spxl_pur_fin = float(rot["SPXL_pure"].iloc[-1])
                rot_inv      = float(rot["Invested"].iloc[-1])

                rk = st.columns(4)
                kpi(rk[0], "ATH Rotation final", fmt_currency(rot_final),    f"{rot_final/rot_inv:.2f}x invested", "#00ff9f")
                kpi(rk[1], "vs pure VOO DCA",    f"{rot_final/voo_pur_fin:.2f}x", fmt_currency(voo_pur_fin),      "#00ffcc")
                kpi(rk[2], "vs pure SPXL DCA",   f"{rot_final/spxl_pur_fin:.2f}x", fmt_currency(spxl_pur_fin),   "#f5a623")
                kpi(rk[3], "Total invested",      fmt_currency(rot_inv),      "single instrument",                 "#aaaaaa")

                fig_rot = go.Figure()
                fig_rot.add_trace(go.Scatter(x=rot.index, y=rot["Strategy"],
                    name="ATH Rotation", line=dict(color="#00ff9f", width=2.5)))
                fig_rot.add_trace(go.Scatter(x=rot.index, y=rot["SPXL_pure"],
                    name="Pure SPXL DCA", line=dict(color="#00d4ff", width=1.5, dash="dash")))
                fig_rot.add_trace(go.Scatter(x=rot.index, y=rot["VOO_pure"],
                    name="Pure VOO DCA", line=dict(color="#888", width=1.5, dash="dot")))
                fig_rot.add_trace(go.Scatter(x=rot.index, y=rot["Invested"],
                    name="Invested", line=dict(color="#444", width=1, dash="dash")))
                fig_rot.update_layout(
                    template="plotly_dark", height=340,
                    paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
                    margin=dict(l=10, r=10, t=10, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                fig_rot.update_yaxes(tickprefix="$")
                st.plotly_chart(fig_rot, use_container_width=True)

                # SPXL allocation % over time
                st.markdown("##### SPXL allocation % — actual history")
                fig_ra = go.Figure()
                fig_ra.add_trace(go.Scatter(
                    x=rot.index, y=rot["SPXL_alloc"],
                    fill="tozeroy", fillcolor="rgba(0,212,255,0.12)",
                    line=dict(color="#00d4ff", width=1.2),
                ))
                fig_ra.add_hline(y=rot_base,  line=dict(color="#00ff9f", width=1, dash="dot"),
                                 annotation_text=f"ATH base {rot_base}%", annotation_position="right")
                fig_ra.add_hline(y=100, line=dict(color="#ff4b4b", width=1, dash="dot"),
                                 annotation_text="100% SPXL", annotation_position="right")
                fig_ra.update_layout(
                    template="plotly_dark", height=150,
                    paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
                    margin=dict(l=10, r=10, t=10, b=10),
                    yaxis=dict(range=[0, 105], ticksuffix="%"),
                    showlegend=False,
                )
                st.plotly_chart(fig_ra, use_container_width=True)
                st.caption("Watch the allocation spike in Mar 2020 (-34%), Dec 2018 (-20%), and Oct 2022 (-25%) — those are the moments this strategy loads heaviest into SPXL for the recovery.")

    # -------------------------------------------------------------------------
    # TAB 2: FORWARD PROJECTION (MONTE CARLO)
    # -------------------------------------------------------------------------
    with tab2:
        st.markdown("GBM Monte Carlo with Ito drag — P10 / P50 / P90 fan chart.")

        with st.spinner(f"Simulating {n_sim} paths × 2 instruments..."):
            voo_paths, voo_invested = dca_gbm_paths(proj_years, monthly_inv, ann_ret, ann_vol, voo_expense, 1, n_sim, seed=1, initial=initial_inv)
            spxl_paths, spxl_invested = dca_gbm_paths(proj_years, monthly_inv, ann_ret, ann_vol, spxl_expense, leverage, n_sim, seed=2, initial=initial_inv)

        voo_p50 = np.percentile(voo_paths, 50, axis=1)
        spxl_p10 = np.percentile(spxl_paths, 10, axis=1)
        spxl_p50 = np.percentile(spxl_paths, 50, axis=1)
        spxl_p90 = np.percentile(spxl_paths, 90, axis=1)

        total_inv = monthly_inv * (proj_years * 252 // 21)
        vf = voo_p50[-1]; sf = spxl_p50[-1]; s10 = spxl_p10[-1]; s90 = spxl_p90[-1]

        k = st.columns(5)
        kpi(k[0], "VOO P50", fmt_currency(vf), f"{vf/total_inv:.2f}x invested", "#00ff9f")
        kpi(k[1], "SPXL P50", fmt_currency(sf), f"{sf/total_inv:.2f}x invested", "#00d4ff")
        kpi(k[2], "P10 Downside", fmt_currency(s10), "worst 10%", "#f5a623")
        kpi(k[3], "P90 Upside", fmt_currency(s90), "best 10%", "#00ff9f")
        kpi(k[4], "Vol Drag", f"{vol_drag(leverage, ann_vol)*100:.2f}%/yr", "Ito annual", "#ff4b4b")

        days_total = proj_years * 252 + 1
        x_axis = np.linspace(0, proj_years, days_total)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x_axis, y=spxl_p90, fill=None, line=dict(width=0), showlegend=False))
        fig2.add_trace(go.Scatter(x=x_axis, y=spxl_p10, fill="tonexty", fillcolor="rgba(0,212,255,0.12)", line=dict(width=0), name="SPXL P10–P90"))
        fig2.add_trace(go.Scatter(x=x_axis, y=spxl_p50, name="SPXL P50", line=dict(color="#00d4ff", width=2.5)))
        fig2.add_trace(go.Scatter(x=x_axis, y=voo_p50, name="VOO P50", line=dict(color="#00ff9f", width=2, dash="dash")))
        fig2.update_layout(template="plotly_dark", height=420, paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14", margin=dict(l=10, r=10, t=10, b=10))
        fig2.update_xaxes(title_text="Year")
        fig2.update_yaxes(tickprefix="$")
        st.plotly_chart(fig2, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 3: SPRINGBOARD — Opportunistic Predator + Profit Lock
    # -------------------------------------------------------------------------
    with tab3:
        st.markdown(
            "**The Opportunistic Predator** · Base state: 100% VOO (zero vol decay). "
            "Scale into SPXL in tiers as market drops from its 50-day high. "
            "**Profit Lock:** when market hits a new ATH, exit all SPXL → back to 100% VOO. Rinse and repeat."
        )

        st.markdown("##### Tier table — SPXL % at each drop from 50-day high")
        tier_cols = st.columns(4)
        tier_labels   = ["At 50d high (0%)", "Drop -10%", "Drop -20%", "Drop -30%+"]
        tier_dd       = [0.00, 0.10, 0.20, 0.30]
        tier_defaults = [0, 20, 52, 100]
        tier_helps    = [
            "Base state — 100% VOO, zero vol drag",
            "Scout — first SPXL kicker at -10%",
            "Aggressive — meaningful leverage at -20%",
            "All-in — 100% SPXL coiled spring at -30%",
        ]
        tier_vals = []
        for i, col in enumerate(tier_cols):
            with col:
                v = st.slider(tier_labels[i], 0, 100, tier_defaults[i], 1,
                              help=tier_helps[i], key=f"tier_{i}")
                tier_vals.append(v)
                st.caption(f"VOO {100-v}% · SPXL {v}%")

        tiers_input = list(zip(tier_dd, tier_vals))

        ctrl1, ctrl2, ctrl3 = st.columns(3)
        with ctrl1:
            sb_paths  = st.select_slider("MC paths", options=[100, 200, 300, 500], value=200, key="l_paths")
            rebal_sp  = st.slider("Rebal speed %/month", 10, 60, 30, 10,
                help="% of allocation gap closed each month.") / 100
        with ctrl2:
            wg = st.slider("Whipsaw guard (days)", 0, 30, 5, 1,
                help="Min days in a tier before Profit Lock can fire. Prevents selling SPXL immediately after buying.")
        with ctrl3:
            target_val = st.number_input("Days-to-target ($)", value=1_000_000, step=100_000,
                help="The 'Days to Millionaire' comparison target.")

        # Live allocation preview table
        st.markdown("**Live allocation preview + recovery math:**")
        preview_rows = []
        for dd_pct in [0, 10, 20, 30, 40]:
            spxl_v   = _tier_spxl(dd_pct / 100, tiers_input)
            voo_v    = 100 - spxl_v
            port_bounce_10 = spxl_v / 100 * leverage * 10 + voo_v / 100 * 10
            preview_rows.append({
                "Drop from 50d high": f"-{dd_pct}%",
                "SPXL": f"{spxl_v:.0f}%",
                "VOO":  f"{voo_v:.0f}%",
                "10% bounce → portfolio gain": f"+{port_bounce_10:.1f}%",
            })
        st.dataframe(pd.DataFrame(preview_rows), hide_index=True, use_container_width=True)

        with st.spinner(f"Running {sb_paths} Monte Carlo paths..."):
            sb = run_springboard(
                proj_years, monthly_inv, ann_ret, ann_vol,
                voo_expense, spxl_expense, leverage,
                tiers=tiers_input,
                rebal_speed=rebal_sp,
                n_paths=sb_paths,
                initial=initial_inv,
                whipsaw_guard=wg,
            )

        sp50 = float(sb["spring_p50"][-1])
        vp50 = float(sb["voo_p50"][-1])
        xp50 = float(sb["spxl_p50"][-1])
        total_inv_sb = initial_inv + monthly_inv * (proj_years * 252 // 21)
        cagr_sb   = (sp50 / max(total_inv_sb, 1)) ** (1 / max(proj_years, 1)) - 1
        cagr_voo  = (vp50 / max(total_inv_sb, 1)) ** (1 / max(proj_years, 1)) - 1
        cagr_spxl = (xp50 / max(total_inv_sb, 1)) ** (1 / max(proj_years, 1)) - 1

        # Days-to-target on median path
        dtm_spring = _days_to_target(sb["spring_p50"], target_val)
        dtm_spxl   = _days_to_target(sb["spxl_p50"],   target_val)
        dtm_voo    = _days_to_target(sb["voo_p50"],     target_val)

        def fmt_days(d):
            if d < 0: return "not reached"
            yrs = d / 252
            return f"{yrs:.1f} yrs ({d} days)"

        k = st.columns(5)
        kpi(k[0], "Strategy P50",   fmt_currency(sp50),           f"CAGR {cagr_sb*100:.1f}%",   "#00ff9f")
        kpi(k[1], "vs Pure SPXL",   f"{sp50/max(xp50,1):.2f}x",  f"{fmt_currency(xp50)} · {cagr_spxl*100:.1f}% CAGR", "#00ffcc")
        kpi(k[2], "vs Pure VOO",    f"{sp50/max(vp50,1):.2f}x",  f"{fmt_currency(vp50)} · {cagr_voo*100:.1f}% CAGR",  "#aaaaaa")
        kpi(k[3], "Avg SPXL alloc", f"{sb['avg_spxl']:.0f}%",    "median path",                 "#ff4b4b")
        kpi(k[4], "P10 Downside",   fmt_currency(float(sb["spring_p10"][-1])), "worst 10%",      "#f5a623")

        # Days to target comparison
        st.markdown(f"##### Days to ${target_val:,.0f} (median path)")
        dtm_cols = st.columns(3)
        def dtm_color(d, others):
            if d < 0: return "#f5a623"
            return "#00ff9f" if d == min(x for x in others if x > 0) else "#aaaaaa"
        all_dtm = [dtm_spring, dtm_spxl, dtm_voo]
        kpi(dtm_cols[0], "Springboard strategy", fmt_days(dtm_spring), "this strategy",  dtm_color(dtm_spring, all_dtm))
        kpi(dtm_cols[1], "Pure SPXL DCA",        fmt_days(dtm_spxl),  "glass cannon",   dtm_color(dtm_spxl,   all_dtm))
        kpi(dtm_cols[2], "Pure VOO DCA",          fmt_days(dtm_voo),   "safe but slow",  dtm_color(dtm_voo,    all_dtm))

        x_axis = np.linspace(0, proj_years, len(sb["spring_p50"]))

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=x_axis, y=sb["spring_p90"], fill=None, line=dict(width=0), showlegend=False))
        fig3.add_trace(go.Scatter(x=x_axis, y=sb["spring_p10"], fill="tonexty",
                                  fillcolor="rgba(0,255,159,0.07)", line=dict(width=0), name="Strategy P10–P90"))
        fig3.add_trace(go.Scatter(x=x_axis, y=sb["spring_p25"], fill=None, line=dict(width=0), showlegend=False))
        fig3.add_trace(go.Scatter(x=x_axis, y=sb["spring_p75"], fill="tonexty",
                                  fillcolor="rgba(0,255,159,0.15)", line=dict(width=0), name="Strategy P25–P75"))
        fig3.add_trace(go.Scatter(x=x_axis, y=sb["spring_p50"],
                                  name="Springboard P50", line=dict(color="#00ff9f", width=2.5)))
        fig3.add_trace(go.Scatter(x=x_axis, y=sb["spxl_p50"],
                                  name="Pure SPXL P50", line=dict(color="#00d4ff", width=1.5, dash="dash")))
        fig3.add_trace(go.Scatter(x=x_axis, y=sb["voo_p50"],
                                  name="Pure VOO P50", line=dict(color="#888", width=1.5, dash="dot")))
        if target_val > 0:
            fig3.add_hline(y=target_val, line=dict(color="#f5a623", width=1, dash="dot"),
                           annotation_text=f"${target_val/1e6:.1f}M target", annotation_position="right")
        fig3.update_layout(
            template="plotly_dark", height=360,
            paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig3.update_xaxes(title_text="Year")
        fig3.update_yaxes(tickprefix="$")
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("##### SPXL allocation % — median simulation path")
        alloc_pct = sb["spxl_alloc_p50"] * 100
        fig_alloc = go.Figure()
        fig_alloc.add_trace(go.Scatter(
            x=x_axis, y=alloc_pct, fill="tozeroy",
            fillcolor="rgba(0,212,255,0.12)", line=dict(color="#00d4ff", width=1.5),
        ))
        for dd, pct in tiers_input:
            if pct > 0:
                fig_alloc.add_hline(y=pct, line=dict(color="#333", width=1, dash="dot"),
                    annotation_text=f"T{int(dd*100)}%→{pct}%", annotation_position="right")
        fig_alloc.update_layout(
            template="plotly_dark", height=150,
            paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(range=[0, 105], ticksuffix="%"),
            showlegend=False,
        )
        fig_alloc.update_xaxes(title_text="Year")
        st.plotly_chart(fig_alloc, use_container_width=True)

        # Real historical backtest
        st.markdown("---")
        st.markdown("### Real historical result (2010–today)")
        with st.spinner("Running real backtest on actual prices..."):
            try:
                prices_cached = load_historical()
                if not prices_cached.empty:
                    rot = run_ath_rotation_backtest(
                        prices_cached, monthly_inv,
                        tiers=tiers_input,
                        rebal_speed=rebal_sp,
                        initial=initial_inv,
                        whipsaw_guard=wg,
                    )
                    rot_final    = float(rot["Strategy"].iloc[-1])
                    voo_pur_fin  = float(rot["VOO_pure"].iloc[-1])
                    spxl_pur_fin = float(rot["SPXL_pure"].iloc[-1])
                    rot_inv      = float(rot["Invested"].iloc[-1])

                    # Days to target on real data
                    dtm_r_spring = _days_to_target(rot["Strategy"].values,  target_val)
                    dtm_r_spxl   = _days_to_target(rot["SPXL_pure"].values, target_val)
                    dtm_r_voo    = _days_to_target(rot["VOO_pure"].values,   target_val)

                    rk = st.columns(4)
                    kpi(rk[0], "Strategy final",   fmt_currency(rot_final),                          f"{rot_final/max(rot_inv,1):.2f}x invested",   "#00ff9f")
                    kpi(rk[1], "vs pure SPXL DCA", f"{rot_final/max(spxl_pur_fin,1):.2f}x",         fmt_currency(spxl_pur_fin),                    "#00ffcc")
                    kpi(rk[2], "vs pure VOO DCA",  f"{rot_final/max(voo_pur_fin,1):.2f}x",          fmt_currency(voo_pur_fin),                     "#aaaaaa")
                    kpi(rk[3], "Total invested",   fmt_currency(rot_inv),                            "single stream",                               "#444")

                    if any(d > 0 for d in [dtm_r_spring, dtm_r_spxl, dtm_r_voo]):
                        st.markdown(f"**Real days to ${target_val:,.0f}:**")
                        dc = st.columns(3)
                        all_r = [dtm_r_spring, dtm_r_spxl, dtm_r_voo]
                        kpi(dc[0], "Springboard",  fmt_days(dtm_r_spring), "real history", dtm_color(dtm_r_spring, all_r))
                        kpi(dc[1], "Pure SPXL DCA",fmt_days(dtm_r_spxl),  "real history", dtm_color(dtm_r_spxl,   all_r))
                        kpi(dc[2], "Pure VOO DCA", fmt_days(dtm_r_voo),   "real history", dtm_color(dtm_r_voo,    all_r))

                    fig_rot = go.Figure()
                    fig_rot.add_trace(go.Scatter(x=rot.index, y=rot["Strategy"],
                        name="Springboard", line=dict(color="#00ff9f", width=2.5)))
                    fig_rot.add_trace(go.Scatter(x=rot.index, y=rot["SPXL_pure"],
                        name="Pure SPXL DCA", line=dict(color="#00d4ff", width=1.5, dash="dash")))
                    fig_rot.add_trace(go.Scatter(x=rot.index, y=rot["VOO_pure"],
                        name="Pure VOO DCA", line=dict(color="#888", width=1.5, dash="dot")))
                    fig_rot.add_trace(go.Scatter(x=rot.index, y=rot["Invested"],
                        name="Invested", line=dict(color="#333", width=1, dash="dash")))

                # ── LEAPS overlay on historical chart ──────────────────────────
                try:
                    leaps_hist = run_leaps_historical(
                        prices, monthly_inv, initial=initial_inv,
                        voo_core_pct=0.80, leaps_pct=0.20,
                        trigger_dd=0.10, theta_mo_pct=0.005,
                    )
                    fig_rot.add_trace(go.Scatter(
                        x=leaps_hist.index, y=leaps_hist["Strategy"],
                        name="Omega LEAPS Hybrid", line=dict(color="#ff9f00", width=2, dash="dot"),
                    ))
                except Exception:
                    pass
                    if target_val > 0:
                        fig_rot.add_hline(y=target_val, line=dict(color="#f5a623", width=1, dash="dot"),
                            annotation_text=f"${target_val/1e6:.1f}M", annotation_position="right")
                    fig_rot.update_layout(
                        template="plotly_dark", height=320,
                        paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
                        margin=dict(l=10, r=10, t=10, b=10),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    fig_rot.update_yaxes(tickprefix="$")
                    st.plotly_chart(fig_rot, use_container_width=True)

                    st.markdown("##### SPXL allocation % — real history")
                    fig_ra = go.Figure()
                    fig_ra.add_trace(go.Scatter(
                        x=rot.index, y=rot["SPXL_alloc"],
                        fill="tozeroy", fillcolor="rgba(0,212,255,0.12)",
                        line=dict(color="#00d4ff", width=1.2),
                    ))
                    fig_ra.update_layout(
                        template="plotly_dark", height=130,
                        paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
                        margin=dict(l=10, r=10, t=10, b=10),
                        yaxis=dict(range=[0, 105], ticksuffix="%"),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_ra, use_container_width=True)
                    st.caption(
                        "Allocation spikes: Mar 2020 (-34%), Q4 2018 (-20%), 2022 bear (-25%). "
                        "Each spike is followed by a Profit Lock flush back to 0% SPXL when the new ATH is hit. "
                        "That's the cycle working exactly as designed."
                    )
                else:
                    st.info("Historical data unavailable — run locally or on Streamlit Cloud with network access.")
            except Exception as e:
                st.warning(f"Historical backtest skipped: {e}")
    # -------------------------------------------------------------------------
    # TAB 4: THE MATH
    # -------------------------------------------------------------------------
    with tab4:
        drag = vol_drag(leverage, ann_vol)
        net_spxl = net_annual(leverage, ann_ret, ann_vol, spxl_expense)
        net_voo = net_annual(1, ann_ret, ann_vol, voo_expense)

        st.markdown("### Mathematical Foundation")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Ito's Lemma — Vol Drag")
            st.latex(r"\text{Vol Drag} = \frac{L(L-1)}{2} \sigma^2")
            st.markdown(f"With L={leverage:.1f}x, σ={ann_vol*100:.1f}%: **drag = {drag*100:.2f}%/yr**")

            st.markdown("#### Net SPXL Annual Return")
            st.latex(r"r_{SPXL} = L \cdot r_{SPX} - \frac{L(L-1)}{2}\sigma^2 - \text{expense}")
            st.markdown(f"= {leverage:.1f} × {ann_ret*100:.1f}% − {drag*100:.2f}% − {spxl_expense*100:.2f}% = **{net_spxl*100:.2f}%/yr**")

            st.markdown("#### GBM Price Path")
            st.latex(r"S_{t+dt} = S_t \cdot e^{\left(\mu - \frac{\sigma^2}{2}\right)dt + \sigma\sqrt{dt}\,Z}")

        with col2:
            st.markdown("#### Springboard Signal Logic")
            st.code("""
# Enter SPXL when ALL conditions met:
price > MA(200)          # trend filter
realized_vol < threshold # volatility gate
momentum(10d) > 0        # momentum gate

# OR re-enter on pullback:
price < peak * (1 - pullback_pct)

# DCA multiplier active in SPXL:
contrib = monthly * dca_multiplier

# Exit SPXL when signal breaks:
if not core_signal → switch to VOO
            """, language="python")

            st.markdown("#### Risk: Drawdown Estimates")
            dd_data = {
                "S&P Move": ["-10%", "-20%", "-30%", "-50%"],
                f"SPXL ({leverage:.1f}x)": [
                    f"~{min(100, leverage * 10):.0f}%",
                    f"~{min(100, leverage * 20 + 3):.0f}%",
                    f"~{min(100, leverage * 30 + 8):.0f}%",
                    f"~{min(100, leverage * 50 + 15):.0f}%",
                ],
                "Risk Level": ["Moderate", "Severe", "Severe", "Catastrophic"],
            }
            st.dataframe(pd.DataFrame(dd_data), hide_index=True, use_container_width=True)

            st.markdown("#### Break-even Volatility")
            st.latex(r"\sigma_{break} = \sqrt{\frac{(L-1) \cdot r_{SPX}}{L(L-1)/2}} \approx 24\%")
            st.markdown("Above ~24% ann. vol, vol drag erodes the leverage benefit.")


    # -------------------------------------------------------------------------
    # TAB 5: OMEGA LEAPS — Pure Deep-ITM Calls, No VOO
    # -------------------------------------------------------------------------
    with tab5:
        st.markdown(
            "**Omega LEAPS — Pure Deep-ITM Calls.** "
            "100% of capital deployed into deep-ITM SPY/SPLG calls at every entry signal. "
            "Cash earns MMF yield (~4%) when not deployed. "
            "**Profit Lock:** sell everything on new ATH → back to MMF. Roll every 9 months."
        )

        lc1, lc2, lc3 = st.columns(3)
        with lc1:
            st.markdown("##### Entry & exit")
            trigger_s = st.slider("Entry trigger (% below ATH)", 5, 25, 10, 1, key="l_trigger",
                help="Deploy 100% of cash into LEAPS when market drops this far from ATH.")
            delta_s   = st.slider("Target delta", 75, 95, 88, 1, key="l_delta",
                help="0.85-0.90 = deep ITM. Higher = less theta, more intrinsic.") / 100
            tenor_s   = st.slider("LEAPS tenor (years)", 1.0, 2.5, 1.75, 0.25, key="l_tenor",
                help="Time to expiry on purchase. 1.5-2yr is standard.")

        with lc2:
            st.markdown("##### Cost model")
            theta_s    = st.slider("Theta drag (%/month)", 0.2, 1.0, 0.5, 0.1, key="l_theta",
                help="Monthly decay cost while active. Deep ITM keeps this very low.") / 100
            roll_mo_s  = st.slider("Roll every N months", 6, 12, 9, 1, key="l_roll_mo",
                help="Extend to next 2-year cycle every N months. ~1% transaction cost.")
            roll_cost_s= st.slider("Roll cost %", 0.5, 2.0, 1.0, 0.1, key="l_roll_cost") / 100
            mmf_rate   = st.slider("MMF yield % (idle cash)", 2.0, 6.0, 4.0, 0.5, key="l_mmf",
                help="What your cash earns while waiting for the next entry signal.") / 100

        with lc3:
            st.markdown("##### Simulation")
            leaps_paths = st.select_slider("MC paths", options=[100, 200, 300, 500], value=200, key="l_paths")
            st.markdown("")
            st.markdown("**How it works:**")
            st.caption(f"1. Cash sits in MMF at {mmf_rate*100:.1f}%/yr")
            st.caption(f"2. At -{trigger_s}% from ATH → 100% into δ={delta_s:.2f} LEAPS")
            st.caption(f"3. Daily P&L: δ × index return − {theta_s*100:.2f}%/day theta")
            st.caption(f"4. New ATH hit → sell all → back to MMF")
            st.caption(f"5. Roll every {roll_mo_s}mo at {roll_cost_s*100:.1f}% cost")

        with st.spinner(f"Running {leaps_paths} pure LEAPS paths..."):
            lb = run_leaps_hybrid(
                proj_years, monthly_inv, ann_ret, ann_vol, voo_expense,
                trigger_dd    = trigger_s,
                delta_target  = delta_s,
                leaps_tenor   = tenor_s,
                roll_months   = roll_mo_s,
                roll_cost_pct = roll_cost_s,
                theta_mo_pct  = theta_s,
                risk_free     = mmf_rate,
                n_paths       = leaps_paths,
                initial       = initial_inv,
            )

        lp50  = float(lb["hybrid_p50"][-1])
        vp50l = float(lb["voo_p50"][-1])
        xp50l = float(lb["spxl_p50"][-1])
        total_inv_l   = initial_inv + monthly_inv * (proj_years * 252 // 21)
        cagr_l    = (lp50  / max(total_inv_l, 1)) ** (1 / max(proj_years, 1)) - 1
        cagr_vool = (vp50l / max(total_inv_l, 1)) ** (1 / max(proj_years, 1)) - 1
        cagr_xl   = (xp50l / max(total_inv_l, 1)) ** (1 / max(proj_years, 1)) - 1
        leaps_pct_time = float(np.mean(lb["leaps_active"]) * 100)

        dtm_l  = _days_to_target(lb["hybrid_p50"], 1_000_000)
        dtm_lx = _days_to_target(lb["spxl_p50"],   1_000_000)
        dtm_lv = _days_to_target(lb["voo_p50"],    1_000_000)

        def fmt_d(d):
            return "not reached" if d < 0 else f"{d/252:.1f} yrs"

        def dtm_col(d, others):
            valid = [x for x in others if x > 0]
            if d < 0: return "#f5a623"
            return "#ff9f00" if (valid and d == min(valid)) else "#aaaaaa"

        k = st.columns(5)
        kpi(k[0], "Pure LEAPS P50",   fmt_currency(lp50),           f"CAGR {cagr_l*100:.1f}%",         "#ff9f00")
        kpi(k[1], "vs Pure SPXL",     f"{lp50/max(xp50l,1):.2f}x", f"SPXL {cagr_xl*100:.1f}% CAGR",   "#00d4ff")
        kpi(k[2], "vs Pure VOO",      f"{lp50/max(vp50l,1):.2f}x", f"VOO {cagr_vool*100:.1f}% CAGR",   "#888")
        kpi(k[3], "LEAPS active",     f"{leaps_pct_time:.0f}%",     "of trading days",                  "#ff4b4b")
        kpi(k[4], "Days to $1M",      fmt_d(dtm_l),                 f"SPXL: {fmt_d(dtm_lx)}",           dtm_col(dtm_l,[dtm_l,dtm_lx,dtm_lv]))

        # Days to $1M comparison strip
        dc = st.columns(3)
        all_dtm = [dtm_l, dtm_lx, dtm_lv]
        kpi(dc[0], "LEAPS $1M",    fmt_d(dtm_l),  "pure LEAPS",   dtm_col(dtm_l,  all_dtm))
        kpi(dc[1], "SPXL $1M",     fmt_d(dtm_lx), "pure SPXL",    dtm_col(dtm_lx, all_dtm))
        kpi(dc[2], "VOO $1M",      fmt_d(dtm_lv), "pure VOO",     dtm_col(dtm_lv, all_dtm))

        x_axis_l = np.linspace(0, proj_years, len(lb["hybrid_p50"]))

        fig_l = go.Figure()
        fig_l.add_trace(go.Scatter(x=x_axis_l, y=lb["hybrid_p90"], fill=None, line=dict(width=0), showlegend=False))
        fig_l.add_trace(go.Scatter(x=x_axis_l, y=lb["hybrid_p10"], fill="tonexty",
            fillcolor="rgba(255,159,0,0.07)", line=dict(width=0), name="P10-P90"))
        fig_l.add_trace(go.Scatter(x=x_axis_l, y=lb["hybrid_p25"], fill=None, line=dict(width=0), showlegend=False))
        fig_l.add_trace(go.Scatter(x=x_axis_l, y=lb["hybrid_p75"], fill="tonexty",
            fillcolor="rgba(255,159,0,0.15)", line=dict(width=0), name="P25-P75"))
        fig_l.add_trace(go.Scatter(x=x_axis_l, y=lb["hybrid_p50"],
            name="Pure LEAPS P50", line=dict(color="#ff9f00", width=2.5)))
        fig_l.add_trace(go.Scatter(x=x_axis_l, y=lb["spxl_p50"],
            name="Pure SPXL P50", line=dict(color="#00d4ff", width=1.5, dash="dash")))
        fig_l.add_trace(go.Scatter(x=x_axis_l, y=lb["voo_p50"],
            name="Pure VOO P50", line=dict(color="#888", width=1.5, dash="dot")))
        fig_l.add_hline(y=1_000_000, line=dict(color="#f5a623", width=1, dash="dot"),
            annotation_text="$1M", annotation_position="right")
        fig_l.update_layout(
            template="plotly_dark", height=380,
            paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_l.update_xaxes(title_text="Year")
        fig_l.update_yaxes(tickprefix="$")
        st.plotly_chart(fig_l, use_container_width=True)

        st.markdown("##### LEAPS active % over time (fraction of paths in-position)")
        fig_la = go.Figure()
        fig_la.add_trace(go.Scatter(
            x=x_axis_l, y=lb["leaps_active"] * 100,
            fill="tozeroy", fillcolor="rgba(255,159,0,0.15)",
            line=dict(color="#ff9f00", width=1.5),
        ))
        fig_la.add_hline(y=trigger_s, line=dict(color="#444", width=1, dash="dot"),
            annotation_text=f"entry at -{trigger_s}%", annotation_position="right")
        fig_la.update_layout(
            template="plotly_dark", height=130,
            paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(range=[0, 105], ticksuffix="%"),
            showlegend=False,
        )
        fig_la.update_xaxes(title_text="Year")
        st.plotly_chart(fig_la, use_container_width=True)

        # Real historical backtest
        st.markdown("---")
        st.markdown("### Real historical LEAPS backtest (2010–today)")
        with st.spinner("Running pure LEAPS on real prices..."):
            try:
                prices_l = load_historical()
                if not prices_l.empty:
                    lh = run_leaps_historical(
                        prices_l, monthly_inv,
                        initial       = initial_inv,
                        trigger_dd    = trigger_s,
                        delta_target  = delta_s,
                        leaps_tenor   = tenor_s,
                        roll_months   = roll_mo_s,
                        roll_cost_pct = roll_cost_s,
                        theta_mo_pct  = theta_s,
                        risk_free     = mmf_rate,
                    )
                    lh_final    = float(lh["Strategy"].iloc[-1])
                    voo_lh_fin  = float(lh["VOO_pure"].iloc[-1])
                    spxl_lh_fin = float(lh["SPXL_pure"].iloc[-1])
                    lh_inv      = float(lh["Invested"].iloc[-1])
                    leaps_active_hist = float(lh["LEAPS_active"].mean() * 100)

                    rk = st.columns(4)
                    kpi(rk[0], "LEAPS final",       fmt_currency(lh_final),                        f"{lh_final/max(lh_inv,1):.2f}x invested",  "#ff9f00")
                    kpi(rk[1], "vs pure SPXL DCA",  f"{lh_final/max(spxl_lh_fin,1):.2f}x",        fmt_currency(spxl_lh_fin),                  "#00d4ff")
                    kpi(rk[2], "vs pure VOO DCA",   f"{lh_final/max(voo_lh_fin,1):.2f}x",         fmt_currency(voo_lh_fin),                   "#888")
                    kpi(rk[3], "LEAPS active hist", f"{leaps_active_hist:.0f}%",                   "of real trading days",                     "#ff4b4b")

                    fig_lh = go.Figure()
                    fig_lh.add_trace(go.Scatter(x=lh.index, y=lh["Strategy"],
                        name="Pure LEAPS", line=dict(color="#ff9f00", width=2.5)))
                    fig_lh.add_trace(go.Scatter(x=lh.index, y=lh["SPXL_pure"],
                        name="Pure SPXL DCA", line=dict(color="#00d4ff", width=1.5, dash="dash")))
                    fig_lh.add_trace(go.Scatter(x=lh.index, y=lh["VOO_pure"],
                        name="Pure VOO DCA", line=dict(color="#888", width=1.5, dash="dot")))
                    fig_lh.add_trace(go.Scatter(x=lh.index, y=lh["Invested"],
                        name="Invested", line=dict(color="#333", width=1, dash="dash")))
                    # shade active LEAPS periods
                    active = lh["LEAPS_active"].values
                    in_a = False; t0 = None
                    for ii, dt_ in enumerate(lh.index):
                        if active[ii] and not in_a:
                            t0 = dt_; in_a = True
                        elif not active[ii] and in_a:
                            fig_lh.add_vrect(x0=t0, x1=dt_,
                                fillcolor="rgba(255,159,0,0.09)", line_width=0)
                            in_a = False
                    fig_lh.update_layout(
                        template="plotly_dark", height=320,
                        paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
                        margin=dict(l=10, r=10, t=10, b=10),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    fig_lh.update_yaxes(tickprefix="$")
                    st.plotly_chart(fig_lh, use_container_width=True)
                    st.caption(
                        "Orange shaded = LEAPS deployed. "
                        "Cash earns MMF yield between deployments. "
                        "Each profit lock returns everything to cash on the new ATH."
                    )
                else:
                    st.info("Historical prices unavailable. Forward projection above is fully functional.")
            except Exception as e:
                st.warning(f"Historical LEAPS backtest skipped: {e}")

        st.markdown("---")
        st.markdown("### The decay math: why LEAPS beats SPXL in crashes")
        mc1, mc2 = st.columns(2)
        with mc1:
            drag_ann = vol_drag(3.0, ann_vol)
            st.markdown("**SPXL vol drag** (paid every single day, bull or bear)")
            st.latex(r"	ext{Drag} = rac{L(L-1)}{2}\sigma^2")
            st.markdown(f"= **{drag_ann*100:.2f}%/yr** at σ={ann_vol*100:.0f}% — permanent, compounding")
            st.markdown("**LEAPS theta** (paid only while deployed)")
            eff_theta = theta_s * 12 * leaps_pct_time / 100
            st.markdown(f"= {theta_s*100:.1f}%/mo × {leaps_pct_time:.0f}% active = **{eff_theta:.2f}%/yr effective**")
            st.markdown(f"Theta advantage vs SPXL drag: **{(drag_ann - eff_theta)*100:.1f}%/yr saved**")
        with mc2:
            st.markdown("**Comparison table**")
            eff_lev_leaps = delta_s / 0.15  # approx: delta * S/C where C ≈ 15% of S
            dd_tbl = {
                "":              ["SPXL (3x daily)",       f"Pure LEAPS δ={delta_s:.2f}"],
                "Leverage":      ["3.0x",                  f"~{eff_lev_leaps:.1f}x on cash"],
                "Decay (idle)":  ["Full drag always",      "MMF yield (positive!)"],
                "Decay (active)":[ f"{drag_ann*100:.1f}%/yr", f"{theta_s*12*100:.1f}%/yr theta"],
                "Max drawdown":  ["~90%+ (2008)",          "~60-70% (options ≥ 0)"],
                "Wipeout risk":  ["Real",                  "None — options floor at zero"],
            }
            st.dataframe(pd.DataFrame(dd_tbl), hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
