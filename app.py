# =============================================================================
# VOO vs SPXL  |  Mathematically Correct Leveraged DCA Simulator
# =============================================================================
# Key math fixes vs naive implementations:
#
#  1. Volatility decay (beta-slippage):
#     SPXL daily return = 3×r_spy - (1/2)×(3²-3)×σ²×dt  [Ito's lemma]
#     Annual drag ≈ (L²-L)/2 × σ² = 4 × σ²
#     At σ=18%: drag = 4 × 0.0324 = 12.96% /yr — critical missing piece
#
#  2. Expense ratios applied proportionally to NAV, not subtracted flat
#     VOO: 0.03%/yr | SPXL: 0.91%/yr
#
#  3. Log-normal GBM (geometric Brownian motion) — prevents negative prices
#     log-returns are normal; price changes are log-normal
#
#  4. Historical backtest vs projection — two separate tabs
#
#  5. Monte Carlo fan (10th/50th/90th percentile) for projection uncertainty
#
#  6. Drawdown analysis — essential for 3× leverage decisions
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, date
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 0. PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="VOO vs SPXL — Leveraged DCA",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .main  { background-color: #0b0e14; color: #e1e1e1; }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    h1,h2,h3 { font-family: 'IBM Plex Mono', monospace; }
    [data-testid="stSidebar"] { background-color: #0e1117; }
    .stSpinner > div { border-top-color: #4a9eff !important; }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# 1. SIDEBAR
# =============================================================================
st.sidebar.markdown("## Strategy Controls")

monthly_inv = st.sidebar.number_input(
    "Monthly Contribution ($)", value=100, min_value=10, step=50,
    help="Amount invested at the start of every month into BOTH strategies."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Market Assumptions")

ann_ret = st.sidebar.slider(
    "S&P 500 Annual Return (%)", 0.0, 20.0, 10.0, 0.5,
    help="Expected annual return of S&P 500 before fees. Historic avg ~10%."
) / 100

ann_vol = st.sidebar.slider(
    "Annual Volatility (%)", 5.0, 45.0, 18.0, 0.5,
    help="S&P 500 realised annualised volatility. Historic avg ~16-18%."
) / 100

st.sidebar.markdown("---")
st.sidebar.markdown("### Projection Settings")

proj_years = st.sidebar.slider(
    "Years to Project Forward", 1, 40, 20,
    help="Projection starts from today and extends this many years."
)

n_sim = st.sidebar.select_slider(
    "Monte Carlo Paths", options=[100, 500, 1000, 5000], value=1000,
    help="More paths = smoother percentile bands but slower."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Instrument Parameters")

voo_expense  = st.sidebar.number_input("VOO Expense Ratio (%)",  value=0.03, step=0.01, format="%.2f") / 100
spxl_expense = st.sidebar.number_input("SPXL Expense Ratio (%)", value=0.91, step=0.01, format="%.2f") / 100
leverage     = st.sidebar.number_input("SPXL Leverage Factor",    value=3.0,  step=0.5,  format="%.1f",
                                        help="SPXL = 3× S&P 500. Some prefer UPRO (also 3×) or SSO (2×).")

# =============================================================================
# 2. MATHEMATICAL CORE
# =============================================================================

def vol_drag(lev: float, sigma: float) -> float:
    """
    Annual volatility decay (beta-slippage) for a leveraged ETF.
    Derived from Ito's lemma for a continuously rebalanced leveraged fund:

      dF/F = L × (dS/S) - (1/2)(L² - L)σ² dt

    Annual drag = (L² - L) / 2 × σ²  =  L(L-1)/2 × σ²

    This is the unavoidable mathematical cost of daily reset leverage.
    At 3× and σ=18%: drag = 3×2/2 × 0.0324 = 9.72%/yr
    """
    return lev * (lev - 1) / 2 * sigma ** 2


def net_annual(lev: float, ann_return: float, sigma: float, expense: float) -> float:
    """Net expected annual return after vol drag and expense ratio."""
    return lev * ann_return - vol_drag(lev, sigma) - expense


def dca_gbm_paths(years: int, monthly: float, ann_r: float, ann_s: float,
                   expense: float, lev: float, n_paths: int,
                   seed: int = None) -> np.ndarray:
    """
    Simulate DCA portfolio values using Geometric Brownian Motion.

    Daily log-returns for the ETF:
      r_daily = (L × μ - L(L-1)/2 × σ² - expense/252) × dt
                + L × σ × √dt × Z
    where Z ~ N(0,1) and dt = 1/252.

    Contributions: $monthly added at the start of each calendar month
    (every 21 trading days).

    Returns: array shape (days+1, n_paths)
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    days        = years * 252
    dt          = 1 / 252
    drift_daily = (lev * ann_r - vol_drag(lev, ann_s) - expense) * dt
    sigma_daily = lev * ann_s * np.sqrt(dt)

    # shape: (days, n_paths)
    Z        = rng.standard_normal((days, n_paths))
    log_rets = drift_daily - 0.5 * sigma_daily**2 + sigma_daily * Z

    portfolio = np.zeros((days + 1, n_paths))  # value in dollars

    for i in range(days):
        # Monthly contribution (start of every 21-bar block, not day 0)
        if i > 0 and i % 21 == 0:
            portfolio[i] += monthly
        # Compound
        portfolio[i + 1] = portfolio[i] * np.exp(log_rets[i])

    # Add final month contribution credit
    total_contributions = monthly * (days // 21)
    return portfolio, total_contributions


@st.cache_data(ttl=86400)
def load_historical() -> pd.DataFrame:
    """Download VOO and SPXL historical data from Jan 2010."""
    raw = yf.download(["VOO", "SPXL"], start="2010-01-01",
                      auto_adjust=True, progress=False)
    # yfinance ≥0.2.28 returns MultiIndex (field, ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0).unique().tolist()
        # Try Close, then Adj Close, then first available field
        field = "Close" if "Close" in lvl0 else                 "Adj Close" if "Adj Close" in lvl0 else lvl0[0]
        df = raw[field].copy()
    elif "Close" in raw.columns:
        df = raw[["Close"]].copy()
    else:
        df = raw.copy()
    df.columns = [str(c) for c in df.columns]
    # Keep only rows where both tickers have data
    needed = [c for c in ["VOO", "SPXL"] if c in df.columns]
    if not needed:
        return pd.DataFrame()
    df = df[needed].dropna()
    return df


def run_historical_dca(prices: pd.DataFrame, monthly: float) -> pd.DataFrame:
    """
    Simulate actual DCA into VOO and SPXL using real historical prices.
    Contributions are made on the first trading day of each calendar month.
    Returns a DataFrame with portfolio values over time.
    """
    results = {"Date": [], "VOO": [], "SPXL": [], "Invested": []}

    voo_shares  = 0.0
    spxl_shares = 0.0
    invested    = 0.0
    last_month  = -1

    for date, row in prices.iterrows():
        # First trading day of each new month → buy shares
        m = date.month
        if m != last_month:
            vp  = float(row.get("VOO",  0))
            sp  = float(row.get("SPXL", 0))
            if vp > 0:
                voo_shares  += monthly / vp
            if sp > 0:
                spxl_shares += monthly / sp
            invested  += monthly * 2          # same $ into each
            last_month = m

        voo_val  = voo_shares  * float(row.get("VOO",  0))
        spxl_val = spxl_shares * float(row.get("SPXL", 0))

        results["Date"].append(date)
        results["VOO"].append(round(voo_val, 2))
        results["SPXL"].append(round(spxl_val, 2))
        results["Invested"].append(round(invested / 2, 2))  # per-strategy

    return pd.DataFrame(results).set_index("Date")


def compute_drawdown(series: pd.Series) -> pd.Series:
    roll_max = series.cummax()
    return ((series - roll_max) / roll_max * 100).clip(upper=0)


def format_currency(v: float) -> str:
    if v >= 1_000_000:
        return f"${v/1_000_000:.2f}M"
    elif v >= 1_000:
        return f"${v:,.0f}"
    return f"${v:.2f}"


# =============================================================================
# 3. HTML HELPERS
# =============================================================================
def _kpi(col, label, val, sub=None, color="#e8eaf0"):
    sub_html = (f'<p style="margin:2px 0 0;font-size:11px;color:#4a5568;font-family:monospace;">{sub}</p>'
                if sub else "")
    col.markdown(
        '<div style="background:#111318;padding:14px 16px;border-radius:4px;'
        'border:1px solid rgba(255,255,255,0.07);text-align:center;">'
        '<p style="margin:0 0 3px;font-size:9px;color:#4a5568;font-family:monospace;'
        'text-transform:uppercase;letter-spacing:0.1em;">' + label + '</p>'
        '<p style="margin:0;font-family:monospace;font-size:20px;font-weight:600;color:'
        + color + ';">' + str(val) + '</p>'
        + sub_html + '</div>',
        unsafe_allow_html=True,
    )


def _math_card(title, formula, explanation, color="#4a9eff"):
    st.markdown(
        '<div style="background:#111318;border:1px solid rgba(255,255,255,0.07);'
        'border-left:3px solid ' + color + ';border-radius:4px;padding:14px 18px;">'
        '<p style="font-family:monospace;font-size:11px;font-weight:600;color:' + color + ';margin:0 0 6px;">'
        + title + '</p>'
        '<p style="font-family:monospace;font-size:12px;color:#e8c96d;margin:0 0 6px;">'
        + formula + '</p>'
        '<p style="font-size:11px;color:#8892a4;margin:0;line-height:1.65;">'
        + explanation + '</p>'
        '</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# 3b.  SPRINGBOARD ENGINE
# =============================================================================
def run_springboard(years: int, monthly: float, ann_r: float, ann_s: float,
                    voo_exp: float, spxl_exp: float, lev: float,
                    trigger_pct: float, n_paths: int = 200,
                    seed: int = 42) -> dict:
    """
    Springboard Strategy (Mathematically Correct):
      - Normally holds VOO (1× S&P500)
      - Switches to SPXL (3×) when S&P500 falls trigger_pct% below 50-day high
      - Returns to VOO once S&P recovers to the PREVIOUS high
      - Uses log-normal GBM with proper vol-drag on SPXL

    Returns median path, percentile bands, and time-in-leverage stats.
    """
    rng    = np.random.default_rng(seed)
    days   = years * 252
    dt     = 1 / 252

    # GBM parameters for SPX (underlying)
    mu_spx     = ann_r
    sig_spx    = ann_s
    drift_spx  = (mu_spx - 0.5 * sig_spx**2) * dt
    sigma_spx  = sig_spx * np.sqrt(dt)

    # Vol drag for leveraged ETF
    drag_spxl  = lev * (lev - 1) / 2 * ann_s**2

    # Daily log-return params
    drift_voo  = (1.0 * ann_r - 0.5 * sig_spx**2 - voo_exp)  * dt
    sig_voo    = sig_spx * np.sqrt(dt)

    drift_spxl = (lev * ann_r - drag_spxl - spxl_exp - 0.5 * (lev*sig_spx)**2) * dt
    sig_spxl   = lev * sig_spx * np.sqrt(dt)

    all_voo    = np.zeros((days + 1, n_paths))
    all_spxl   = np.zeros((days + 1, n_paths))
    all_spring = np.zeros((days + 1, n_paths))
    lev_frac   = np.zeros(n_paths)

    Z = rng.standard_normal((days, n_paths))

    for p in range(n_paths):
        spx_log  = np.cumsum(drift_spx + sigma_spx * Z[:, p])
        spx_idx  = np.concatenate([[1.0], np.exp(spx_log)])   # SPX price index

        # Rolling 50-day high of SPX index
        spx_s     = pd.Series(spx_idx)
        roll_high = spx_s.rolling(50, min_periods=1).max().values
        trigger   = 1 - trigger_pct / 100

        # Portfolio values
        voo_val    = 0.0
        spxl_val   = 0.0
        spring_val = 0.0
        in_lev     = False
        rec_high   = 0.0
        lev_days_p = 0

        for i in range(days):
            # Monthly contribution (first day of each ~21-bar block)
            if i > 0 and i % 21 == 0:
                voo_val    += monthly
                spxl_val   += monthly
                spring_val += monthly

            # Compound
            r_voo    = drift_voo   + sig_voo   * Z[i, p]
            r_spxl   = drift_spxl  + sig_spxl  * Z[i, p]   # correlated same Z

            voo_val    *= np.exp(r_voo)
            spxl_val   *= np.exp(r_spxl)

            if in_lev:
                spring_val *= np.exp(r_spxl)
                lev_days_p += 1
            else:
                spring_val *= np.exp(r_voo)

            all_voo[i+1, p]    = voo_val
            all_spxl[i+1, p]   = spxl_val
            all_spring[i+1, p] = spring_val

            # Springboard trigger/exit logic
            curr_spx  = spx_idx[i + 1]
            curr_high = roll_high[i + 1]
            if not in_lev:
                if curr_spx < trigger * curr_high:
                    in_lev  = True
                    rec_high = curr_high
            else:
                if curr_spx >= rec_high:
                    in_lev = False

        lev_frac[p] = lev_days_p / days

    pcts = [10, 25, 50, 75, 90]
    return {
        "voo_p50":    np.percentile(all_voo,    50, axis=1),
        "spxl_p50":   np.percentile(all_spxl,   50, axis=1),
        "spring_p10": np.percentile(all_spring, 10, axis=1),
        "spring_p25": np.percentile(all_spring, 25, axis=1),
        "spring_p50": np.percentile(all_spring, 50, axis=1),
        "spring_p75": np.percentile(all_spring, 75, axis=1),
        "spring_p90": np.percentile(all_spring, 90, axis=1),
        "lev_pct_median":  float(np.median(lev_frac) * 100),
        "lev_pct_p90":     float(np.percentile(lev_frac, 90) * 100),
        "n_paths":    n_paths,
    }


# =============================================================================
# 4. MAIN
# =============================================================================
def main():
    st.title("VOO vs SPXL  |  Leveraged DCA Simulator")
    st.caption(
        f"${monthly_inv}/month · {proj_years}yr projection · "
        f"{int(leverage)}× leverage · "
        f"S&P assumption: {ann_ret*100:.1f}% return, {ann_vol*100:.0f}% vol"
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Historical Backtest (2010–Today)",
        "🔮 Forward Projection (Monte Carlo)",
        "🌱 Springboard Strategy",
        "📐 The Math",
    ])

    # =========================================================================
    # TAB 1 — HISTORICAL BACKTEST
    # =========================================================================
    with tab1:
        st.markdown(
            '<p style="font-family:monospace;font-size:11px;color:#4a5568;margin-bottom:12px;">'
            'Real DCA into VOO and SPXL using actual daily prices from Jan 2010. '
            'Contributions made on the first trading day of each calendar month.</p>',
            unsafe_allow_html=True,
        )

        with st.spinner("Loading historical prices..."):
            prices = load_historical()

        if prices.empty or "VOO" not in prices.columns or "SPXL" not in prices.columns:
            st.error("Could not load VOO or SPXL data. Check your internet connection.")
            return

        hist = run_historical_dca(prices, monthly_inv)
        if hist.empty or len(hist) == 0:
            st.error("Historical data returned no rows. Try refreshing.")
            return
        invested_final = hist["Invested"].iloc[-1]
        voo_hist_final = hist["VOO"].iloc[-1]
        spxl_hist_final = hist["SPXL"].iloc[-1]
        years_hist = (hist.index[-1] - hist.index[0]).days / 365.25

        voo_cagr  = ((voo_hist_final  / invested_final) ** (1 / years_hist) - 1) * 100
        spxl_cagr = ((spxl_hist_final / invested_final) ** (1 / years_hist) - 1) * 100

        # KPI strip
        k = st.columns(6)
        _kpi(k[0], "VOO Final",    format_currency(voo_hist_final),
             f"{years_hist:.1f}yr DCA", "#4a9eff")
        _kpi(k[1], "SPXL Final",   format_currency(spxl_hist_final),
             f"vs ${invested_final:,.0f} invested", "#ff4b4b")
        _kpi(k[2], "SPXL/VOO",     f"{spxl_hist_final/voo_hist_final:.2f}×",
             "outperformance ratio", "#00ffcc")
        _kpi(k[3], "VOO CAGR",     f"{voo_cagr:.1f}%",  "on invested capital", "#4a9eff")
        _kpi(k[4], "SPXL CAGR",    f"{spxl_cagr:.1f}%", "on invested capital", "#ff4b4b")
        _kpi(k[5], "Principal",     format_currency(invested_final),
             f"${monthly_inv}/mo × {int(years_hist*12)}mo", "#8892a4")

        st.markdown("<br>", unsafe_allow_html=True)

        # Main equity chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.7, 0.3], vertical_spacing=0.04)

        fig.add_trace(go.Scatter(
            x=hist.index, y=hist["Invested"],
            name="Capital Invested", mode="lines",
            line=dict(color="rgba(255,255,255,0.25)", width=1.5, dash="dot"),
            fill="tozeroy", fillcolor="rgba(255,255,255,0.03)",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist["VOO"],
            name="VOO", line=dict(color="#4a9eff", width=2.5),
            hovertemplate="%{x|%b %Y}  VOO: $%{y:,.0f}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist["SPXL"],
            name="SPXL (3×)", line=dict(color="#ff4b4b", width=2.5),
            hovertemplate="%{x|%b %Y}  SPXL: $%{y:,.0f}<extra></extra>",
        ), row=1, col=1)

        # Drawdown
        dd_voo  = compute_drawdown(hist["VOO"])
        dd_spxl = compute_drawdown(hist["SPXL"])
        fig.add_trace(go.Scatter(
            x=hist.index, y=dd_voo,
            name="VOO Drawdown", fill="tozeroy",
            fillcolor="rgba(74,158,255,0.2)", line=dict(color="#4a9eff", width=1),
            hovertemplate="%{x|%b %Y}  %{y:.1f}%<extra></extra>",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=hist.index, y=dd_spxl,
            name="SPXL Drawdown", fill="tozeroy",
            fillcolor="rgba(255,75,75,0.2)", line=dict(color="#ff4b4b", width=1),
            hovertemplate="%{x|%b %Y}  %{y:.1f}%<extra></extra>",
        ), row=2, col=1)

        fig.update_layout(
            template="plotly_dark", height=560,
            paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
            margin=dict(l=12, r=12, t=12, b=12),
            legend=dict(orientation="h", yanchor="bottom", y=1.01,
                        xanchor="left", x=0,
                        font=dict(family="IBM Plex Mono", size=10, color="#8892a4"),
                        bgcolor="rgba(0,0,0,0)"),
            hovermode="x unified", font=dict(family="IBM Plex Mono"),
        )
        fig.update_yaxes(showgrid=False, zeroline=False, tickprefix="$", row=1, col=1)
        fig.update_yaxes(showgrid=False, zeroline=True, zerolinecolor="rgba(255,255,255,0.1)",
                         ticksuffix="%", title_text="Drawdown", row=2, col=1)
        fig.update_xaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

        # Worst drawdown callout
        worst_voo  = dd_voo.min()
        worst_spxl = dd_spxl.min()
        st.markdown(
            '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:4px;">'
            '<div style="background:rgba(74,158,255,0.07);border:1px solid rgba(74,158,255,0.25);'
            'border-radius:4px;padding:10px 14px;">'
            '<p style="font-family:monospace;font-size:10px;color:#4a9eff;margin:0 0 3px;">'
            'VOO WORST DRAWDOWN</p>'
            '<p style="font-family:monospace;font-size:20px;font-weight:600;color:#4a9eff;margin:0;">'
            f'{worst_voo:.1f}%</p></div>'
            '<div style="background:rgba(255,75,75,0.07);border:1px solid rgba(255,75,75,0.25);'
            'border-radius:4px;padding:10px 14px;">'
            '<p style="font-family:monospace;font-size:10px;color:#ff4b4b;margin:0 0 3px;">'
            'SPXL WORST DRAWDOWN</p>'
            '<p style="font-family:monospace;font-size:20px;font-weight:600;color:#ff4b4b;margin:0;">'
            f'{worst_spxl:.1f}%'
            '<span style="font-size:11px;color:#8892a4;margin-left:8px;">Can you stomach this?</span>'
            '</p></div></div>',
            unsafe_allow_html=True,
        )

    # =========================================================================
    # TAB 2 — FORWARD PROJECTION (MONTE CARLO)
    # =========================================================================
    with tab2:
        # Pre-compute key metrics
        voo_net  = net_annual(1.0, ann_ret, ann_vol, voo_expense)
        spxl_net = net_annual(leverage, ann_ret, ann_vol, spxl_expense)
        drag     = vol_drag(leverage, ann_vol)

        st.markdown(
            '<div style="background:#111318;border-radius:4px;padding:12px 18px;'
            'margin-bottom:16px;display:flex;gap:24px;flex-wrap:wrap;">'
            '<span style="font-family:monospace;font-size:11px;color:#4a5568;">'
            'Expected net annual return: '
            '<b style="color:#4a9eff;">VOO ' + f'{voo_net*100:.2f}%</b>'
            '&nbsp;&nbsp;vs&nbsp;&nbsp;'
            '<b style="color:#ff4b4b;">SPXL ' + f'{spxl_net*100:.2f}%</b>'
            '</span>'
            '<span style="font-family:monospace;font-size:11px;color:#f5a623;">'
            '⚠ Vol drag on SPXL: ' + f'{drag*100:.2f}%/yr'
            ' (the math cost of daily reset leverage)'
            '</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        with st.spinner(f"Running {n_sim:,} Monte Carlo paths..."):
            voo_mc,  voo_invested  = dca_gbm_paths(
                proj_years, monthly_inv, ann_ret, ann_vol, voo_expense,  1.0, n_sim, seed=42)
            spxl_mc, spxl_invested = dca_gbm_paths(
                proj_years, monthly_inv, ann_ret, ann_vol, spxl_expense, leverage, n_sim, seed=42)

        # Percentiles at each day
        pcts = [10, 25, 50, 75, 90]
        voo_p  = np.percentile(voo_mc,  pcts, axis=1)   # shape (5, days+1)
        spxl_p = np.percentile(spxl_mc, pcts, axis=1)

        # Final day stats
        voo_final_vals  = voo_mc[-1]
        spxl_final_vals = spxl_mc[-1]

        # KPI strip
        k = st.columns(6)
        _kpi(k[0], "VOO Median",    format_currency(float(np.median(voo_final_vals))),
             f"{proj_years}yr projection", "#4a9eff")
        _kpi(k[1], "SPXL Median",   format_currency(float(np.median(spxl_final_vals))),
             f"at {leverage:.0f}× leverage", "#ff4b4b")
        _kpi(k[2], "SPXL/VOO (P50)", f"{np.median(spxl_final_vals)/np.median(voo_final_vals):.1f}×",
             "median ratio", "#00ffcc")
        _kpi(k[3], "SPXL P10 (bad)", format_currency(float(np.percentile(spxl_final_vals, 10))),
             "worst 10% outcome", "#f5a623")
        _kpi(k[4], "SPXL P90 (good)",format_currency(float(np.percentile(spxl_final_vals, 90))),
             "best 10% outcome", "#00d4a0")
        _kpi(k[5], "Total Invested", format_currency(voo_invested),
             f"${monthly_inv}/mo × {proj_years*12}mo", "#8892a4")

        st.markdown("<br>", unsafe_allow_html=True)

        # Fan chart
        days_arr = np.arange(proj_years * 252 + 1)
        years_arr = days_arr / 252

        fig2 = make_subplots(rows=1, cols=2,
                             subplot_titles=["VOO (1×)", f"SPXL ({leverage:.0f}×)"],
                             shared_yaxes=False)

        def add_fan(fig, pcts_arr, col_hex, col_num):
            # Shaded bands P10–P90, P25–P75
            fig.add_trace(go.Scatter(
                x=np.concatenate([years_arr, years_arr[::-1]]),
                y=np.concatenate([pcts_arr[0], pcts_arr[4][::-1]]),
                fill="toself", fillcolor=col_hex.replace(")", ",0.10)").replace("rgb", "rgba"),
                line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
                name="P10–P90",
            ), row=1, col=col_num)
            fig.add_trace(go.Scatter(
                x=np.concatenate([years_arr, years_arr[::-1]]),
                y=np.concatenate([pcts_arr[1], pcts_arr[3][::-1]]),
                fill="toself", fillcolor=col_hex.replace(")", ",0.18)").replace("rgb", "rgba"),
                line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
                name="P25–P75",
            ), row=1, col=col_num)
            # Median line
            fig.add_trace(go.Scatter(
                x=years_arr, y=pcts_arr[2],
                name="Median (P50)", line=dict(color=col_hex, width=2.5),
                hovertemplate="Year %{x:.1f}  $%{y:,.0f}<extra></extra>",
            ), row=1, col=col_num)

        add_fan(fig2, voo_p,  "rgb(74,158,255)",  1)
        add_fan(fig2, spxl_p, "rgb(255,75,75)",   2)

        # Principal line on both
        principal = np.arange(len(years_arr)) // 21 * monthly_inv
        for cn in [1, 2]:
            fig2.add_trace(go.Scatter(
                x=years_arr, y=principal,
                name="Capital Invested", showlegend=(cn==1),
                line=dict(color="rgba(255,255,255,0.25)", width=1.5, dash="dot"),
            ), row=1, col=cn)

        fig2.update_layout(
            template="plotly_dark", height=480,
            paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
            margin=dict(l=12, r=12, t=40, b=12),
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0,
                        font=dict(family="IBM Plex Mono", size=10, color="#8892a4"),
                        bgcolor="rgba(0,0,0,0)"),
            hovermode="x unified", font=dict(family="IBM Plex Mono"),
        )
        fig2.update_xaxes(showgrid=False, title_text="Years")
        fig2.update_yaxes(showgrid=False, zeroline=False, tickprefix="$")
        st.plotly_chart(fig2, use_container_width=True)

        # Distribution of final values
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(
            x=voo_final_vals, nbinsx=80, name="VOO Final",
            marker_color="rgba(74,158,255,0.6)",
            hovertemplate="$%{x:,.0f}  count: %{y}<extra></extra>",
        ))
        fig3.add_trace(go.Histogram(
            x=spxl_final_vals, nbinsx=80, name="SPXL Final",
            marker_color="rgba(255,75,75,0.5)",
            hovertemplate="$%{x:,.0f}  count: %{y}<extra></extra>",
        ))
        fig3.update_layout(
            template="plotly_dark", height=280, barmode="overlay",
            paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
            margin=dict(l=12, r=12, t=10, b=12),
            legend=dict(font=dict(family="IBM Plex Mono", size=10, color="#8892a4"),
                        bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(showgrid=False, tickprefix="$", title="Final Portfolio Value"),
            yaxis=dict(showgrid=False),
            font=dict(family="IBM Plex Mono"),
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.caption(
            f"Distribution of {n_sim:,} simulated final values after {proj_years} years. "
            "SPXL's distribution is wider — higher upside but also more paths ending below VOO."
        )

    # =========================================================================
    # TAB 3 — SPRINGBOARD STRATEGY
    # =========================================================================
    with tab3:
        st.markdown(
            '<p style="font-family:monospace;font-size:11px;color:#4a5568;margin-bottom:12px;">'
            'The Springboard strategy holds VOO normally, but switches to SPXL (3×) during '
            'S&P 500 corrections and returns to VOO once the market recovers to its previous high. '
            'It uses leverage only when mean-reversion probability is highest.</p>',
            unsafe_allow_html=True,
        )

        # Springboard-specific sidebar controls rendered inline
        sb_col1, sb_col2 = st.columns([1, 2])
        with sb_col1:
            trigger_pct = st.slider(
                "Correction Trigger — % below 50-day high",
                3.0, 25.0, 10.0, 0.5,
                help="Enter SPXL when S&P drops this far below its 50-day rolling high.",
            )
            sb_paths = st.select_slider(
                "Monte Carlo paths", options=[50, 200, 500], value=200,
            )

        with sb_col2:
            st.markdown(
                '<div style="background:#111318;border-radius:4px;padding:14px 18px;'
                'border-left:3px solid #00ffcc;">'
                '<p style="font-family:monospace;font-size:10px;color:#00ffcc;margin:0 0 8px;'
                'text-transform:uppercase;letter-spacing:0.1em;">How Springboard Works</p>'
                '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">'
                '<div><p style="font-family:monospace;font-size:11px;color:#e8eaf0;margin:0 0 4px;">① Normal (VOO)</p>'
                '<p style="font-size:11px;color:#4a5568;margin:0;line-height:1.6;">'
                'Hold 1× S&P 500. Steady compounding. Zero vol drag.</p></div>'
                '<div><p style="font-family:monospace;font-size:11px;color:#ff4b4b;margin:0 0 4px;">② Correction → SPXL</p>'
                '<p style="font-size:11px;color:#4a5568;margin:0;line-height:1.6;">'
                'Switch to 3× when S&P drops >' + str(trigger_pct) + '% below 50-day high. '
                'Maximum reversion probability.</p></div>'
                '<div><p style="font-family:monospace;font-size:11px;color:#00ffcc;margin:0 0 4px;">③ Recovery → VOO</p>'
                '<p style="font-size:11px;color:#4a5568;margin:0;line-height:1.6;">'
                'Return to VOO once market recovers to previous high. '
                'Lock gains, escape vol drag.</p></div>'
                '</div></div>',
                unsafe_allow_html=True,
            )

        with st.spinner(f"Running {sb_paths} Springboard paths..."):
            sb = run_springboard(
                proj_years, monthly_inv, ann_ret, ann_vol,
                voo_expense, spxl_expense, leverage,
                trigger_pct, n_paths=sb_paths,
            )

        days_arr_sb = np.arange(proj_years * 252 + 1)
        yrs_arr_sb  = days_arr_sb / 252

        # KPI strip
        principal_sb = monthly_inv * proj_years * 12
        sp50 = float(sb["spring_p50"][-1])
        vp50 = float(sb["voo_p50"][-1])
        mult = sp50 / vp50 if vp50 > 0 else 0

        k = st.columns(5)
        _kpi(k[0], "Springboard P50", format_currency(sp50),
             f"vs VOO {format_currency(vp50)}", "#00ffcc")
        _kpi(k[1], "Springboard/VOO", f"{mult:.2f}×",
             "median outperformance", "#00ffcc")
        _kpi(k[2], "Springboard P10", format_currency(float(sb["spring_p10"][-1])),
             "worst 10% outcome", "#f5a623")
        _kpi(k[3], "Springboard P90", format_currency(float(sb["spring_p90"][-1])),
             "best 10% outcome", "#00d4a0")
        _kpi(k[4], "Avg Time in SPXL", f"{sb['lev_pct_median']:.1f}%",
             f"up to {sb['lev_pct_p90']:.1f}% in bad markets", "#ff4b4b")

        st.markdown("<br>", unsafe_allow_html=True)

        # Springboard fan chart vs VOO and SPXL medians
        fig_sb = go.Figure()

        # SPXL median (reference — always-leveraged)
        fig_sb.add_trace(go.Scatter(
            x=yrs_arr_sb, y=sb["spxl_p50"],
            name="SPXL P50 (always 3×)", mode="lines",
            line=dict(color="#ff4b4b", width=1.5, dash="dot"),
            hovertemplate="Year %{x:.1f}  SPXL: $%{y:,.0f}<extra></extra>",
        ))
        # VOO median
        fig_sb.add_trace(go.Scatter(
            x=yrs_arr_sb, y=sb["voo_p50"],
            name="VOO P50 (always 1×)", mode="lines",
            line=dict(color="#4a9eff", width=2),
            hovertemplate="Year %{x:.1f}  VOO: $%{y:,.0f}<extra></extra>",
        ))
        # Springboard fan P10-P90
        fig_sb.add_trace(go.Scatter(
            x=np.concatenate([yrs_arr_sb, yrs_arr_sb[::-1]]),
            y=np.concatenate([sb["spring_p10"], sb["spring_p90"][::-1]]),
            fill="toself", fillcolor="rgba(0,255,159,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Springboard P10–P90", hoverinfo="skip",
        ))
        fig_sb.add_trace(go.Scatter(
            x=np.concatenate([yrs_arr_sb, yrs_arr_sb[::-1]]),
            y=np.concatenate([sb["spring_p25"], sb["spring_p75"][::-1]]),
            fill="toself", fillcolor="rgba(0,255,159,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Springboard P25–P75", hoverinfo="skip",
        ))
        # Springboard median (bold)
        fig_sb.add_trace(go.Scatter(
            x=yrs_arr_sb, y=sb["spring_p50"],
            name="Springboard P50 ✦", mode="lines",
            line=dict(color="#00ff9f", width=3),
            hovertemplate="Year %{x:.1f}  Springboard: $%{y:,.0f}<extra></extra>",
        ))

        # Capital invested line
        cap_line = (np.arange(len(yrs_arr_sb)) // 21) * monthly_inv
        fig_sb.add_trace(go.Scatter(
            x=yrs_arr_sb, y=cap_line,
            name="Capital Invested", mode="lines",
            line=dict(color="rgba(255,255,255,0.2)", width=1.2, dash="dot"),
        ))

        fig_sb.update_layout(
            template="plotly_dark", height=460,
            paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
            margin=dict(l=12, r=12, t=12, b=12),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                        font=dict(family="IBM Plex Mono", size=10, color="#8892a4"),
                        bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(showgrid=False, title_text="Years"),
            yaxis=dict(showgrid=False, zeroline=False, tickprefix="$"),
            hovermode="x unified", font=dict(family="IBM Plex Mono"),
        )
        st.plotly_chart(fig_sb, use_container_width=True)

        # Key insight callout
        st.markdown(
            '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:4px;">'

            '<div style="background:rgba(0,255,159,0.07);border:1px solid rgba(0,255,159,0.25);'
            'border-radius:4px;padding:12px 16px;">'
            '<p style="font-family:monospace;font-size:10px;color:#00ffcc;margin:0 0 6px;">'
            'WHY SPRINGBOARD BEATS PURE SPXL</p>'
            '<p style="font-size:11px;color:#8892a4;margin:0;line-height:1.65;">'
            'Leverage is only applied after a correction — when upside is highest and '
            'vol is elevated (maximising leverage gains). The rest of the time VOO '
            'avoids the daily vol-drag of ' + f"{vol_drag(leverage, ann_vol)*100:.2f}%" + '/yr '
            'that constantly erodes SPXL in normal markets.'
            '</p></div>'

            '<div style="background:rgba(245,166,23,0.07);border:1px solid rgba(245,166,23,0.25);'
            'border-radius:4px;padding:12px 16px;">'
            '<p style="font-family:monospace;font-size:10px;color:#f5a623;margin:0 0 6px;">'
            'RISKS TO UNDERSTAND</p>'
            '<p style="font-size:11px;color:#8892a4;margin:0;line-height:1.65;">'
            'In prolonged bear markets (2000-2002, 2008-2009), the trigger fires and '
            'SPXL keeps falling. The recovery-high exit does not protect against '
            'multi-year drawdowns. Also: switching costs and taxes at each transition.'
            '</p></div>'

            '</div>',
            unsafe_allow_html=True,
        )

    # =========================================================================
    # TAB 4 — THE MATH (was TAB 3)
    # =========================================================================
    with tab4:
        st.markdown(
            "### Why leveraged ETFs don't simply return 3× the index\n"
            "The mathematics of daily-reset leverage creates a permanent drag "
            "on long-term returns. Here's the complete picture."
        )

        col_a, col_b = st.columns(2)

        with col_a:
            _math_card(
                "Volatility Decay (Beta-Slippage)",
                "Annual drag = L(L−1)/2 × σ²",
                "Derived from Itō's lemma. A 3× ETF that resets daily is not '3× buy and hold'. "
                "The daily reset means sideways volatility destroys value. "
                f"At {ann_vol*100:.0f}% vol: drag = {leverage:.0f}×{leverage-1:.0f}/2 × {ann_vol**2:.4f} "
                f"= <b>{vol_drag(leverage,ann_vol)*100:.2f}%/yr</b>.",
                "#ff4b4b",
            )
            st.markdown("<br>", unsafe_allow_html=True)
            _math_card(
                "Net Expected Return",
                f"E[r_SPXL] = {leverage:.0f}×μ − drag − fee",
                f"= {leverage:.0f} × {ann_ret*100:.1f}% "
                f"− {vol_drag(leverage,ann_vol)*100:.2f}% "
                f"− {spxl_expense*100:.2f}% "
                f"= <b>{net_annual(leverage,ann_ret,ann_vol,spxl_expense)*100:.2f}%/yr</b><br>"
                f"VOO: 1×{ann_ret*100:.1f}% − {voo_expense*100:.2f}% "
                f"= <b>{net_annual(1,ann_ret,ann_vol,voo_expense)*100:.2f}%/yr</b>",
                "#4a9eff",
            )

        with col_b:
            _math_card(
                "Geometric Brownian Motion (GBM)",
                "dS = μS dt + σS dW",
                "Log-returns are normally distributed; prices are log-normal. "
                "This prevents the simulation from generating negative portfolio values "
                "(which simple arithmetic return models incorrectly allow).<br>"
                "Each daily log-return: r = (μ − ½σ²)dt + σ√dt × Z, where Z~N(0,1).",
                "#00ffcc",
            )
            st.markdown("<br>", unsafe_allow_html=True)
            _math_card(
                "Monte Carlo Fan",
                "P10 / P25 / P50 / P75 / P90",
                f"Runs {n_sim:,} independent paths. The fan bands show the range of outcomes. "
                "SPXL's fan is much wider — in good markets it massively outperforms; "
                "in bad markets it can lose >90% while VOO loses 50%. "
                "The P10 (worst 10%) outcome for SPXL is the key risk metric.",
                "#a78bfa",
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Interactive vol drag table
        st.markdown("### Vol Drag Sensitivity Table")
        st.markdown(
            "How the annual drag changes with volatility and leverage factor. "
            "The current scenario is highlighted."
        )

        vols = [10, 15, 18, 20, 25, 30, 35, 40]
        levs = [1.5, 2.0, 2.5, 3.0, 3.5]
        data = {}
        for lv in levs:
            col_name = f"{lv:.1f}× (e.g. {'SSO' if lv==2 else 'SPXL' if lv==3 else 'ETF'})"
            data[col_name] = [f"{vol_drag(lv, v/100)*100:.2f}%" for v in vols]
        drag_df = pd.DataFrame(data, index=[f"{v}% vol" for v in vols])
        st.dataframe(drag_df, use_container_width=True)

        st.markdown(
            '<div style="background:rgba(245,166,23,0.08);border:1px solid rgba(245,166,23,0.3);'
            'border-radius:4px;padding:12px 16px;margin-top:12px;">'
            '<p style="font-family:monospace;font-size:11px;color:#f5a623;margin:0 0 6px;">⚠ Key Insight</p>'
            '<p style="font-size:12px;color:#8892a4;margin:0;line-height:1.7;">'
            'Volatility drag is <b style="color:#e8eaf0;">quadratic in leverage and volatility</b>. '
            'Going from 2× to 3× leverage triples the drag. '
            'Going from 18% to 35% vol nearly quadruples it. '
            'This is why SPXL underperforms 3× S&P 500 by a large margin over long periods, '
            'even when the index is trending upward.'
            '</p></div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
