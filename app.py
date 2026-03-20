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


def dca_gbm_paths(years, monthly, ann_r, ann_s, expense, lev, n_paths, seed=42):
    rng = np.random.default_rng(seed)
    days = years * 252
    dt = 1 / 252
    drag = vol_drag(lev, ann_s)
    drift_daily = (lev * ann_r - drag - expense - 0.5 * (lev * ann_s) ** 2) * dt
    sigma_daily = lev * ann_s * np.sqrt(dt)

    Z = rng.standard_normal((days, n_paths))
    log_rets = drift_daily + sigma_daily * Z

    portfolio = np.zeros((days + 1, n_paths))
    for i in range(days):
        if i > 0 and i % 21 == 0:
            portfolio[i] += monthly
        portfolio[i + 1] = portfolio[i] * np.exp(log_rets[i])

    invested = monthly * (days // 21)
    return portfolio, invested


@st.cache_data(ttl=86400)
def load_historical() -> pd.DataFrame:
    try:
        raw = yf.download(["VOO", "SPXL"], start="2010-01-01", auto_adjust=True, progress=False)
        if raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            # yfinance >= 0.2.x returns (field, ticker) MultiIndex
            level0 = raw.columns.get_level_values(0).unique().tolist()
            field = "Close" if "Close" in level0 else ("Adj Close" if "Adj Close" in level0 else level0[0])
            df = raw[field].copy()
        else:
            df = raw.copy()
        df.columns = [str(c) for c in df.columns]
        # Ensure both tickers present
        for col in ["VOO", "SPXL"]:
            if col not in df.columns:
                return pd.DataFrame()
        df = df[["VOO", "SPXL"]].dropna()
        return df
    except Exception:
        return pd.DataFrame()


def run_historical_dca(prices: pd.DataFrame, monthly: float) -> pd.DataFrame:
    results = {"Date": [], "VOO": [], "SPXL": [], "Invested": []}
    voo_shares = spxl_shares = invested = 0.0
    last_month = -1

    for date, row in prices.iterrows():
        m = date.month
        if m != last_month:
            vp = float(row.get("VOO", 0))
            sp = float(row.get("SPXL", 0))
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
# SPRINGBOARD ENGINE — Simple $90 SPXL / $10 VOO DCA + Drawdown Rebalancer
# =============================================================================
# Rules:
#   Every month: invest spxl_monthly into SPXL, voo_monthly into VOO (base DCA)
#   Additionally: track VOO's drawdown from its all-time-high (ATH)
#     - Linear shift: at dd_pct% below ATH, move shift_pct% of VOO portfolio → SPXL
#     - At max_dd% below ATH → 100% of VOO portfolio value is in SPXL
#     - As VOO recovers back toward ATH, reverse the shift proportionally
#   Monthly contribution redirect: same linear ratio applied to the $10 VOO DCA
#     e.g. at -20% drawdown → redirect 50% of that month's VOO $ into SPXL instead
# =============================================================================

def _drawdown_shift_pct(drawdown: float, max_dd: float) -> float:
    """
    How much of the VOO portfolio to shift into SPXL.
    drawdown: positive fraction (e.g. 0.20 = 20% below ATH)
    Returns fraction 0.0 → 1.0
    """
    return float(np.clip(drawdown / max_dd, 0.0, 1.0))


def run_springboard(
    years, monthly,          # total monthly DCA
    ann_r, ann_s,
    voo_exp, spxl_exp, lev,
    spxl_monthly_pct=0.90,   # base split: 90% of monthly → SPXL
    max_dd=0.40,             # drawdown at which 100% of VOO value shifts to SPXL
    n_paths=200, seed=42,
):
    """
    spxl_monthly_pct: fraction of monthly DCA that always goes to SPXL (default 0.90)
    voo_monthly_pct:  remainder always goes to VOO (default 0.10)
    On top of that, drawdown rebalancer shifts VOO portfolio value to SPXL linearly.
    """
    rng = np.random.default_rng(seed)
    days  = years * 252
    dt    = 1 / 252

    voo_monthly  = monthly * (1.0 - spxl_monthly_pct)
    spxl_monthly = monthly * spxl_monthly_pct

    drag_spxl   = vol_drag(lev, ann_s)
    drift_voo   = (ann_r - voo_exp  - 0.5 * ann_s**2)          * dt
    sig_voo     = ann_s * np.sqrt(dt)
    drift_spxl  = (lev*ann_r - drag_spxl - spxl_exp - 0.5*(lev*ann_s)**2) * dt
    sig_spxl    = lev * ann_s * np.sqrt(dt)

    # also need a "reference VOO price" for ATH tracking — use a clean GBM index
    drift_spx   = (ann_r - 0.5 * ann_s**2) * dt
    sig_spx     = ann_s * np.sqrt(dt)

    all_spring       = np.zeros((days + 1, n_paths))
    all_voo_baseline = np.zeros((days + 1, n_paths))
    all_spxl_alloc   = np.zeros((days + 1, n_paths))   # fraction in SPXL each day
    shifted_frac     = np.zeros(n_paths)

    Z = rng.standard_normal((days, n_paths))

    for p in range(n_paths):
        # --- portfolio buckets ---
        voo_bucket  = 0.0   # VOO portion (shrinks when we shift to SPXL)
        spxl_bucket = 0.0   # SPXL portion (grows from base DCA + shifted value)
        voo_base    = 0.0   # pure VOO baseline (no shifting) for comparison

        voo_ath     = 1.0   # track ATH of VOO price index
        voo_price   = 1.0   # synthetic VOO price index

        total_shifted_days = 0

        for i in range(days):
            # ── monthly contributions ──────────────────────────────────────
            if i > 0 and i % 21 == 0:
                # current drawdown
                dd = max(0.0, (voo_ath - voo_price) / voo_ath)
                shift = _drawdown_shift_pct(dd, max_dd)

                # base DCA: spxl_monthly always goes to SPXL
                spxl_bucket += spxl_monthly
                voo_base    += voo_monthly

                # VOO DCA redirect: shift fraction of voo_monthly → SPXL
                spxl_bucket += voo_monthly * shift
                voo_bucket  += voo_monthly * (1.0 - shift)

                # portfolio rebalance: move shift% of current voo_bucket → spxl_bucket
                # (only rebalance the delta from last month's shift level)
                rebal_amount = voo_bucket * shift
                spxl_bucket += rebal_amount
                voo_bucket  -= rebal_amount

            # ── daily returns ──────────────────────────────────────────────
            r_voo  = drift_voo  + sig_voo  * Z[i, p]
            r_spxl = drift_spxl + sig_spxl * Z[i, p]
            r_spx  = drift_spx  + sig_spx  * Z[i, p]

            voo_bucket   = max(voo_bucket  * np.exp(r_voo),  0.0)
            spxl_bucket  = max(spxl_bucket * np.exp(r_spxl), 0.0)
            voo_base     = max(voo_base    * np.exp(r_voo),  0.0)

            # update VOO price index + ATH
            voo_price *= np.exp(r_spx)
            voo_ath    = max(voo_ath, voo_price)

            total_val = voo_bucket + spxl_bucket
            spxl_frac = spxl_bucket / total_val if total_val > 0 else spxl_monthly_pct

            all_spring[i+1, p]       = total_val
            all_voo_baseline[i+1, p] = voo_base
            all_spxl_alloc[i+1, p]   = spxl_frac

            if spxl_frac > spxl_monthly_pct:
                total_shifted_days += 1

        shifted_frac[p] = total_shifted_days / days

    return {
        "voo_p50":        np.percentile(all_voo_baseline, 50, axis=1),
        "spring_p10":     np.percentile(all_spring, 10, axis=1),
        "spring_p25":     np.percentile(all_spring, 25, axis=1),
        "spring_p50":     np.percentile(all_spring, 50, axis=1),
        "spring_p75":     np.percentile(all_spring, 75, axis=1),
        "spring_p90":     np.percentile(all_spring, 90, axis=1),
        "spxl_alloc_p50": np.percentile(all_spxl_alloc, 50, axis=1),
        "shifted_pct":    float(np.median(shifted_frac) * 100),
    }
# =============================================================================
# MAIN
# =============================================================================
def main():
    st.title("VOO vs SPXL — Springboard P200 + Volatility Edge")
    st.caption(
        f"${monthly_inv}/month · {proj_years}yr projection · "
        f"{ann_ret*100:.1f}% return · {ann_vol*100:.1f}% vol · {leverage:.1f}x leverage"
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Historical Backtest",
        "🔮 Forward Projection",
        "🚀 Springboard P200",
        "📐 The Math",
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
            hist = run_historical_dca(prices, monthly_inv)
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

    # -------------------------------------------------------------------------
    # TAB 2: FORWARD PROJECTION (MONTE CARLO)
    # -------------------------------------------------------------------------
    with tab2:
        st.markdown("GBM Monte Carlo with Ito drag — P10 / P50 / P90 fan chart.")

        with st.spinner(f"Simulating {n_sim} paths × 2 instruments..."):
            voo_paths, voo_invested = dca_gbm_paths(proj_years, monthly_inv, ann_ret, ann_vol, voo_expense, 1, n_sim, seed=1)
            spxl_paths, spxl_invested = dca_gbm_paths(proj_years, monthly_inv, ann_ret, ann_vol, spxl_expense, leverage, n_sim, seed=2)

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
    # TAB 3: SPRINGBOARD — $90 SPXL / $10 VOO + Drawdown Rebalancer
    # -------------------------------------------------------------------------
    with tab3:
        st.markdown(
            "**Base DCA:** $90/month SPXL · $10/month VOO — always. "
            "**Drawdown rebalancer:** as VOO drops from its ATH, portfolio value + monthly contributions "
            "shift linearly from VOO → SPXL. Fully reverses as VOO recovers."
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("##### Base DCA split")
            spxl_pct = st.slider(
                "SPXL % of monthly DCA", 50, 100, 90, 5,
                help="Default: 90% SPXL / 10% VOO every month regardless of market conditions."
            )
            voo_pct = 100 - spxl_pct
            st.caption(f"→ SPXL ${monthly_inv * spxl_pct / 100:.0f} · VOO ${monthly_inv * voo_pct / 100:.0f} per month")

        with c2:
            st.markdown("##### Drawdown trigger levels")
            st.caption("Linear shift: VOO portfolio value moves to SPXL as price falls from ATH")
            max_dd_pct = st.slider(
                "Drawdown at 100% shift (%)", 20, 60, 40, 5,
                help="At this drawdown from ATH, 100% of VOO bucket value moves to SPXL."
            )
            # Show the linear table
            levels = [10, 20, 30, max_dd_pct]
            rows = ""
            for d in sorted(set(levels)):
                shift = min(d / max_dd_pct * 100, 100)
                spxl_eff = spxl_pct + (100 - spxl_pct) * (shift / 100)
                rows += "| -{}% | {:.0f}% VOO shifted | ~{:.0f}% SPXL |\n".format(d, shift, spxl_eff)
            st.markdown("| Drop from ATH | VOO shift | Effective SPXL |\n|---|---|---|\n" + rows)

        with c3:
            st.markdown("##### Simulation")
            sb_paths = st.select_slider("Monte Carlo paths", options=[100, 200, 300, 500], value=200)

        with st.spinner(f"Running {sb_paths} paths..."):
            sb = run_springboard(
                proj_years, monthly_inv, ann_ret, ann_vol,
                voo_expense, spxl_expense, leverage,
                spxl_monthly_pct=spxl_pct / 100,
                max_dd=max_dd_pct / 100,
                n_paths=sb_paths,
            )

        sp50     = float(sb["spring_p50"][-1])
        vp50     = float(sb["voo_p50"][-1])
        total_inv_sb = monthly_inv * (proj_years * 252 // 21)
        cagr_sb  = (sp50 / max(total_inv_sb, 1)) ** (1 / max(proj_years, 1)) - 1
        cagr_voo = (vp50 / max(total_inv_sb, 1)) ** (1 / max(proj_years, 1)) - 1

        k = st.columns(5)
        kpi(k[0], "Springboard P50",   fmt_currency(sp50),                          "median final value",          "#00ff9f")
        kpi(k[1], "Est. CAGR",         f"{cagr_sb*100:.1f}%",                        f"vs VOO-only {cagr_voo*100:.1f}%", "#00ffcc")
        kpi(k[2], "P10 Downside",      fmt_currency(float(sb["spring_p10"][-1])),    "worst 10%",                   "#f5a623")
        kpi(k[3], "P90 Upside",        fmt_currency(float(sb["spring_p90"][-1])),    "best 10%",                    "#00d4a0")
        kpi(k[4], "Days shifted >base",f"{sb['shifted_pct']:.0f}%",                  "extra SPXL vs base split",    "#ff4b4b")

        x_axis = np.linspace(0, proj_years, len(sb["spring_p50"]))

        # Fan chart
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=x_axis, y=sb["spring_p90"], fill=None, line=dict(width=0), showlegend=False))
        fig3.add_trace(go.Scatter(x=x_axis, y=sb["spring_p10"], fill="tonexty",
                                  fillcolor="rgba(0,255,159,0.08)", line=dict(width=0), name="P10–P90"))
        fig3.add_trace(go.Scatter(x=x_axis, y=sb["spring_p25"], fill=None, line=dict(width=0), showlegend=False))
        fig3.add_trace(go.Scatter(x=x_axis, y=sb["spring_p75"], fill="tonexty",
                                  fillcolor="rgba(0,255,159,0.15)", line=dict(width=0), name="P25–P75"))
        fig3.add_trace(go.Scatter(x=x_axis, y=sb["spring_p50"],
                                  name="Springboard P50", line=dict(color="#00ff9f", width=2.5)))
        fig3.add_trace(go.Scatter(x=x_axis, y=sb["voo_p50"],
                                  name="VOO-only baseline", line=dict(color="#888", width=1.5, dash="dash")))
        fig3.update_layout(
            template="plotly_dark", height=360,
            paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig3.update_xaxes(title_text="Year")
        fig3.update_yaxes(tickprefix="$")
        st.plotly_chart(fig3, use_container_width=True)

        # SPXL allocation over time
        st.markdown("##### SPXL allocation % over time (median path)")
        alloc_pct = sb["spxl_alloc_p50"] * 100
        fig_alloc = go.Figure()
        fig_alloc.add_trace(go.Scatter(
            x=x_axis, y=alloc_pct,
            fill="tozeroy", fillcolor="rgba(0,212,255,0.12)",
            line=dict(color="#00d4ff", width=1.5),
        ))
        fig_alloc.add_hline(y=spxl_pct, line=dict(color="#00ff9f", width=1, dash="dot"),
                            annotation_text=f"base {spxl_pct}%", annotation_position="right")
        fig_alloc.add_hline(y=100, line=dict(color="#ff4b4b", width=1, dash="dot"),
                            annotation_text="100% SPXL", annotation_position="right")
        fig_alloc.update_layout(
            template="plotly_dark", height=160,
            paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(range=[0, 105], ticksuffix="%"),
            showlegend=False,
        )
        fig_alloc.update_xaxes(title_text="Year")
        st.plotly_chart(fig_alloc, use_container_width=True)

        st.caption(
            f"During drawdowns the green SPXL allocation line spikes upward — "
            f"that's the rebalancer loading heavy for the recovery. "
            f"It gradually returns to the {spxl_pct}% base as VOO reclaims its ATH."
        )
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


if __name__ == "__main__":
    main()
