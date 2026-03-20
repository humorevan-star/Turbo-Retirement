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


def run_ath_rotation_backtest(
    prices: pd.DataFrame,
    monthly: float,
    spxl_base_pct: float = 0.90,
    spxl_floor: float = 0.10,
    ath_max: float = 0.15,
    dd_max: float = 0.40,
) -> pd.DataFrame:
    """
    Real historical backtest of the ATH Rotation strategy.
    Uses actual VOO + SPXL prices. Dollar-value portfolio, not shares.
    Each month: contribute monthly * target_spxl to SPXL bucket, rest to VOO bucket.
    Soft rebalance 20% of gap toward target each month.
    """
    voo_bucket  = 0.0
    spxl_bucket = 0.0
    voo_ath     = None   # ATH of VOO price
    last_month  = -1

    results = {"Date": [], "Strategy": [], "Invested": [], "SPXL_alloc": [],
               "VOO_pure": [], "SPXL_pure": []}
    voo_shares_pure  = 0.0
    spxl_shares_pure = 0.0
    invested = 0.0

    for date, row in prices.iterrows():
        voo_price  = float(row.get("VOO",  0))
        spxl_price = float(row.get("SPXL", 0))
        if voo_price <= 0 or spxl_price <= 0:
            continue

        # track ATH of VOO price
        if voo_ath is None:
            voo_ath = voo_price
        else:
            voo_ath = max(voo_ath, voo_price)

        dd          = max(0.0, (voo_ath - voo_price) / voo_ath)
        ath_premium = max(0.0, (voo_price - voo_ath) / voo_ath)  # always 0 since ath tracks max

        if dd > 0:
            t = min(dd / dd_max, 1.0)
            target_spxl = spxl_base_pct + t * (1.0 - spxl_base_pct)
        else:
            t = min(ath_premium / ath_max, 1.0)
            target_spxl = spxl_base_pct - t * (spxl_base_pct - spxl_floor)

        target_spxl = float(np.clip(target_spxl, spxl_floor, 1.0))

        m = date.month
        if m != last_month:
            # DCA at target ratio
            spxl_bucket += monthly * target_spxl
            voo_bucket  += monthly * (1.0 - target_spxl)
            invested    += monthly

            # pure benchmarks
            if voo_price  > 0: voo_shares_pure  += monthly / voo_price
            if spxl_price > 0: spxl_shares_pure += monthly / spxl_price

            # soft rebalance: close 20% of gap
            total = voo_bucket + spxl_bucket
            if total > 0:
                current_spxl = spxl_bucket / total
                gap   = target_spxl - current_spxl
                rebal = total * gap * 0.20
                # rebal is positive → move from VOO to SPXL
                # convert dollar rebal into shares
                if rebal > 0 and voo_bucket >= rebal:
                    spxl_bucket += rebal
                    voo_bucket  -= rebal
                elif rebal < 0 and spxl_bucket >= abs(rebal):
                    voo_bucket  += abs(rebal)
                    spxl_bucket -= abs(rebal)

            last_month = m

        # grow buckets daily by price change
        # (we track dollar value, so just mark-to-market via return)
        # use previous close to compute daily return
        pass  # price appreciation handled via portfolio value below

        total_val   = voo_bucket + spxl_bucket
        spxl_frac   = spxl_bucket / total_val if total_val > 0 else target_spxl

        results["Date"].append(date)
        results["Strategy"].append(round(total_val, 2))
        results["Invested"].append(round(invested, 2))
        results["SPXL_alloc"].append(round(spxl_frac * 100, 1))
        results["VOO_pure"].append(round(voo_shares_pure * voo_price, 2))
        results["SPXL_pure"].append(round(spxl_shares_pure * spxl_price, 2))

    df_out = pd.DataFrame(results).set_index("Date")

    # Apply daily returns to buckets properly using vectorised price returns
    # Re-run properly with daily price-return compounding
    voo_prices  = prices["VOO"].values
    spxl_prices = prices["SPXL"].values
    dates       = prices.index

    voo_b = 0.0; spxl_b = 0.0; voo_ath2 = None; lm = -1
    invested2 = 0.0
    out_strategy = np.zeros(len(dates))
    out_alloc    = np.zeros(len(dates))

    for i, date in enumerate(dates):
        vp = voo_prices[i]; sp = spxl_prices[i]
        if vp <= 0 or sp <= 0:
            out_strategy[i] = voo_b + spxl_b
            out_alloc[i]    = spxl_b / max(voo_b + spxl_b, 1)
            continue

        # daily price return
        if i > 0:
            voo_ret  = vp / voo_prices[i-1]  if voo_prices[i-1]  > 0 else 1.0
            spxl_ret = sp / spxl_prices[i-1] if spxl_prices[i-1] > 0 else 1.0
            voo_b  *= voo_ret
            spxl_b *= spxl_ret

        voo_ath2 = max(voo_ath2, vp) if voo_ath2 else vp
        dd2  = max(0.0, (voo_ath2 - vp) / voo_ath2)
        atp2 = 0.0  # vp always <= voo_ath2

        if dd2 > 0:
            t2 = min(dd2 / dd_max, 1.0)
            tgt = spxl_base_pct + t2 * (1.0 - spxl_base_pct)
        else:
            tgt = spxl_base_pct

        tgt = float(np.clip(tgt, spxl_floor, 1.0))

        if date.month != lm:
            spxl_b   += monthly * tgt
            voo_b    += monthly * (1.0 - tgt)
            invested2 += monthly
            total2 = voo_b + spxl_b
            if total2 > 0:
                cur = spxl_b / total2
                gap = tgt - cur
                rb  = total2 * gap * 0.20
                if rb > 0 and voo_b >= rb:
                    spxl_b += rb; voo_b -= rb
                elif rb < 0 and spxl_b >= abs(rb):
                    voo_b += abs(rb); spxl_b -= abs(rb)
            lm = date.month

        total2 = voo_b + spxl_b
        out_strategy[i] = total2
        out_alloc[i]    = spxl_b / total2 if total2 > 0 else tgt

    df2 = pd.DataFrame({
        "Strategy":  np.round(out_strategy, 2),
        "Invested":  0.0,  # filled below
        "SPXL_alloc": np.round(out_alloc * 100, 1),
        "VOO_pure":  df_out["VOO_pure"].values,
        "SPXL_pure": df_out["SPXL_pure"].values,
    }, index=dates)

    # rebuild invested
    inv = 0.0; lm2 = -1
    inv_arr = []
    for date in dates:
        if date.month != lm2:
            inv += monthly; lm2 = date.month
        inv_arr.append(inv)
    df2["Invested"] = inv_arr
    return df2


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
# SPRINGBOARD ENGINE — ATH Rotation + Drawdown Reload
# =============================================================================
# Core insight: SPXL's best risk/reward is BELOW ATH (recovery upside is huge).
#               SPXL's worst risk/reward is AT/ABOVE ATH (max vol drag, min upside).
#
# Rules:
#   1. Base DCA: spxl_base_pct% → SPXL, rest → VOO every month
#   2. ATH rotation (reduce SPXL as VOO makes new highs above prior ATH):
#      - Price at ATH              → hold at base split
#      - Price X% above prior ATH  → shift Y% of SPXL bucket → VOO (scale linearly)
#      - At ath_max% above ATH     → floor: spxl_floor% SPXL, rest VOO
#   3. Drawdown reload (increase SPXL as VOO falls from ATH):
#      - Price X% below ATH        → shift Z% of VOO bucket → SPXL (scale linearly)
#      - At dd_max% below ATH      → 100% SPXL
#   Monthly DCA contributions also follow the same ratio as the current allocation.
# =============================================================================

def run_springboard(
    years, monthly,
    ann_r, ann_s,
    voo_exp, spxl_exp, lev,
    spxl_base_pct=0.90,   # SPXL % at exactly ATH (neutral zone)
    spxl_floor=0.10,      # minimum SPXL % when well above ATH
    ath_max=0.15,         # how far above ATH before hitting spxl_floor
    dd_max=0.40,          # drawdown at which we hit 100% SPXL
    n_paths=200, seed=42,
):
    rng  = np.random.default_rng(seed)
    days = years * 252
    dt   = 1 / 252

    drag_spxl  = vol_drag(lev, ann_s)
    drift_voo  = (ann_r - voo_exp  - 0.5 * ann_s**2) * dt
    sig_voo    = ann_s * np.sqrt(dt)
    drift_spxl = (lev*ann_r - drag_spxl - spxl_exp - 0.5*(lev*ann_s)**2) * dt
    sig_spxl   = lev * ann_s * np.sqrt(dt)
    drift_idx  = (ann_r - 0.5 * ann_s**2) * dt   # clean index for ATH tracking
    sig_idx    = ann_s * np.sqrt(dt)

    all_spring     = np.zeros((days+1, n_paths))
    all_voo_base   = np.zeros((days+1, n_paths))
    all_spxl_alloc = np.zeros((days+1, n_paths))

    Z = rng.standard_normal((days, n_paths))

    for p in range(n_paths):
        voo_bucket  = 0.0
        spxl_bucket = 0.0
        voo_base    = 0.0   # pure VOO-only for comparison

        idx      = 1.0   # synthetic market index
        idx_ath  = 1.0   # all-time high of the index

        for i in range(days):
            # ── daily returns ────────────────────────────────────────────
            r_voo  = drift_voo  + sig_voo  * Z[i, p]
            r_spxl = drift_spxl + sig_spxl * Z[i, p]
            r_idx  = drift_idx  + sig_idx  * Z[i, p]

            # grow buckets
            voo_bucket  = max(voo_bucket  * np.exp(r_voo),  0.0)
            spxl_bucket = max(spxl_bucket * np.exp(r_spxl), 0.0)
            voo_base    = max(voo_base    * np.exp(r_voo),  0.0)

            # update index + ATH
            idx     *= np.exp(r_idx)
            idx_ath  = max(idx_ath, idx)

            # ── compute target SPXL allocation ──────────────────────────
            dd  = max(0.0, (idx_ath - idx) / idx_ath)   # 0 at ATH, positive below
            ath_premium = max(0.0, (idx - idx_ath) / idx_ath)  # 0 at/below ATH, positive above

            if dd > 0:
                # below ATH: ramp SPXL up linearly from base → 100%
                t = min(dd / dd_max, 1.0)
                target_spxl = spxl_base_pct + t * (1.0 - spxl_base_pct)
            else:
                # at or above ATH: ramp SPXL down linearly from base → floor
                t = min(ath_premium / ath_max, 1.0)
                target_spxl = spxl_base_pct - t * (spxl_base_pct - spxl_floor)

            target_spxl = float(np.clip(target_spxl, spxl_floor, 1.0))
            target_voo  = 1.0 - target_spxl

            # ── monthly: contribute + rebalance to target ────────────────
            if i > 0 and i % 21 == 0:
                # DCA at current target ratio
                spxl_bucket += monthly * target_spxl
                voo_bucket  += monthly * target_voo
                voo_base    += monthly

                # soft rebalance: move 20% of the gap toward target each month
                # (avoids tax/cost of hard daily rebalancing)
                total = voo_bucket + spxl_bucket
                if total > 0:
                    current_spxl_frac = spxl_bucket / total
                    gap = target_spxl - current_spxl_frac
                    rebal = total * gap * 0.20   # 20% of gap closed each month
                    spxl_bucket = max(spxl_bucket + rebal, 0.0)
                    voo_bucket  = max(voo_bucket  - rebal, 0.0)

            total = voo_bucket + spxl_bucket
            spxl_frac = spxl_bucket / total if total > 0 else target_spxl

            all_spring[i+1, p]     = total
            all_voo_base[i+1, p]   = voo_base
            all_spxl_alloc[i+1, p] = spxl_frac

    return {
        "voo_p50":        np.percentile(all_voo_base,   50, axis=1),
        "spring_p10":     np.percentile(all_spring,     10, axis=1),
        "spring_p25":     np.percentile(all_spring,     25, axis=1),
        "spring_p50":     np.percentile(all_spring,     50, axis=1),
        "spring_p75":     np.percentile(all_spring,     75, axis=1),
        "spring_p90":     np.percentile(all_spring,     90, axis=1),
        "spxl_alloc_p50": np.percentile(all_spxl_alloc, 50, axis=1),
        "shifted_pct":    float(np.mean(np.percentile(all_spxl_alloc, 50, axis=1)) * 100),
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

                st.markdown("---")
                st.markdown("### ATH Rotation strategy — real historical backtest")
                st.caption("Same prices, different strategy: shift VOO value to SPXL during drawdowns, reduce SPXL when above ATH.")

                rot_c1, rot_c2, rot_c3, rot_c4 = st.columns(4)
                with rot_c1:
                    rot_base  = st.slider("SPXL % at ATH", 50, 100, 90, 5, key="rot_base")
                with rot_c2:
                    rot_floor = st.slider("Min SPXL % above ATH", 0, 40, 10, 5, key="rot_floor")
                with rot_c3:
                    rot_ath   = st.slider("% above ATH → floor", 5, 30, 15, 5, key="rot_ath")
                with rot_c4:
                    rot_dd    = st.slider("% drawdown → 100% SPXL", 15, 60, 40, 5, key="rot_dd")

                rot = run_ath_rotation_backtest(
                    prices, monthly_inv,
                    spxl_base_pct = rot_base  / 100,
                    spxl_floor    = rot_floor / 100,
                    ath_max       = rot_ath   / 100,
                    dd_max        = rot_dd    / 100,
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
    # TAB 3: SPRINGBOARD — ATH Rotation + Drawdown Reload
    # -------------------------------------------------------------------------
    with tab3:
        st.markdown(
            "**ATH Rotation strategy** — SPXL has the best risk/reward *below* its ATH (recovery upside). "
            "Reduce SPXL as VOO makes new highs (high vol drag, low upside). "
            "Reload SPXL hard on drawdowns (maximum recovery leverage)."
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("##### SPXL at ATH (neutral zone)")
            spxl_base = st.slider("SPXL % when at ATH", 50, 100, 90, 5,
                help="Your base allocation when the market is exactly at its all-time high.")
            spxl_floor = st.slider("Min SPXL % (well above ATH)", 0, 50, 10, 5,
                help="Floor — how little SPXL you hold when market is well extended above ATH.")
            ath_max = st.slider("% above ATH to hit floor", 5, 30, 15, 5,
                help="How far above ATH before SPXL allocation reaches the floor.")

        with c2:
            st.markdown("##### Drawdown reload")
            dd_max = st.slider("Drawdown → 100% SPXL (%)", 15, 60, 40, 5,
                help="How far below ATH before allocation reaches 100% SPXL.")

            # live preview table
            st.markdown("**Allocation preview:**")
            preview_rows = ""
            for label, dd, ath in [
                ("Well above ATH +{}%".format(ath_max), 0, ath_max/100),
                ("At ATH", 0, 0),
                ("-10% from ATH", 10, 0),
                ("-20% from ATH", 20, 0),
                ("-{}% from ATH".format(dd_max), dd_max, 0),
            ]:
                if ath > 0:
                    t = min(ath / (ath_max/100), 1.0)
                    spxl = spxl_base - t * (spxl_base - spxl_floor)
                else:
                    t = min((dd/100) / (dd_max/100), 1.0)
                    spxl = spxl_base + t * (100 - spxl_base)
                spxl = int(np.clip(spxl, spxl_floor, 100))
                preview_rows += "| {} | {}% SPXL / {}% VOO |\n".format(label, spxl, 100-spxl)
            st.markdown("| Market level | Allocation |\n|---|---|\n" + preview_rows)

        with c3:
            st.markdown("##### Simulation")
            sb_paths = st.select_slider("Monte Carlo paths", options=[100, 200, 300, 500], value=200)

        with st.spinner(f"Running {sb_paths} paths..."):
            sb = run_springboard(
                proj_years, monthly_inv, ann_ret, ann_vol,
                voo_expense, spxl_expense, leverage,
                spxl_base_pct = spxl_base  / 100,
                spxl_floor    = spxl_floor / 100,
                ath_max       = ath_max    / 100,
                dd_max        = dd_max     / 100,
                n_paths       = sb_paths,
            )

        sp50      = float(sb["spring_p50"][-1])
        vp50      = float(sb["voo_p50"][-1])
        total_inv_sb = monthly_inv * (proj_years * 252 // 21)
        cagr_sb   = (sp50 / max(total_inv_sb, 1)) ** (1 / max(proj_years, 1)) - 1
        cagr_voo  = (vp50 / max(total_inv_sb, 1)) ** (1 / max(proj_years, 1)) - 1
        avg_spxl  = float(np.mean(sb["spxl_alloc_p50"]) * 100)

        k = st.columns(5)
        kpi(k[0], "Springboard P50",  fmt_currency(sp50),                        "median final value",            "#00ff9f")
        kpi(k[1], "Est. CAGR",        f"{cagr_sb*100:.1f}%",                     f"vs VOO-only {cagr_voo*100:.1f}%", "#00ffcc")
        kpi(k[2], "P10 Downside",     fmt_currency(float(sb["spring_p10"][-1])), "worst 10%",                     "#f5a623")
        kpi(k[3], "P90 Upside",       fmt_currency(float(sb["spring_p90"][-1])), "best 10%",                      "#00d4a0")
        kpi(k[4], "Avg SPXL alloc",   f"{avg_spxl:.0f}%",                        "median path average",           "#ff4b4b")

        x_axis = np.linspace(0, proj_years, len(sb["spring_p50"]))

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=x_axis, y=sb["spring_p90"], fill=None, line=dict(width=0), showlegend=False))
        fig3.add_trace(go.Scatter(x=x_axis, y=sb["spring_p10"], fill="tonexty",
                                  fillcolor="rgba(0,255,159,0.08)", line=dict(width=0), name="P10–P90"))
        fig3.add_trace(go.Scatter(x=x_axis, y=sb["spring_p25"], fill=None, line=dict(width=0), showlegend=False))
        fig3.add_trace(go.Scatter(x=x_axis, y=sb["spring_p75"], fill="tonexty",
                                  fillcolor="rgba(0,255,159,0.16)", line=dict(width=0), name="P25–P75"))
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

        st.markdown("##### SPXL allocation % over time (median path)")
        alloc_pct = sb["spxl_alloc_p50"] * 100
        fig_alloc = go.Figure()
        fig_alloc.add_trace(go.Scatter(
            x=x_axis, y=alloc_pct,
            fill="tozeroy", fillcolor="rgba(0,212,255,0.12)",
            line=dict(color="#00d4ff", width=1.5),
        ))
        fig_alloc.add_hline(y=spxl_base,  line=dict(color="#00ff9f", width=1, dash="dot"),
                            annotation_text=f"ATH base {spxl_base}%",  annotation_position="right")
        fig_alloc.add_hline(y=spxl_floor, line=dict(color="#888",    width=1, dash="dot"),
                            annotation_text=f"floor {spxl_floor}%",    annotation_position="right")
        fig_alloc.add_hline(y=100,        line=dict(color="#ff4b4b",  width=1, dash="dot"),
                            annotation_text="100% SPXL",               annotation_position="right")
        fig_alloc.update_layout(
            template="plotly_dark", height=170,
            paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(range=[0, 105], ticksuffix="%"),
            showlegend=False,
        )
        fig_alloc.update_xaxes(title_text="Year")
        st.plotly_chart(fig_alloc, use_container_width=True)

        st.caption(
            "Allocation dips toward the floor during extended bull runs (high ATH premium = high vol drag risk). "
            "Spikes to 100% during crashes — exactly when 3x leverage earns the most on recovery. "
            "Soft monthly rebalancing (20% of gap) avoids whipsaw from daily rebalancing."
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
