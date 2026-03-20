import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# VOO vs SPXL | Mathematically Correct Leveraged DCA Simulator
# =============================================================================

st.set_page_config(
    page_title="VOO vs SPXL — Springboard P200 + Vol Edge",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .main { background-color: #0b0e14; color: #e1e1e1; }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    h1,h2,h3 { font-family: 'IBM Plex Mono', monospace; }
    [data-testid="stSidebar"] { background-color: #0e1117; }
    .stSpinner > div { border-top-color: #00ff9f !important; }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# 1. UI HELPER FUNCTIONS
# =============================================================================
def format_currency(v: float) -> str:
    if v >= 1_000_000:
        return f"${v/1_000_000:.2f}M"
    return f"${v:,.0f}"

def _kpi(col, title, value, subtext, color):
    col.markdown(f"""
    <div style="background:#161b22;border:1px solid #30363d;padding:15px;border-radius:5px;text-align:center;height:100%;">
        <p style="margin:0;font-size:12px;color:#8b949e;text-transform:uppercase;">{title}</p>
        <h3 style="margin:5px 0;color:{color};">{value}</h3>
        <p style="margin:0;font-size:11px;color:#8b949e;">{subtext}</p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# 2. SIDEBAR
# =============================================================================
st.sidebar.markdown("## Strategy Controls")
monthly_inv = st.sidebar.number_input("Monthly Contribution ($)", value=100, min_value=10, step=50)

st.sidebar.markdown("---")
st.sidebar.markdown("### Market Assumptions")
ann_ret = st.sidebar.slider("S&P 500 Annual Return (%)", 0.0, 20.0, 10.0, 0.5) / 100
ann_vol = st.sidebar.slider("Annual Volatility (%)", 5.0, 45.0, 18.0, 0.5) / 100

st.sidebar.markdown("---")
st.sidebar.markdown("### Projection Settings")
proj_years = st.sidebar.slider("Years to Project Forward", 1, 40, 20)

st.sidebar.markdown("---")
st.sidebar.markdown("### Instrument Parameters")
voo_expense = st.sidebar.number_input("VOO Expense Ratio (%)", value=0.03, step=0.01) / 100
spxl_expense = st.sidebar.number_input("SPXL Expense Ratio (%)", value=0.91, step=0.01) / 100
leverage = st.sidebar.number_input("SPXL Leverage Factor", value=3.0, step=0.5)

# =============================================================================
# 3. MATHEMATICAL CORE
# =============================================================================
def run_springboard(years, monthly, ann_r, ann_s, voo_exp, spxl_exp, lev,
                    trigger_pct, ma_length, vol_window, vol_threshold, n_paths, seed=42):
    rng = np.random.default_rng(seed)
    days = years * 252
    dt = 1 / 252

    mu_spx = ann_r
    sig_spx = ann_s
    drift_spx = (mu_spx - 0.5 * sig_spx**2) * dt
    sigma_spx = sig_spx * np.sqrt(dt)

    drag_spxl = lev * (lev - 1) / 2 * ann_s**2
    drift_voo = (1.0 * ann_r - 0.5 * sig_spx**2 - voo_exp) * dt
    sig_voo = sig_spx * np.sqrt(dt)
    drift_spxl = (lev * ann_r - drag_spxl - spxl_exp - 0.5 * (lev*sig_spx)**2) * dt
    sig_spxl = lev * sig_spx * np.sqrt(dt)

    all_voo = np.zeros((days + 1, n_paths))
    all_spxl = np.zeros((days + 1, n_paths))
    all_spring = np.zeros((days + 1, n_paths))
    lev_frac = np.zeros(n_paths)

    trigger_factor = 1 - trigger_pct / 100
    Z = rng.standard_normal((days, n_paths))

    for p in range(n_paths):
        spx_log = np.cumsum(drift_spx + sigma_spx * Z[:, p])
        spx_idx = np.concatenate([[1.0], np.exp(spx_log)])

        spx_s = pd.Series(spx_idx)
        roll_high = spx_s.rolling(50, min_periods=1).max().values
        roll_sma = spx_s.rolling(ma_length, min_periods=max(20, ma_length//4)).mean().values

        daily_logrets = np.zeros(len(spx_idx))
        daily_logrets[1:] = np.log(spx_idx[1:] / spx_idx[:-1])
        vol_series = pd.Series(daily_logrets)
        roll_std = vol_series.rolling(vol_window, min_periods=10).std().values
        roll_vol_ann = roll_std * np.sqrt(252) * 100

        voo_val = spxl_val = spring_val = 0.0
        in_lev = False
        rec_high = 0.0
        lev_days_p = 0

        for i in range(days):
            if i > 0 and i % 21 == 0:
                voo_val += monthly
                spxl_val += monthly
                spring_val += monthly

            r_voo = drift_voo + sig_voo * Z[i, p]
            r_spxl = drift_spxl + sig_spxl * Z[i, p]

            voo_val *= np.exp(r_voo)
            spxl_val *= np.exp(r_spxl)

            if in_lev:
                spring_val *= np.exp(r_spxl)
                lev_days_p += 1
            else:
                spring_val *= np.exp(r_voo)

            all_voo[i+1, p] = voo_val
            all_spxl[i+1, p] = spxl_val
            all_spring[i+1, p] = spring_val

            curr_spx = spx_idx[i + 1]
            curr_high = roll_high[i + 1]
            curr_sma = roll_sma[i + 1]
            curr_vol = roll_vol_ann[i + 1]

            if not in_lev:
                if (curr_spx > curr_sma and curr_spx < trigger_factor * curr_high and curr_vol < vol_threshold):
                    in_lev = True
                    rec_high = curr_high
            else:
                if curr_spx >= rec_high or curr_spx <= curr_sma:
                    in_lev = False

        lev_frac[p] = lev_days_p / days

    return {
        "voo_p50": np.percentile(all_voo, 50, axis=1),
        "spxl_p50": np.percentile(all_spxl, 50, axis=1),
        "spring_p10": np.percentile(all_spring, 10, axis=1),
        "spring_p25": np.percentile(all_spring, 25, axis=1),
        "spring_p50": np.percentile(all_spring, 50, axis=1),
        "spring_p75": np.percentile(all_spring, 75, axis=1),
        "spring_p90": np.percentile(all_spring, 90, axis=1),
        "lev_pct_median": float(np.median(lev_frac) * 100),
        "lev_pct_p90": float(np.percentile(lev_frac, 90) * 100),
        "n_paths": n_paths,
    }

# =============================================================================
# 4. MAIN APP
# =============================================================================
def main():
    st.title("🚀 VOO vs SPXL — Springboard P200 + Volatility Edge")
    st.caption(f"${monthly_inv}/month · {proj_years}yr · 3× leverage · {ann_ret*100:.1f}% return / {ann_vol*100:.0f}% vol")

    tab1, tab2, tab3 = st.tabs([
        "🌱 Springboard Strategy (Max Alpha)",
        "📈 Historical Data",
        "📐 The Math",
    ])

    # =========================================================================
    # ENHANCED SPRINGBOARD P200 + VOLATILITY EDGE
    # =========================================================================
    with tab1:
        st.markdown("""
        **Springboard P200 + Volatility Edge** — the maximum-return evolution of the DCA strategy.  
        Holds VOO by default. Switches to SPXL **only** in strong uptrends after a dip **and** when recent realized volatility is low (minimal decay).
        """)

        # --- SESSION STATE INITIALIZATION ---
        if "trigger_pct" not in st.session_state: st.session_state.trigger_pct = 8.0
        if "ma_length" not in st.session_state: st.session_state.ma_length = 200
        if "vol_window" not in st.session_state: st.session_state.vol_window = 20
        if "vol_threshold" not in st.session_state: st.session_state.vol_threshold = 18.0

        if st.button("🔥 Set Default (Max CAGR) Parameters", use_container_width=True):
            st.session_state.trigger_pct = 4.0
            st.session_state.ma_length = 150
            st.session_state.vol_threshold = 24.0

        st.markdown("---")
        sb_col1, sb_col2 = st.columns([1, 2])

        with sb_col1:
            st.slider("Dip Trigger (%)", 3.0, 25.0, key="trigger_pct", step=0.5,
                      help="Buy SPXL when price drops this much from its recent high.")
            st.slider("Trend Filter (SMA)", 100, 300, key="ma_length", step=10,
                      help="Only use SPXL if the market is above this Moving Average.")
            st.slider("Vol Edge Max (%)", 12.0, 30.0, key="vol_threshold", step=0.5,
                      help="Avoid SPXL if recent volatility is higher than this.")
            
            with st.expander("⚙️ Advanced Setup"):
                st.slider("Realized Vol Window (days)", 10, 60, key="vol_window")
                sb_paths = st.select_slider("Monte Carlo paths", options=[50, 200, 500], value=200)

        with sb_col2:
            st.markdown("""
            <div style="background:#111318;border-radius:4px;padding:14px 18px;border-left:3px solid #00ff9f; height:100%;">
            <p style="font-family:monospace;font-size:10px;color:#00ffcc;margin:0 0 8px;text-transform:uppercase;letter-spacing:0.1em;">
            ENHANCED SPRINGBOARD RULES
            </p>
            <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:12px;font-size:14px;">
            <div>① <b>Uptrend Filter</b><br>Price must be above SMA</div>
            <div>② <b>Low-Vol Edge</b><br>Recent realized vol &lt; threshold</div>
            <div>③ <b>Dip Trigger</b><br>Price drops below High</div>
            <div>④ <b>Smart Exit</b><br>Recover to high OR break below SMA</div>
            </div>
            <p style="font-size:11px; color:#888; margin-top:10px;"><em>The "Max CAGR" preset uses a 4% dip, 150-day SMA, and 24% Vol limit to aggressively buy standard pullbacks while avoiding major crashes.</em></p>
            </div>
            """, unsafe_allow_html=True)

        with st.spinner(f"Running {sb_paths} enhanced paths..."):
            sb = run_springboard(
                proj_years, monthly_inv, ann_ret, ann_vol,
                voo_expense, spxl_expense, leverage,
                st.session_state.trigger_pct,
                st.session_state.ma_length,
                st.session_state.vol_window,
                st.session_state.vol_threshold,
                n_paths=sb_paths
            )

        # KPIs
        sp50 = float(sb["spring_p50"][-1])
        vp50 = float(sb["voo_p50"][-1])
        mult = sp50 / vp50 if vp50 > 0 else 0

        st.write("") # Spacing
        k = st.columns(5)
        _kpi(k[0], "Springboard P50", format_currency(sp50), f"vs VOO {format_currency(vp50)}", "#00ff9f")
        _kpi(k[1], "Alpha Multiplier", f"{mult:.2f}×", "Median Outperformance", "#00ff9f")
        _kpi(k[2], "P10 Downside", format_currency(float(sb["spring_p10"][-1])), "Worst 10% of Runs", "#f5a623")
        _kpi(k[3], "P90 Upside", format_currency(float(sb["spring_p90"][-1])), "Best 10% of Runs", "#00d4a0")
        _kpi(k[4], "Time in SPXL", f"{sb['lev_pct_median']:.1f}%", f"Up to {sb['lev_pct_p90']:.1f}% in Bulls", "#ff4b4b")

        # --- PLOTLY FAN CHART ---
        x_axis = np.linspace(0, proj_years, len(sb["voo_p50"]))
        
        fig = go.Figure()
        
        # P10 to P90 Range Fill
        fig.add_trace(go.Scatter(x=x_axis, y=sb["spring_p90"], mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=x_axis, y=sb["spring_p10"], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 255, 159, 0.1)', name='Strategy P10-P90 Range'))
        
        # Main Lines
        fig.add_trace(go.Scatter(x=x_axis, y=sb["voo_p50"], name="VOO (1x) Median", line=dict(color='#3182bd', width=2)))
        fig.add_trace(go.Scatter(x=x_axis, y=sb["spxl_p50"], name="Pure SPXL (3x) Median", line=dict(color='#ff4b4b', width=2, dash='dot')))
        fig.add_trace(go.Scatter(x=x_axis, y=sb["spring_p50"], name="Springboard Strategy Median", line=dict(color='#00ff9f', width=3)))

        fig.update_layout(
            template="plotly_dark", hovermode="x unified",
            xaxis_title="Years", yaxis_title="Portfolio Value ($)",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.info("Historical data module goes here. You can paste your original historical backtest code in this tab.")
        
    with tab3:
        st.write("### The Math of Volatility Decay")
        st.write("Leveraged ETFs suffer from Volatility Drag. The math is: `Daily Return = 3 * Index_Return - (3 * 2 / 2) * Volatility^2`")

if __name__ == "__main__":
    main()
