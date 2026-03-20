import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. PAGE CONFIG & STYLING
# ==========================================
st.set_page_config(page_title="Leveraged DCA Simulator", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 10px; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SIDEBAR PARAMETERS
# ==========================================
st.sidebar.header("🕹️ Strategy Controls")
monthly_inv = st.sidebar.number_input("Monthly Contribution ($)", value=100, step=50)
projection_years = st.sidebar.slider("Years to Project", 1, 20, 10)

st.sidebar.subheader("Market Assumptions")
expected_ret = st.sidebar.slider("S&P 500 Annual Return (%)", 0.0, 20.0, 10.0) / 100
market_vol = st.sidebar.slider("Market Volatility (%)", 5.0, 50.0, 18.0) / 100

# ==========================================
# 3. THE SIMULATION ENGINE
# ==========================================
def run_simulation(years, monthly_amt, ann_ret, ann_vol):
    days = years * 252
    months = years * 12
    
    # Daily parameters
    daily_ret = ann_ret / 252
    daily_vol = ann_vol / np.sqrt(252)
    
    # 1. Simulate S&P 500 Daily Returns (Normal Distribution)
    daily_returns_1x = np.random.normal(daily_ret, daily_vol, days)
    
    # 2. Simulate SPXL (3x) Returns with Decay & Fees
    # SPXL Expense Ratio is approx 0.91%
    expense_ratio = 0.0091 / 252
    daily_returns_3x = (daily_returns_1x * 3) - expense_ratio
    
    # Accumulation Logic
    voo_vals = [0]
    spxl_vals = [0]
    current_voo = 0
    current_spxl = 0
    
    for i in range(days):
        # Add monthly contribution every 21 trading days
        if i % 21 == 0:
            current_voo += monthly_amt
            current_spxl += monthly_amt
            
        current_voo *= (1 + daily_returns_1x[i])
        current_spxl *= (1 + daily_returns_3x[i])
        
        voo_vals.append(current_voo)
        spxl_vals.append(current_spxl)
        
    return voo_vals, spxl_vals

# Execute Math
voo_path, spxl_path = run_simulation(projection_years, monthly_inv, expected_ret, market_vol)

# ==========================================
# 4. DASHBOARD UI
# ==========================================
st.title("🚀 SPXL vs VOO: The 10-Year Leveraged Bet")
st.markdown(f"Accumulating **${monthly_inv}/month** over **{projection_years} years**.")

# Metric Row
c1, c2, c3 = st.columns(3)
voo_final = voo_path[-1]
spxl_final = spxl_path[-1]
multiple = spxl_final / voo_final if voo_final > 0 else 0

with c1:
    st.metric("VOO Final (1x)", f"${voo_final:,.0f}")
with c2:
    st.metric("SPXL Final (3x)", f"${spxl_final:,.0f}", f"{multiple:.1f}x VOO")
with c3:
    total_invested = monthly_inv * 12 * projection_years
    st.metric("Total Principal", f"${total_invested:,.0f}")

# Plotly Chart
fig = go.Figure()
fig.add_trace(go.Scatter(y=voo_path, name="VOO (S&P 500)", line=dict(color='#3182bd', width=2)))
fig.add_trace(go.Scatter(y=spxl_path, name="SPXL (3x Leveraged)", line=dict(color='#ff4b4b', width=3)))

fig.update_layout(
    template="plotly_dark",
    hovermode="x unified",
    xaxis_title="Trading Days",
    yaxis_title="Portfolio Value ($)",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    margin=dict(l=20, r=20, t=20, b=20)
)

st.plotly_chart(fig, use_container_width=True)

# Warning Section
st.warning("⚠️ **Volatility Decay Warning:** Leveraged ETFs are designed for daily rebalancing. In sideways markets, SPXL can lose value even if the S&P 500 is flat.")
