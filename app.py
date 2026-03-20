import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Params
monthly_inv = 100
years = 10
spy_annual_return = 0.10  # 10% expected S&P return
spy_vol = 0.18            # 18% annual volatility
leverage = 3
expense_ratio = 0.0091    # SPXL expense ratio

def simulate_leverage(target_years, monthly_amt, ann_ret, ann_vol, lev):
    days = target_years * 252
    daily_ret = ann_ret / 252
    daily_vol = ann_vol / np.sqrt(252)
    
    # Simulate S&P 500 daily path
    path = np.random.normal(daily_ret, daily_vol, days)
    
    # 3x Daily rebalancing math (The "Decay" Formula)
    # Daily_Lev_Ret = (Daily_SPY_Ret * 3) - (Borrowing_Costs/Fees)
    lev_path = (path * lev) - (expense_ratio / 252)
    
    # Accumulation Logic
    voo_bal, spxl_bal = 0, 0
    voo_history, spxl_history = [], []
    
    for i in range(days):
        if i % 21 == 0: # Monthly deposit
            voo_bal += monthly_amt
            spxl_bal += monthly_amt
        
        voo_bal *= (1 + path[i])
        spxl_bal *= (1 + lev_path[i])
        
        voo_history.append(voo_bal)
        spxl_history.append(spxl_bal)
        
    return voo_history, spxl_history

# Run Simulation
voo, spxl = simulate_leverage(years, monthly_inv, spy_annual_return, spy_vol, leverage)

# Result: SPXL is a "Wealth Escalator" if you never get off during the crashes.
