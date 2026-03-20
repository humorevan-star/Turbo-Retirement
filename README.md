# VOO vs SPXL — Springboard Simulator

A mathematically rigorous leveraged DCA simulator comparing VOO and SPXL, featuring:

- **Historical Backtest** — real DCA using Yahoo Finance prices (2010–today) with buy/sell signals
- **Forward Projection** — Monte Carlo fan chart (P10/P50/P90) using Geometric Brownian Motion
- **Springboard P200** — adaptive signal engine: MA200 trend filter + volatility gate + momentum + pullback re-entry + Turbo DCA multiplier
- **The Math** — Ito's lemma vol drag, GBM, break-even volatility, drawdown tables

## Math

The simulator uses mathematically correct Ito drag:

```
Vol Drag = L(L-1)/2 × σ²
Net SPXL return = L × r_SPX − Vol Drag − expense ratio
GBM: S(t+dt) = S(t) × exp((μ − σ²/2)dt + σ√dt × Z)
```

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/voo-spxl-simulator
cd voo-spxl-simulator
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → set `app.py` as the entry point
4. Deploy — free hosting, auto-updates on push

## Parameters (Sidebar)

| Parameter | Default | Description |
|-----------|---------|-------------|
| Monthly Contribution | $500 | DCA amount per month |
| S&P 500 Annual Return | 10% | Expected annual return |
| Annual Volatility | 18% | Expected annualized vol |
| Years to Project | 20 | Simulation horizon |
| Monte Carlo Paths | 300 | Number of GBM paths |
| VOO Expense Ratio | 0.03% | VOO annual fee |
| SPXL Expense Ratio | 0.91% | SPXL annual fee |
| Leverage Factor | 3.0x | SPXL daily reset leverage |

## Springboard Signal Logic

```python
# Enter SPXL when ALL true:
price > MA(200)           # above long-term trend
realized_vol < threshold  # low volatility regime
momentum(10d) > 0         # positive short-term momentum

# OR re-enter on pullback:
price < rolling_peak * (1 - pullback_pct)

# Boost DCA while in SPXL:
contribution = monthly * dca_multiplier  # default 1.5x

# Exit when signal breaks → fall back to VOO
```

## Disclaimer

This is an educational simulation tool. Past performance does not guarantee future results. Leveraged ETFs carry significant risk of permanent capital loss, especially in high-volatility environments. Not financial advice.

## License

MIT
