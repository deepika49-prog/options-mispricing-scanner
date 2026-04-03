# Options Mispricing Scanner

A live options pricing engine that uses **Black-Scholes** and a smoothed **volatility surface** to scan market contracts for mispricing opportunities.

## What it does
- Fetches real-time options chains via yfinance (2000+ contracts per ticker)
- Builds a 3D implied volatility surface using scipy spline interpolation
- Computes theoretical fair prices and Greeks (delta, gamma, vega, theta)
- Applies liquidity filters and ranks contracts by mispricing edge
- Displays everything on an interactive Plotly Dash dashboard

## Tech stack
Python, Black-Scholes, yfinance, FRED API, scipy, pandas, Plotly Dash

## How to run
```bash
pip install -r requirements.txt
python3 dashboard.py
```
Then open http://127.0.0.1:8050
