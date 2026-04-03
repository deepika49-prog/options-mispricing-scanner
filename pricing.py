# pricing.py


import numpy as np
import pandas as pd
from scipy.interpolate import SmoothBivariateSpline
from scipy.stats import norm
from datetime import datetime, date
from loguru import logger


# Black-Scholes implementation

def black_scholes_price(S, K, T, r, sigma, q, option_type):
    """
    Classic Black-Scholes formula.

    Parameters:
        S     : spot price (current stock price)
        K     : strike price
        T     : time to expiry in years (e.g. 0.25 = 3 months)
        r     : risk-free rate as decimal (e.g. 0.05 = 5%)
        sigma : volatility as decimal (e.g. 0.20 = 20%)
        q     : dividend yield as decimal
        option_type : 'call' or 'put'

    Returns:
        Theoretical option price
    """
    if T <= 0 or sigma <= 0:
        return np.nan

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    return price


def compute_greeks(S, K, T, r, sigma, q, option_type):
    """
    Computes the main options Greeks.

    Delta  : how much the option price moves per $1 move in the stock
    Gamma  : how much delta changes per $1 move in the stock
    Vega   : how much the option price moves per 1% change in volatility
    Theta  : how much value the option loses per day (time decay)

    Returns a dict of the four Greeks.
    """
    if T <= 0 or sigma <= 0:
        return {"delta": np.nan, "gamma": np.nan, "vega": np.nan, "theta": np.nan}

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100  # per 1% vol move

    if option_type == "call":
        delta = np.exp(-q * T) * norm.cdf(d1)
        theta = (
            -np.exp(-q * T) * S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
            + q * S * np.exp(-q * T) * norm.cdf(d1)
        ) / 365  # per day
    else:
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        theta = (
            -np.exp(-q * T) * S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
            - q * S * np.exp(-q * T) * norm.cdf(-d1)
        ) / 365

    return {"delta": round(delta, 4), "gamma": round(gamma, 6),
            "vega": round(vega, 4), "theta": round(theta, 4)}


# Volatility surface

def build_vol_surface(df):
    """
    Instead of using each contract's raw implied vol (which is noisy),
    we fit a smooth mathematical surface through all the IV data points.

    Think of it like fitting a curve through a scatter plot — we get a
    smooth function that gives us a 'cleaned up' IV for any strike/expiry.

    Uses scipy's SmoothBivariateSpline, which fits a 2D smooth surface
    through (moneyness, time_to_expiry) → implied_vol data points.

    Returns a fitted spline object, or None if fitting fails.
    """
    # Filter to contracts with valid IV data and enough liquidity
    surface_data = df[
        (df["market_iv"] > 0.01) &
        (df["market_iv"] < 5.0) &   # remove obviously bad IV values
        (df["open_interest"] > 10) &
        (df["time_to_expiry"] > 0.02)  # at least ~1 week out
    ].copy()

    if len(surface_data) < 20:
        logger.warning("Not enough data points to build vol surface — using raw IVs")
        return None

    try:
        spline = SmoothBivariateSpline(
            x=surface_data["moneyness"].values,       # log(K/S) — how far from ATM
            y=surface_data["time_to_expiry"].values,  # years to expiry
            z=surface_data["market_iv"].values,
            kx=3, ky=3,  # cubic spline in both dimensions
            s=len(surface_data) * 0.1  # smoothing factor
        )
        logger.info(f"Vol surface built from {len(surface_data)} data points")
        return spline
    except Exception as e:
        logger.warning(f"Vol surface fitting failed: {e}. Using raw IVs.")
        return None


def get_smoothed_iv(spline, moneyness, time_to_expiry, raw_iv):
    """
    Gets the smoothed IV from the vol surface spline.
    Falls back to raw IV if the spline isn't available or gives a bad value.
    """
    if spline is None:
        return raw_iv
    try:
        smoothed = float(spline(moneyness, time_to_expiry))
        # Sanity check: smoothed IV should be positive and reasonable
        if 0.01 < smoothed < 5.0:
            return smoothed
        return raw_iv
    except Exception:
        return raw_iv


# Main pricing function

def price_options(data: dict, risk_free_rate: float) -> pd.DataFrame:
    """
    Takes the raw data dict from data.py and computes:
      - time_to_expiry for each contract
      - moneyness (how far the strike is from spot)
      - smoothed IV from the vol surface
      - theoretical price from Black-Scholes
      - all four Greeks

    Returns an enriched DataFrame ready for signal analysis.
    """
    df = data["options"].copy()
    S = data["spot_price"]
    q = data["dividend_yield"]
    ticker = data["ticker"]

    today = date.today()

    # Time to expiry
    # Convert expiry date string to a decimal in years
    def calc_tte(expiry_str):
        expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        days = (expiry_date - today).days
        return max(days / 365.0, 0.0)

    df["time_to_expiry"] = df["expiry"].apply(calc_tte)

    # Moneyness
    # log(K/S): 0 means at-the-money, positive = OTM call / ITM put, negative = ITM call / OTM put
    df["moneyness"] = np.log(df["strike"] / S)

    # Build vol surface 
    vol_surface = build_vol_surface(df)

    # Smooth IV per contract
    df["smoothed_iv"] = df.apply(
        lambda row: get_smoothed_iv(vol_surface, row["moneyness"],
                                    row["time_to_expiry"], row["market_iv"]),
        axis=1
    )

    # Black-Scholes theoretical price
    df["model_price"] = df.apply(
        lambda row: black_scholes_price(
            S=S,
            K=row["strike"],
            T=row["time_to_expiry"],
            r=risk_free_rate,
            sigma=row["smoothed_iv"],
            q=q,
            option_type=row["type"]
        ),
        axis=1
    )

    # Greeks
    greeks_list = df.apply(
        lambda row: compute_greeks(
            S=S, K=row["strike"], T=row["time_to_expiry"],
            r=risk_free_rate, sigma=row["smoothed_iv"],
            q=q, option_type=row["type"]
        ),
        axis=1
    )
    greeks_df = pd.DataFrame(greeks_list.tolist())
    df = pd.concat([df.reset_index(drop=True), greeks_df], axis=1)

    df["ticker"] = ticker
    df["spot_price"] = S

    logger.info(f"Priced {len(df)} contracts for {ticker}")
    return df
