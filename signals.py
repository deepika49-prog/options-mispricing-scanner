# signals.py


import pandas as pd
import numpy as np
from loguru import logger


# Filters 


MIN_OPEN_INTEREST = 50      # Minimum open interest : avoids ghost contracts
MIN_VOLUME = 5              # Minimum daily volume
MAX_SPREAD_PCT = 0.20       # Max bid-ask spread as % of mid : filters wide spreads
MIN_TIME_TO_EXPIRY = 7/365  # At least 1 week out : avoids expiry-day weirdness
MAX_TIME_TO_EXPIRY = 1.5    # At most 18 months : very long-dated options are illiquid
MIN_DELTA_ABS = 0.05        # Avoid deep OTM options (delta < 0.05) : BS breaks down
MAX_DELTA_ABS = 0.95        # Avoid deep ITM options : illiquid, wide spreads


def compute_edge(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds edge columns to the DataFrame:
      - edge: market mid price minus model price (in dollars)
      - edge_pct: edge as a percentage of model price
      - direction: 'overpriced' or 'underpriced'

    A large absolute edge_pct is the primary signal we're looking for.
    """
    df = df.copy()

    # Edge in dollars: positive = market price > model (overpriced by market)
    df["edge"] = df["mid"] - df["model_price"]

    # Edge as a percentage of model price — normalizes across different price levels
    df["edge_pct"] = df["edge"] / df["model_price"].replace(0, np.nan) * 100

    # Human-readable direction label
    df["direction"] = df["edge"].apply(lambda x: "overpriced" if x > 0 else "underpriced")

    return df


def apply_liquidity_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes contracts that are too illiquid to trade meaningfully.

    Even if our model says a contract is mispriced, if the bid-ask spread
    is 30% of the price, you can't actually capture that edge in practice.
    """
    original_count = len(df)

    # Remove rows with missing critical data
    df = df.dropna(subset=["model_price", "edge_pct", "delta", "mid"])

    # Remove contracts with zero or near-zero model price (prevents division issues)
    df = df[df["model_price"] > 0.01]

    # Open interest filter
    if "open_interest" in df.columns:
        df = df[df["open_interest"] >= MIN_OPEN_INTEREST]

    # Volume filter
    if "volume" in df.columns:
        df = df[df["volume"].fillna(0) >= MIN_VOLUME]

    # Time to expiry filter
    df = df[
        (df["time_to_expiry"] >= MIN_TIME_TO_EXPIRY) &
        (df["time_to_expiry"] <= MAX_TIME_TO_EXPIRY)
    ]

    # Bid-ask spread filter
    # If bid or ask is 0, the contract is essentially untradeable
    df = df[(df["bid"] > 0) & (df["ask"] > 0)]
    df["spread_pct"] = (df["ask"] - df["bid"]) / df["mid"]
    df = df[df["spread_pct"] <= MAX_SPREAD_PCT]

    # Delta filter — avoid deep ITM/OTM where Black-Scholes is unreliable
    df = df[
        (df["delta"].abs() >= MIN_DELTA_ABS) &
        (df["delta"].abs() <= MAX_DELTA_ABS)
    ]

    # Remove contracts where edge is smaller than half the spread
    # (the spread would eat the entire theoretical profit)
    df = df[df["edge"].abs() > (df["ask"] - df["bid"]) / 2]

    filtered_count = len(df)
    logger.info(f"Liquidity filter: {original_count} → {filtered_count} contracts")
    return df.reset_index(drop=True)


def rank_signals(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Ranks contracts by absolute edge percentage and returns the top N.

    Also adds a 'confidence' score (0–100) based on:
      - Open interest (more = higher confidence)
      - Volume (more = higher confidence)
      - Spread tightness (tighter = higher confidence)
    """
    df = df.copy()

    # Normalize each component to 0-1 scale, then combine
    oi_score = df["open_interest"].clip(upper=5000) / 5000
    vol_score = df["volume"].fillna(0).clip(upper=1000) / 1000
    spread_score = 1 - df["spread_pct"].clip(upper=MAX_SPREAD_PCT) / MAX_SPREAD_PCT

    df["confidence"] = ((oi_score + vol_score + spread_score) / 3 * 100).round(1)

    # Sort by absolute edge percentage (largest mispricing first)
    df = df.sort_values("edge_pct", key=abs, ascending=False)

    # Take top N
    df = df.head(top_n).reset_index(drop=True)
    df.index += 1  # rank starts at 1

    return df


def generate_signals(priced_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Master function — runs the full signal pipeline:
      1. Compute edge
      2. Apply liquidity filters
      3. Rank and return top signals

    Returns a clean DataFrame ready for the dashboard.
    """
    df = compute_edge(priced_df)
    df = apply_liquidity_filters(df)

    if df.empty:
        logger.warning("No signals passed the liquidity filters.")
        return pd.DataFrame()

    signals = rank_signals(df, top_n=top_n)

    # Round display columns
    signals["edge"] = signals["edge"].round(3)
    signals["edge_pct"] = signals["edge_pct"].round(2)
    signals["model_price"] = signals["model_price"].round(3)
    signals["mid"] = signals["mid"].round(3)

    logger.info(f"Generated {len(signals)} signals")
    return signals


def get_vol_surface_data(priced_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts the data needed to plot the 3D volatility surface.
    Returns a DataFrame with strike, time_to_expiry, and smoothed_iv
    for contracts suitable for surface visualization.
    """
    df = priced_df[
        (priced_df["smoothed_iv"] > 0.01) &
        (priced_df["smoothed_iv"] < 3.0) &
        (priced_df["time_to_expiry"] > 0) &
        (priced_df["open_interest"] > 5)
    ].copy()

    # Use calls only for the surface (cleaner, avoids put-call parity noise)
    df = df[df["type"] == "call"]

    return df[["strike", "time_to_expiry", "smoothed_iv", "market_iv", "moneyness"]].dropna()
