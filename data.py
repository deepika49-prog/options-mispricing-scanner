# data.py


import os
import yfinance as yf
import pandas as pd
from fredapi import Fred
from loguru import logger
from dotenv import load_dotenv

load_dotenv()  # reads .env file 


def get_risk_free_rate() -> float:
    """
    Fetches the current 3-month US Treasury Bill rate from FRED.
    This is the standard 'risk-free rate' used in options pricing.
    Returns a decimal, e.g. 0.053 means 5.3%.
    """
    try:
        fred = Fred(api_key=os.getenv("FRED_API_KEY"))
        # DTB3 = 3-Month Treasury Bill: Secondary Market Rate
        series = fred.get_series("DTB3")
        rate = series.dropna().iloc[-1]  # most recent value
        logger.info(f"Risk-free rate: {rate:.3f}%")
        return float(rate) / 100  # convert from percent to decimal
    except Exception as e:
        logger.warning(f"Could not fetch risk-free rate from FRED: {e}. Using 5% default.")
        return 0.05  # safe fallback


def get_options_data(ticker: str) -> dict:
    """
    Fetches everything we need for a single ticker:
      - spot_price: current stock price
      - dividend_yield: annual dividend as a decimal
      - options: a DataFrame of all options contracts (calls + puts)

    The options DataFrame has these columns:
      strike, expiry, type (call/put), bid, ask, mid, impliedVolatility,
      openInterest, volume, lastPrice, inTheMoney
    """
    logger.info(f"Fetching data for {ticker}...")
    stock = yf.Ticker(ticker)

    # Spot price
    info = stock.info
    spot_price = info.get("currentPrice") or info.get("regularMarketPrice")
    if not spot_price:
        # fallback: use last closing price from recent history
        hist = stock.history(period="1d")
        spot_price = float(hist["Close"].iloc[-1])
    logger.info(f"{ticker} spot price: ${spot_price:.2f}")

    # Dividend yield
    dividend_yield = info.get("dividendYield") or 0.0

    # Options chain
    # yfinance gives us a list of expiry dates available for this ticker
    expiry_dates = stock.options
    if not expiry_dates:
        raise ValueError(f"No options data available for {ticker}")

    all_contracts = []

    for expiry in expiry_dates:
        try:
            chain = stock.option_chain(expiry)

            # Process calls
            calls = chain.calls.copy()
            calls["type"] = "call"
            calls["expiry"] = expiry

            # Process puts
            puts = chain.puts.copy()
            puts["type"] = "put"
            puts["expiry"] = expiry

            all_contracts.append(calls)
            all_contracts.append(puts)

        except Exception as e:
            logger.warning(f"Could not fetch chain for {ticker} expiry {expiry}: {e}")
            continue

    if not all_contracts:
        raise ValueError(f"Could not retrieve any options contracts for {ticker}")

    df = pd.concat(all_contracts, ignore_index=True)

    # Rename columns to be more readable
    df = df.rename(columns={
        "impliedVolatility": "market_iv",
        "openInterest": "open_interest",
        "lastPrice": "last_price",
        "inTheMoney": "in_the_money",
    })

    # Calculate the mid-price (average of bid and ask)
    # Mid is a better estimate of fair market price than last trade
    df["mid"] = (df["bid"] + df["ask"]) / 2

    # Keep only the columns we actually use
    keep_cols = [
        "expiry", "type", "strike", "bid", "ask", "mid",
        "market_iv", "open_interest", "volume", "last_price", "in_the_money"
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

    logger.info(f"Fetched {len(df)} contracts across {len(expiry_dates)} expiries for {ticker}")

    return {
        "ticker": ticker,
        "spot_price": spot_price,
        "dividend_yield": dividend_yield,
        "options": df,
    }
