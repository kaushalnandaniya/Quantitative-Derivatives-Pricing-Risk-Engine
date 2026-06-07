"""
Historical Data Service
=========================
Fetches OHLCV data from Kite Connect (if authenticated) or yfinance (fallback).
"""

import os
import logging
import pandas as pd
from typing import Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def get_historical_data(
    symbol: str,
    period_days: int = 365,
    interval: str = "day",
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data.
    Tries Kite API first (if credentials present), falls back to yfinance.

    Args:
        symbol: Stock ticker symbol (e.g. "RELIANCE", "NIFTY", "SBIN")
        period_days: Number of days of historical data
        interval: "day", "60minute", "15minute" etc.

    Returns:
        DataFrame with columns: open, high, low, close, volume (lowercase)
    """
    # Try Kite first
    kite_key = os.getenv("KITE_API_KEY")
    kite_token = os.getenv("KITE_ACCESS_TOKEN")

    if kite_key and kite_token:
        try:
            df = _fetch_from_kite(symbol, period_days, interval, kite_key, kite_token)
            if df is not None and not df.empty:
                logger.info(f"Historical data for {symbol} fetched from Kite ({len(df)} bars)")
                return df
        except Exception as e:
            logger.warning(f"Kite data fetch failed for {symbol}: {e}")

    # Fallback to yfinance
    try:
        df = _fetch_from_yfinance(symbol, period_days, interval)
        if df is not None and not df.empty:
            logger.info(f"Historical data for {symbol} fetched from yfinance ({len(df)} bars)")
            return df
    except Exception as e:
        logger.warning(f"yfinance data fetch failed for {symbol}: {e}")

    # Generate mock data as last resort
    logger.warning(f"Using mock data for {symbol}")
    return _generate_mock_data(symbol, period_days)


def _fetch_from_kite(symbol: str, period_days: int, interval: str,
                     api_key: str, access_token: str) -> Optional[pd.DataFrame]:
    """Fetch from Kite Connect API."""
    from kiteconnect import KiteConnect

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)

    # Map common names to Kite instrument tokens
    # For a real implementation, you'd look up the instrument token from the instruments list
    instrument_map = {
        "NIFTY": "NSE:NIFTY 50",
        "BANKNIFTY": "NSE:NIFTY BANK",
    }

    trading_symbol = instrument_map.get(symbol.upper(), f"NSE:{symbol.upper()}")

    to_date = datetime.now()
    from_date = to_date - timedelta(days=period_days)

    kite_interval = {
        "day": "day", "60minute": "60minute",
        "15minute": "15minute", "5minute": "5minute",
    }.get(interval, "day")

    data = kite.historical_data(
        instrument_token=trading_symbol,
        from_date=from_date.strftime("%Y-%m-%d"),
        to_date=to_date.strftime("%Y-%m-%d"),
        interval=kite_interval,
    )

    if not data:
        return None

    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    df.columns = [c.lower() for c in df.columns]
    return df[['open', 'high', 'low', 'close', 'volume']]


def _fetch_from_yfinance(symbol: str, period_days: int, interval: str) -> Optional[pd.DataFrame]:
    """Fetch from Yahoo Finance."""
    import yfinance as yf

    # Map common Indian symbols
    yf_map = {
        "NIFTY": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "SENSEX": "^BSESN",
    }

    yf_symbol = yf_map.get(symbol.upper(), f"{symbol.upper()}.NS")

    yf_interval = {"day": "1d", "60minute": "1h", "15minute": "15m"}.get(interval, "1d")

    period_str = f"{period_days}d"
    if period_days > 365:
        period_str = f"{period_days // 365}y"

    ticker = yf.Ticker(yf_symbol)
    df = ticker.history(period=period_str, interval=yf_interval)

    if df.empty:
        return None

    df.columns = [c.lower() for c in df.columns]

    # Ensure required columns
    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            return None

    return df[required]


def _generate_mock_data(symbol: str, period_days: int) -> pd.DataFrame:
    """Generate realistic mock OHLCV data."""
    import numpy as np

    base_prices = {
        "NIFTY": 24000, "BANKNIFTY": 51000, "RELIANCE": 2800,
        "TCS": 3800, "INFY": 1600, "HDFCBANK": 1700,
        "SBIN": 820, "ITC": 450,
    }
    base = base_prices.get(symbol.upper(), 1000)

    rng = np.random.default_rng(42)
    n = period_days

    dates = pd.date_range(end=datetime.now(), periods=n, freq='B')
    returns = rng.normal(0.0005, 0.015, n)

    close = [base]
    for r in returns[1:]:
        close.append(close[-1] * (1 + r))
    close = np.array(close[:n])

    high = close * (1 + rng.uniform(0.002, 0.02, n))
    low = close * (1 - rng.uniform(0.002, 0.02, n))
    open_ = low + (high - low) * rng.uniform(0.3, 0.7, n)
    volume = rng.integers(100000, 5000000, n).astype(float)

    df = pd.DataFrame({
        'open': open_, 'high': high, 'low': low,
        'close': close, 'volume': volume
    }, index=dates[:n])

    return df


def search_symbols(query: str) -> list:
    """Search for stock symbols matching a query."""
    # Common NSE symbols
    all_symbols = [
        {"symbol": "NIFTY", "name": "NIFTY 50 Index", "exchange": "NSE"},
        {"symbol": "BANKNIFTY", "name": "NIFTY Bank Index", "exchange": "NSE"},
        {"symbol": "RELIANCE", "name": "Reliance Industries", "exchange": "NSE"},
        {"symbol": "TCS", "name": "Tata Consultancy Services", "exchange": "NSE"},
        {"symbol": "INFY", "name": "Infosys Ltd", "exchange": "NSE"},
        {"symbol": "HDFCBANK", "name": "HDFC Bank Ltd", "exchange": "NSE"},
        {"symbol": "ICICIBANK", "name": "ICICI Bank Ltd", "exchange": "NSE"},
        {"symbol": "SBIN", "name": "State Bank of India", "exchange": "NSE"},
        {"symbol": "ITC", "name": "ITC Ltd", "exchange": "NSE"},
        {"symbol": "WIPRO", "name": "Wipro Ltd", "exchange": "NSE"},
        {"symbol": "TATAMOTORS", "name": "Tata Motors Ltd", "exchange": "NSE"},
        {"symbol": "ADANIENT", "name": "Adani Enterprises", "exchange": "NSE"},
        {"symbol": "BHARTIARTL", "name": "Bharti Airtel", "exchange": "NSE"},
        {"symbol": "BAJFINANCE", "name": "Bajaj Finance", "exchange": "NSE"},
        {"symbol": "MARUTI", "name": "Maruti Suzuki India", "exchange": "NSE"},
        {"symbol": "SUNPHARMA", "name": "Sun Pharma", "exchange": "NSE"},
        {"symbol": "TITAN", "name": "Titan Company", "exchange": "NSE"},
        {"symbol": "ASIANPAINT", "name": "Asian Paints", "exchange": "NSE"},
        {"symbol": "LT", "name": "Larsen & Toubro", "exchange": "NSE"},
        {"symbol": "AXISBANK", "name": "Axis Bank Ltd", "exchange": "NSE"},
        {"symbol": "KOTAKBANK", "name": "Kotak Mahindra Bank", "exchange": "NSE"},
        {"symbol": "HINDUNILVR", "name": "Hindustan Unilever", "exchange": "NSE"},
        {"symbol": "ONGC", "name": "Oil & Natural Gas Corp", "exchange": "NSE"},
        {"symbol": "NTPC", "name": "NTPC Ltd", "exchange": "NSE"},
        {"symbol": "POWERGRID", "name": "Power Grid Corp", "exchange": "NSE"},
        {"symbol": "TATASTEEL", "name": "Tata Steel Ltd", "exchange": "NSE"},
        {"symbol": "JSWSTEEL", "name": "JSW Steel Ltd", "exchange": "NSE"},
        {"symbol": "COALINDIA", "name": "Coal India Ltd", "exchange": "NSE"},
        {"symbol": "HCLTECH", "name": "HCL Technologies", "exchange": "NSE"},
        {"symbol": "TECHM", "name": "Tech Mahindra", "exchange": "NSE"},
    ]

    q = query.upper().strip()
    if not q:
        return all_symbols[:10]

    results = [s for s in all_symbols if q in s["symbol"] or q in s["name"].upper()]
    return results[:10]
