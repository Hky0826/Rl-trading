# File: data_handler_optimized.py
# Description: Ultra-fast data preprocessing with heavy caching and vectorization
# =============================================================================
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import numba
import config
import traceback
import joblib
import os
from pathlib import Path
import hashlib

# Global cache for processed data
_INDICATOR_CACHE = {}
_DATA_HASH_CACHE = {}

def get_data_hash(df):
    """Generate hash of dataframe for caching purposes"""
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

@numba.jit(nopython=True, cache=True, fastmath=True)
def _calculate_atr_vectorized(high, low, close, period):
    """Vectorized ATR calculation with Numba optimization"""
    n = len(close)
    tr = np.zeros(n, dtype=np.float32)
    atr = np.zeros(n, dtype=np.float32)
    
    # Vectorized true range calculation
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    # Exponential moving average for ATR (faster than SMA)
    if period > 0 and n >= period:
        alpha = 2.0 / (period + 1)
        atr[period-1] = np.mean(tr[1:period+1])
        
        for i in range(period, n):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
    
    return atr

@numba.jit(nopython=True, cache=True, fastmath=True)
def _calculate_rsi_vectorized(close, period):
    """Ultra-fast RSI calculation"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n, dtype=np.float32)
    
    gains = np.zeros(n, dtype=np.float32)
    losses = np.zeros(n, dtype=np.float32)
    
    # Calculate gains and losses
    for i in range(1, n):
        delta = close[i] - close[i-1]
        if delta > 0:
            gains[i] = delta
        else:
            losses[i] = -delta
    
    # Use exponential moving average instead of Wilder's smoothing (faster)
    alpha = 1.0 / period
    avg_gain = np.zeros(n, dtype=np.float32)
    avg_loss = np.zeros(n, dtype=np.float32)
    
    # Initial values
    avg_gain[period] = np.mean(gains[1:period+1])
    avg_loss[period] = np.mean(losses[1:period+1])
    
    # Exponential smoothing
    for i in range(period + 1, n):
        avg_gain[i] = alpha * gains[i] + (1 - alpha) * avg_gain[i-1]
        avg_loss[i] = alpha * losses[i] + (1 - alpha) * avg_loss[i-1]
    
    # Calculate RSI
    rsi = np.zeros(n, dtype=np.float32)
    for i in range(period, n):
        if avg_loss[i] > 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

@numba.jit(nopython=True, cache=True, fastmath=True) 
def _calculate_stochastic_vectorized(high, low, close, k_period, d_period):
    """Vectorized stochastic calculation"""
    n = len(close)
    stoch_k = np.zeros(n, dtype=np.float32)
    stoch_d = np.zeros(n, dtype=np.float32)
    
    for i in range(k_period-1, n):
        start_idx = i - k_period + 1
        lowest_low = np.min(low[start_idx:i+1])
        highest_high = np.max(high[start_idx:i+1])
        
        if highest_high > lowest_low:
            stoch_k[i] = 100.0 * (close[i] - lowest_low) / (highest_high - lowest_low)
        else:
            stoch_k[i] = 50.0
    
    # Calculate %D (moving average of %K)
    for i in range(k_period + d_period - 2, n):
        start_idx = i - d_period + 1
        stoch_d[i] = np.mean(stoch_k[start_idx:i+1])
    
    return stoch_k, stoch_d

def cache_indicators(df, cache_key):
    """Cache computed indicators to disk"""
    cache_dir = Path("indicator_cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{cache_key}.pkl"
    
    try:
        joblib.dump(df, cache_file, compress=3)
        logging.debug(f"Cached indicators to {cache_file}")
    except Exception as e:
        logging.warning(f"Failed to cache indicators: {e}")

def load_cached_indicators(cache_key):
    """Load cached indicators from disk"""
    cache_dir = Path("indicator_cache")
    cache_file = cache_dir / f"{cache_key}.pkl"
    
    if cache_file.exists():
        try:
            df = joblib.load(cache_file)
            logging.info(f"Loaded cached indicators from {cache_file}")
            return df
        except Exception as e:
            logging.warning(f"Failed to load cached indicators: {e}")
            # Remove corrupted cache file
            try:
                cache_file.unlink()
            except:
                pass
    
    return None

def prepare_indicators_optimized(data_m5, data_h1, force_recalculate=False):
    """Ultra-fast indicator preparation with aggressive caching"""
    
    # Generate cache key based on data content
    cache_key = f"indicators_{get_data_hash(data_m5)}_{get_data_hash(data_h1)}"
    
    # Try to load from cache first
    if not force_recalculate:
        cached_df = load_cached_indicators(cache_key)
        if cached_df is not None:
            return cached_df
    
    logging.info("Computing indicators (not cached)...")
    start_time = datetime.now()
    
    # Work on copies to avoid modifying originals
    df = data_m5.copy()
    h1_df = data_h1.copy()
    
    # Convert to float32 early for memory efficiency
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)
        if col in h1_df.columns:
            h1_df[col] = h1_df[col].astype(np.float32)
    
    # Pre-extract numpy arrays for vectorized operations
    h1_close = h1_df['Close'].values
    h1_high = h1_df['High'].values
    h1_low = h1_df['Low'].values
    
    m5_close = df['Close'].values
    m5_high = df['High'].values
    m5_low = df['Low'].values
    m5_open = df['Open'].values
    
    # H1 indicators (vectorized)
    h1_df['H1_RSI'] = _calculate_rsi_vectorized(h1_close, config.RSI_PERIOD)
    h1_df['H1_EMA'] = h1_df['Close'].ewm(span=config.H1_EMA_PERIOD, adjust=False).mean().astype(np.float32)
    
    # M5 indicators (vectorized)
    df['ATR'] = _calculate_atr_vectorized(m5_high, m5_low, m5_close, config.M5_ATR_PERIOD)
    df['RSI'] = _calculate_rsi_vectorized(m5_close, config.RSI_PERIOD)
    
    # MACD (vectorized)
    ema_fast = df['Close'].ewm(span=config.MA_FAST_PERIOD, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=config.MA_SLOW_PERIOD, adjust=False).mean()
    df['MACD_line'] = (ema_fast - ema_slow).astype(np.float32)
    df['MACD_signal'] = df['MACD_line'].ewm(span=config.MACD_SIGNAL_PERIOD, adjust=False).mean().astype(np.float32)
    df['MACD_hist'] = (df['MACD_line'] - df['MACD_signal']).astype(np.float32)
    
    # Stochastic (vectorized)
    stoch_k, stoch_d = _calculate_stochastic_vectorized(
        m5_high, m5_low, m5_close, config.STOCH_K_PERIOD, config.STOCH_D_PERIOD
    )
    df['stoch_k'] = stoch_k
    df['stoch_d'] = stoch_d
    
    # Candlestick patterns (vectorized)
    high_val = df['High'].values
    low_val = df['Low'].values
    open_val = df['Open'].values
    close_val = df['Close'].values
    
    df['upper_wick'] = high_val - np.maximum(open_val, close_val)
    df['lower_wick'] = np.minimum(open_val, close_val) - low_val
    df['body_size'] = np.abs(close_val - open_val)
    
    # Merge H1 data (forward fill)
    df['H1_Close'] = h1_df['Close'].reindex(df.index, method='ffill').astype(np.float32)
    df['H1_EMA'] = h1_df['H1_EMA'].reindex(df.index, method='ffill').astype(np.float32)
    df['H1_RSI'] = h1_df['H1_RSI'].reindex(df.index, method='ffill').astype(np.float32)
    
    # Simplified ADX calculation (computationally expensive, so we'll simplify or skip)
    # For speed, we'll use a simple trend strength indicator instead
    price_change = np.diff(m5_close, prepend=m5_close[0])
    df['ADX'] = pd.Series(price_change).rolling(config.ADX_PERIOD).std().fillna(0).astype(np.float32)
    df['H1_ADX'] = df['ADX']  # Simplified
    
    # Fast normalization (reduced window for speed)
    norm_window = config.NORMALIZATION_WINDOW
    features_to_normalize = [
        'RSI', 'MACD_line', 'MACD_hist', 'ADX', 'stoch_k', 'stoch_d',
        'upper_wick', 'lower_wick', 'body_size', 'H1_RSI', 'H1_ADX'
    ]
    
    for col in features_to_normalize:
        if col in df.columns:
            rolling_mean = df[col].rolling(window=norm_window, min_periods=1).mean()
            rolling_std = df[col].rolling(window=norm_window, min_periods=1).std().fillna(1.0)
            df[f'{col}_Z'] = ((df[col] - rolling_mean) / rolling_std).astype(np.float32)
    
    # Time features (vectorized)
    hour_norm = df.index.hour / 23.0
    day_norm = df.index.dayofweek / 6.0
    
    df['hour_sin'] = np.sin(2 * np.pi * hour_norm).astype(np.float32)
    df['hour_cos'] = np.cos(2 * np.pi * hour_norm).astype(np.float32)
    df['day_sin'] = np.sin(2 * np.pi * day_norm).astype(np.float32)
    df['day_cos'] = np.cos(2 * np.pi * day_norm).astype(np.float32)
    
    # Clean up data
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Cache the results
    cache_indicators(df, cache_key)
    
    computation_time = (datetime.now() - start_time).total_seconds()
    logging.info(f"Indicators computed in {computation_time:.2f}s")
    
    return df

def fetch_data_cached(ticker, start_date_str=config.START_DATE, end_date_str=config.END_DATE):
    """Fetch data with caching support"""
    cache_key = f"raw_data_{ticker}_{start_date_str}_{end_date_str}"
    cache_dir = Path("data_cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{cache_key}.parquet"
    
    # Try to load from cache
    if cache_file.exists():
        try:
            cached_data = joblib.load(cache_file)
            logging.info(f"Loaded cached data from {cache_file}")
            return cached_data['m5'], cached_data['h1']
        except Exception as e:
            logging.warning(f"Failed to load cached data: {e}")
            try:
                cache_file.unlink()
            except:
                pass
    
    # Fetch fresh data
    logging.info("Fetching fresh data from MT5...")
    df_m5, df_h1 = fetch_data(ticker, start_date_str, end_date_str)
    
    if df_m5 is not None and df_h1 is not None:
        # Cache the data
        try:
            cache_data = {'m5': df_m5, 'h1': df_h1}
            joblib.dump(cache_data, cache_file, compress=3)
            logging.info(f"Cached raw data to {cache_file}")
        except Exception as e:
            logging.warning(f"Failed to cache data: {e}")
    
    return df_m5, df_h1

# Import the original functions for compatibility
from data_handler import fetch_data, fetch_live_data

# Replace the original function with optimized version
prepare_indicators = prepare_indicators_optimized