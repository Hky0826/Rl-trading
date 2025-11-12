# File: data_handler.py
# Description: Optimized indicator calculations with Numba - FULL FEATURE SET
# =============================================================================
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import numba
import config
import traceback
import MetaTrader5 as mt5

# ===============================
# Numba-accelerated indicators
# ===============================

@numba.jit(nopython=True)
def _calculate_atr_numba(high, low, close, period):
    n = len(close)
    tr = np.zeros(n)
    atr = np.zeros(n)

    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    if period > 0 and n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


@numba.jit(nopython=True)
def _calculate_rsi_numba(close, period):
    n = len(close)
    if n < period + 1:
        return np.zeros(n)

    gain = np.zeros(n)
    loss = np.zeros(n)

    # manual diff (avoids np.diff)
    for i in range(1, n):
        delta = close[i] - close[i - 1]
        if delta > 0:
            gain[i] = delta
        else:
            loss[i] = -delta

    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)

    avg_gain[period] = np.mean(gain[1:period + 1])
    avg_loss[period] = np.mean(loss[1:period + 1])

    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period

    rsi = np.zeros(n)
    for i in range(period, n):
        rs = avg_gain[i] / (avg_loss[i] + 1e-9)
        rsi[i] = 100 - (100 / (1 + rs))

    return rsi


@numba.jit(nopython=True)
def _calculate_adx_numba(high, low, close, period):
    n = len(close)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down

    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    atr = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)

    if period > 0 and n >= period:
        atr[period - 1] = np.mean(tr[:period])
        plus_di[period - 1] = np.mean(plus_dm[:period])
        minus_di[period - 1] = np.mean(minus_dm[:period])

        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
            plus_di[i] = (plus_di[i - 1] * (period - 1) + plus_dm[i]) / period
            minus_di[i] = (minus_di[i - 1] * (period - 1) + minus_dm[i]) / period

        for i in range(period - 1, n):
            if atr[i] > 0:
                pdi = 100 * (plus_di[i] / atr[i])
                mdi = 100 * (minus_di[i] / atr[i])
                if (pdi + mdi) > 0:
                    dx[i] = 100 * abs(pdi - mdi) / (pdi + mdi)

        if n >= (2 * period - 1):
            adx[2 * period - 2] = np.mean(dx[period - 1: 2 * period - 1])
            for i in range(2 * period - 1, n):
                adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx

# ===============================
# Data fetchers
# ===============================

def fetch_data(ticker, start_date_str=config.START_DATE, end_date_str=config.END_DATE):
    """Fetch historical data from MT5"""
    import MetaTrader5 as mt5
    mt5.initialize()
    try:
        start_dt = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date_str, '%Y-%m-%d')

        rates_primary = mt5.copy_rates_range(ticker, config.PRIMARY_TIMEFRAME_MT5, start_dt, end_dt)
        rates_trend = mt5.copy_rates_range(ticker, config.TREND_TIMEFRAME_MT5, start_dt, end_dt)

        if rates_primary is None or rates_trend is None:
            return None, None

        df_primary = pd.DataFrame(rates_primary)
        df_primary['time'] = pd.to_datetime(df_primary['time'], unit='s')
        df_primary.set_index('time', inplace=True)

        df_trend = pd.DataFrame(rates_trend)
        df_trend['time'] = pd.to_datetime(df_trend['time'], unit='s')
        df_trend.set_index('time', inplace=True)

        for df in [df_primary, df_trend]:
            df.rename(
                columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'tick_volume': 'Volume',
                },
                inplace=True,
            )

        return df_primary.dropna(), df_trend.dropna()

    except Exception as e:
        logging.error(f"An error occurred in fetch_data: {e}")
        return None, None


def fetch_live_data(ticker, num_candles=300):
    """Fetch live data from MT5"""
    import MetaTrader5 as mt5
    mt5.initialize()
    try:
        rates_primary = mt5.copy_rates_from_pos(ticker, config.PRIMARY_TIMEFRAME_MT5, 0, num_candles)
        rates_trend = mt5.copy_rates_from_pos(ticker, config.TREND_TIMEFRAME_MT5, 0, num_candles)

        if (
            rates_primary is None or len(rates_primary) == 0
            or rates_trend is None or len(rates_trend) == 0
        ):
            logging.warning(f"MT5 returned no data for {ticker}. The market might be closed or symbol not available.")
            return None, None

        df_primary = pd.DataFrame(rates_primary)
        df_primary['time'] = pd.to_datetime(df_primary['time'], unit='s')
        df_primary.set_index('time', inplace=True)

        df_trend = pd.DataFrame(rates_trend)
        df_trend['time'] = pd.to_datetime(df_trend['time'], unit='s')
        df_trend.set_index('time', inplace=True)

        for df in [df_primary, df_trend]:
            df.rename(
                columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'tick_volume': 'Volume',
                },
                inplace=True,
            )

        return df_primary.dropna(), df_trend.dropna()

    except Exception as e:
        logging.error(f"An error occurred in fetch_live_data. Original error: {e}")
        logging.error(traceback.format_exc())
        return None, None

# ===============================
# Indicator preparation - FULL FEATURE SET
# ===============================

def prepare_indicators(data_m5, data_h1):
    """
    Prepare all indicators for the FULL FEATURE SET (35 features).
    
    Features generated:
    - OHLCV: Open, High, Low, Close, Volume
    - M5 Indicators: ATR, RSI, MACD_line, MACD_signal, MACD_hist, ADX, stoch_k, stoch_d
    - Candle patterns: upper_wick, lower_wick, body_size
    - H1 features: H1_Close, H1_EMA, H1_RSI, H1_ADX
    - Z-scored: RSI_Z, MACD_line_Z, MACD_hist_Z, ADX_Z, stoch_k_Z, stoch_d_Z,
                upper_wick_Z, lower_wick_Z, body_size_Z, H1_RSI_Z, H1_ADX_Z
    - Time features: hour_sin, hour_cos, day_sin, day_cos
    """
    df = data_m5.copy()
    h1_df = data_h1.copy()

    # ============================================
    # H1 Timeframe Indicators
    # ============================================
    h1_df['H1_RSI'] = _calculate_rsi_numba(h1_df['Close'].values, config.RSI_PERIOD)
    h1_df['H1_ADX'] = _calculate_adx_numba(h1_df['High'].values, h1_df['Low'].values, h1_df['Close'].values, config.ADX_PERIOD)
    h1_df['H1_EMA'] = h1_df['Close'].ewm(span=config.H1_EMA_PERIOD, adjust=False).mean()

    # ============================================
    # M5 Timeframe Indicators
    # ============================================
    
    # ATR
    df['ATR'] = _calculate_atr_numba(df['High'].values, df['Low'].values, df['Close'].values, config.M5_ATR_PERIOD)
    
    # RSI
    df['RSI'] = _calculate_rsi_numba(df['Close'].values, config.RSI_PERIOD)

    # MACD (Fast EMA - Slow EMA)
    ma_fast = df['Close'].ewm(span=config.MA_FAST_PERIOD, adjust=False).mean()
    ma_slow = df['Close'].ewm(span=config.MA_SLOW_PERIOD, adjust=False).mean()
    df['MACD_line'] = ma_fast - ma_slow
    
    # MACD Signal line (EMA of MACD line)
    df['MACD_signal'] = df['MACD_line'].ewm(span=config.MACD_SIGNAL_PERIOD, adjust=False).mean()
    
    # MACD Histogram (MACD line - Signal line)
    df['MACD_hist'] = df['MACD_line'] - df['MACD_signal']

    # ADX
    df['ADX'] = _calculate_adx_numba(df['High'].values, df['Low'].values, df['Close'].values, config.ADX_PERIOD)

    # Stochastic Oscillator
    low_k = df['Low'].rolling(window=config.STOCH_K_PERIOD).min()
    high_k = df['High'].rolling(window=config.STOCH_K_PERIOD).max()
    df['stoch_k'] = 100 * ((df['Close'] - low_k) / (high_k - low_k + 1e-9))
    df['stoch_d'] = df['stoch_k'].rolling(window=config.STOCH_D_PERIOD).mean()

    # ============================================
    # Candle Pattern Features
    # ============================================
    df['upper_wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['lower_wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['body_size'] = (df['Close'] - df['Open']).abs()

    # ============================================
    # Merge H1 features into M5 dataframe
    # ============================================
    df['H1_Close'] = h1_df['Close'].reindex(df.index, method='ffill')
    df['H1_EMA'] = h1_df['H1_EMA'].reindex(df.index, method='ffill')
    df['H1_RSI'] = h1_df['H1_RSI'].reindex(df.index, method='ffill')
    df['H1_ADX'] = h1_df['H1_ADX'].reindex(df.index, method='ffill')

    # ============================================
    # Z-Score Normalization (rolling window)
    # ============================================
    norm_window = config.NORMALIZATION_WINDOW
    
    features_to_normalize = [
        'RSI', 'MACD_line', 'MACD_hist', 'ADX',
        'stoch_k', 'stoch_d',
        'upper_wick', 'lower_wick', 'body_size',
        'H1_RSI', 'H1_ADX'
    ]
    
    for col in features_to_normalize:
        rolling_mean = df[col].rolling(window=norm_window, min_periods=1).mean()
        rolling_std = df[col].rolling(window=norm_window, min_periods=1).std()
        # Avoid division by zero
        rolling_std = rolling_std.replace(0, 1e-9)
        df[f'{col}_Z'] = (df[col] - rolling_mean) / rolling_std

    # ============================================
    # Time-based Features (cyclical encoding)
    # ============================================
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 23.0)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 23.0)
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 6.0)
    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 6.0)

    # ============================================
    # Cleanup
    # ============================================
    # Replace inf/-inf with NaN, then drop NaN rows
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # ============================================
    # Validation - Ensure all 35 features exist
    # ============================================
    required_features = [
        'Open', 'High', 'Low', 'Close', 'Volume', 
        'ATR', 'RSI', 'MACD_line', 'MACD_signal', 'MACD_hist',
        'ADX', 'stoch_k', 'stoch_d', 
        'upper_wick', 'lower_wick', 'body_size', 
        'H1_Close', 'H1_EMA', 'H1_RSI', 'H1_ADX', 
        'RSI_Z', 'MACD_line_Z', 'MACD_hist_Z', 'ADX_Z', 
        'stoch_k_Z', 'stoch_d_Z', 'upper_wick_Z', 'lower_wick_Z', 'body_size_Z', 
        'H1_RSI_Z', 'H1_ADX_Z', 
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
    ]
    
    missing_features = [feat for feat in required_features if feat not in df.columns]
    if missing_features:
        logging.error(f"Missing required features after preparation: {missing_features}")
        raise ValueError(f"Missing required features: {missing_features}")
    
    logging.info(f"Successfully prepared {len(required_features)} features for {len(df)} candles")

    return df


def validate_feature_set(df):
    """
    Validate that the dataframe contains all 35 required features.
    Returns True if valid, raises ValueError if not.
    """
    required_features = [
        'Open', 'High', 'Low', 'Close', 'Volume', 
        'ATR', 'RSI', 'MACD_line', 'MACD_signal', 'MACD_hist',
        'ADX', 'stoch_k', 'stoch_d', 
        'upper_wick', 'lower_wick', 'body_size', 
        'H1_Close', 'H1_EMA', 'H1_RSI', 'H1_ADX', 
        'RSI_Z', 'MACD_line_Z', 'MACD_hist_Z', 'ADX_Z', 
        'stoch_k_Z', 'stoch_d_Z', 'upper_wick_Z', 'lower_wick_Z', 'body_size_Z', 
        'H1_RSI_Z', 'H1_ADX_Z', 
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
    ]
    
    missing = [feat for feat in required_features if feat not in df.columns]
    
    if missing:
        raise ValueError(f"Missing {len(missing)} required features: {missing}")
    
    # Check for NaN values
    nan_counts = df[required_features].isnull().sum()
    if nan_counts.sum() > 0:
        logging.warning(f"NaN values found in features:\n{nan_counts[nan_counts > 0]}")
    
    # Check for inf values
    inf_counts = np.isinf(df[required_features]).sum()
    if inf_counts.sum() > 0:
        logging.warning(f"Inf values found in features:\n{inf_counts[inf_counts > 0]}")
    
    logging.info(f"✓ All {len(required_features)} features validated successfully")
    return True


def main():
    """Test the data handler"""
    ticker = "EURUSDc"
    
    print("Testing live data fetch and indicator preparation...")
    m5df, h1df = fetch_live_data(ticker)
    
    if m5df is not None and h1df is not None:
        print(f"\nFetched M5 data: {len(m5df)} candles")
        print(f"Fetched H1 data: {len(h1df)} candles")
        
        prepared_df = prepare_indicators(m5df, h1df)
        
        print(f"\nPrepared data: {len(prepared_df)} candles with {len(prepared_df.columns)} columns")
        print(f"\nFeatures: {list(prepared_df.columns)}")
        
        # Validate
        try:
            validate_feature_set(prepared_df)
            print("\n✓ Feature set validation PASSED")
        except ValueError as e:
            print(f"\n✗ Feature set validation FAILED: {e}")
        
        # Show sample
        print(f"\nSample data (last 5 rows):")
        print(prepared_df.tail())
    else:
        print("Failed to fetch data")

if __name__ == "__main__":
    main()