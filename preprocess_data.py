# File: preprocess_data.py
# Description: (OPTIMIZED) High-speed realistic data preprocessing with minimal RAM usage
# =============================================================================
import logging
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import MetaTrader5 as mt5
from collections import deque
import numba

import config
import data_handler

@numba.jit(nopython=True)
def rolling_mean_std_numba(values, window_size):
    """Fast rolling mean and std calculation using numba"""
    n = len(values)
    means = np.empty(n)
    stds = np.empty(n)
    
    for i in range(n):
        start_idx = max(0, i - window_size + 1)
        window_data = values[start_idx:i+1]
        means[i] = np.mean(window_data)
        stds[i] = np.std(window_data)
    
    return means, stds

@numba.jit(nopython=True) 
def calculate_ema_numba(values, span):
    """Fast EMA calculation"""
    alpha = 2.0 / (span + 1.0)
    result = np.empty_like(values)
    result[0] = values[0]
    
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
    
    return result

@numba.jit(nopython=True)
def calculate_macd_numba(close_prices, fast_period, slow_period, signal_period):
    """Fast MACD calculation"""
    ema_fast = calculate_ema_numba(close_prices, fast_period)
    ema_slow = calculate_ema_numba(close_prices, slow_period)
    macd_line = ema_fast - ema_slow
    macd_signal = calculate_ema_numba(macd_line, signal_period)
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist

@numba.jit(nopython=True)
def calculate_stochastic_numba(high, low, close, k_period, d_period):
    """Fast Stochastic oscillator calculation"""
    n = len(close)
    stoch_k = np.empty(n)
    
    for i in range(n):
        start_idx = max(0, i - k_period + 1)
        period_high = np.max(high[start_idx:i+1])
        period_low = np.min(low[start_idx:i+1])
        
        if period_high == period_low:
            stoch_k[i] = 50.0
        else:
            stoch_k[i] = 100.0 * (close[i] - period_low) / (period_high - period_low)
    
    # Calculate %D as moving average of %K
    stoch_d = np.empty(n)
    for i in range(n):
        start_idx = max(0, i - d_period + 1)
        stoch_d[i] = np.mean(stoch_k[start_idx:i+1])
    
    return stoch_k, stoch_d

class StreamingIndicators:
    """Memory-efficient streaming indicator calculator"""
    
    def __init__(self, lookback_window=300):
        self.lookback_window = lookback_window
        
        # Rolling buffers for M5 data
        self.m5_ohlcv = deque(maxlen=lookback_window)
        self.m5_atr = deque(maxlen=lookback_window)
        self.m5_rsi = deque(maxlen=lookback_window)
        self.m5_adx = deque(maxlen=lookback_window)
        self.m5_macd_line = deque(maxlen=lookback_window)
        self.m5_macd_signal = deque(maxlen=lookback_window)
        self.m5_macd_hist = deque(maxlen=lookback_window)
        self.m5_stoch_k = deque(maxlen=lookback_window)
        self.m5_stoch_d = deque(maxlen=lookback_window)
        
        # Rolling buffers for H1 data  
        self.h1_close = deque(maxlen=lookback_window//6)  # H1 has 6x fewer candles than M5
        self.h1_ema = deque(maxlen=lookback_window//6)
        self.h1_rsi = deque(maxlen=lookback_window//6)
        self.h1_adx = deque(maxlen=lookback_window//6)
        
        # Normalization buffers
        self.norm_window = config.NORMALIZATION_WINDOW
        self.norm_buffers = {
            'RSI': deque(maxlen=self.norm_window),
            'MACD_line': deque(maxlen=self.norm_window),
            'MACD_hist': deque(maxlen=self.norm_window),
            'ADX': deque(maxlen=self.norm_window),
            'stoch_k': deque(maxlen=self.norm_window),
            'stoch_d': deque(maxlen=self.norm_window),
            'upper_wick': deque(maxlen=self.norm_window),
            'lower_wick': deque(maxlen=self.norm_window),
            'body_size': deque(maxlen=self.norm_window),
            'H1_RSI': deque(maxlen=self.norm_window),
            'H1_ADX': deque(maxlen=self.norm_window)
        }
        
        # Last known H1 values for forward filling
        self.last_h1_values = {'close': 0, 'ema': 0, 'rsi': 0, 'adx': 0}
        
    def update_m5(self, candle_row):
        """Update with new M5 candle"""
        o, h, l, c, v = candle_row[['Open', 'High', 'Low', 'Close', 'Volume']].values
        self.m5_ohlcv.append((o, h, l, c, v))
        
        if len(self.m5_ohlcv) < 50:  # Need minimum data for indicators
            return None
            
        # Extract arrays for calculations
        ohlcv_array = np.array(self.m5_ohlcv)
        high_arr = ohlcv_array[:, 1]
        low_arr = ohlcv_array[:, 2] 
        close_arr = ohlcv_array[:, 3]
        
        # Calculate indicators
        atr = data_handler._calculate_atr_numba(high_arr, low_arr, close_arr, config.M5_ATR_PERIOD)[-1]
        rsi = data_handler._calculate_rsi_numba(close_arr, config.RSI_PERIOD)[-1]
        adx = data_handler._calculate_adx_numba(high_arr, low_arr, close_arr, config.ADX_PERIOD)[-1]
        
        macd_line, macd_signal, macd_hist = calculate_macd_numba(
            close_arr, config.MA_FAST_PERIOD, config.MA_SLOW_PERIOD, config.MACD_SIGNAL_PERIOD
        )
        
        stoch_k, stoch_d = calculate_stochastic_numba(
            high_arr, low_arr, close_arr, config.STOCH_K_PERIOD, config.STOCH_D_PERIOD
        )
        
        # Store latest values
        self.m5_atr.append(atr)
        self.m5_rsi.append(rsi)
        self.m5_adx.append(adx)
        self.m5_macd_line.append(macd_line[-1])
        self.m5_macd_signal.append(macd_signal[-1])
        self.m5_macd_hist.append(macd_hist[-1])
        self.m5_stoch_k.append(stoch_k[-1])
        self.m5_stoch_d.append(stoch_d[-1])
        
        return {
            'atr': atr, 'rsi': rsi, 'adx': adx,
            'macd_line': macd_line[-1], 'macd_signal': macd_signal[-1], 'macd_hist': macd_hist[-1],
            'stoch_k': stoch_k[-1], 'stoch_d': stoch_d[-1]
        }
    
    def update_h1(self, h1_candle_row):
        """Update with new H1 candle when available"""
        if h1_candle_row is not None:
            close = h1_candle_row['Close']
            
            self.h1_close.append(close)
            
            if len(self.h1_close) >= config.H1_EMA_PERIOD:
                # Calculate H1 indicators
                close_arr = np.array(self.h1_close)
                
                ema = calculate_ema_numba(close_arr, config.H1_EMA_PERIOD)[-1]
                rsi = data_handler._calculate_rsi_numba(close_arr, config.RSI_PERIOD)[-1]
                
                # For ADX we need OHLC data - simplified approximation
                high_arr = close_arr * 1.001  # Approximate high
                low_arr = close_arr * 0.999   # Approximate low
                adx = data_handler._calculate_adx_numba(high_arr, low_arr, close_arr, config.ADX_PERIOD)[-1]
                
                self.h1_ema.append(ema)
                self.h1_rsi.append(rsi) 
                self.h1_adx.append(adx)
                
                # Update last known values
                self.last_h1_values = {'close': close, 'ema': ema, 'rsi': rsi, 'adx': adx}
    
    def get_current_features(self, candle_row, timestamp):
        """Get complete feature vector for current candle"""
        if len(self.m5_ohlcv) < self.norm_window:
            return None
            
        # Basic candle features
        o, h, l, c = candle_row[['Open', 'High', 'Low', 'Close']].values
        
        # Candlestick patterns
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        body_size = abs(c - o)
        
        # Update normalization buffers
        current_indicators = {
            'RSI': self.m5_rsi[-1] if self.m5_rsi else 50,
            'MACD_line': self.m5_macd_line[-1] if self.m5_macd_line else 0,
            'MACD_hist': self.m5_macd_hist[-1] if self.m5_macd_hist else 0,
            'ADX': self.m5_adx[-1] if self.m5_adx else 0,
            'stoch_k': self.m5_stoch_k[-1] if self.m5_stoch_k else 50,
            'stoch_d': self.m5_stoch_d[-1] if self.m5_stoch_d else 50,
            'upper_wick': upper_wick,
            'lower_wick': lower_wick,
            'body_size': body_size,
            'H1_RSI': self.last_h1_values['rsi'],
            'H1_ADX': self.last_h1_values['adx']
        }
        
        # Update normalization buffers and calculate Z-scores
        normalized_features = {}
        for key, value in current_indicators.items():
            self.norm_buffers[key].append(value)
            
            if len(self.norm_buffers[key]) >= 2:
                buffer_array = np.array(self.norm_buffers[key])
                mean_val = np.mean(buffer_array)
                std_val = np.std(buffer_array)
                
                if std_val > 1e-8:
                    normalized_features[f'{key}_Z'] = (value - mean_val) / std_val
                else:
                    normalized_features[f'{key}_Z'] = 0.0
            else:
                normalized_features[f'{key}_Z'] = 0.0
        
        # Time-based features  
        hour_sin = np.sin(2 * np.pi * timestamp.hour / 23.0)
        hour_cos = np.cos(2 * np.pi * timestamp.hour / 23.0)
        day_sin = np.sin(2 * np.pi * timestamp.dayofweek / 6.0)
        day_cos = np.cos(2 * np.pi * timestamp.dayofweek / 6.0)
        
        # Combine all features
        feature_dict = {
            'Close': c,
            'ATR': self.m5_atr[-1] if self.m5_atr else 0.001,
            'H1_Close': self.last_h1_values['close'],
            'H1_EMA': self.last_h1_values['ema'],
            'hour_sin': hour_sin,
            'hour_cos': hour_cos,
            'day_sin': day_sin,
            'day_cos': day_cos,
            **current_indicators,
            **normalized_features
        }
        
        return feature_dict

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("--- Starting OPTIMIZED Realistic Data Preprocessing ---")
    
    start_date = config.START_DATE 
    end_date = config.END_DATE

    output_dir = "processed_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    ticker = config.TICKERS[0]
    output_path = os.path.join(output_dir, f"{ticker}_processed.parquet")
    
    logging.info(f"Fetching raw data for {ticker} from {start_date} to {end_date}...")
    
    # Initialize MT5 connection
    if not mt5.initialize():
        logging.critical("MT5 initialization failed. Please ensure the terminal is running.")
        return

    try:
        raw_m5_df, raw_h1_df = data_handler.fetch_data(ticker, start_date, end_date)
        if raw_m5_df is None:
            logging.error(f"Could not fetch data for {ticker}. Exiting.")
            return
    finally:
        mt5.shutdown()
    
    logging.info(f"Data fetched: {len(raw_m5_df)} M5 candles, {len(raw_h1_df)} H1 candles")
    
    # Initialize streaming calculator
    stream_calc = StreamingIndicators(lookback_window=config.INDICATOR_LOOKBACK_CANDLES)
    
    # Pre-populate with initial data for stable indicator calculation
    logging.info("Pre-populating indicators...")
    warmup_size = config.INDICATOR_LOOKBACK_CANDLES
    
    for i in tqdm(range(min(warmup_size, len(raw_m5_df))), desc="Warmup"):
        candle_row = raw_m5_df.iloc[i]
        stream_calc.update_m5(candle_row)
        
        # Update H1 when timestamp aligns (every 12 M5 candles for M5->H1)
        if i % 12 == 0 and i // 12 < len(raw_h1_df):
            stream_calc.update_h1(raw_h1_df.iloc[i // 12])
    
    # Process remaining data and collect features
    logging.info("Generating realistic features...")
    processed_data = []
    
    # Use batch processing to reduce memory allocations
    batch_size = 1000
    start_idx = warmup_size
    
    while start_idx < len(raw_m5_df):
        end_idx = min(start_idx + batch_size, len(raw_m5_df))
        batch_data = []
        
        for i in tqdm(range(start_idx, end_idx), desc=f"Batch {start_idx//batch_size + 1}", leave=False):
            candle_row = raw_m5_df.iloc[i]
            timestamp = raw_m5_df.index[i]
            
            # Update indicators  
            m5_indicators = stream_calc.update_m5(candle_row)
            
            # Update H1 data when available
            h1_idx = i // 12
            if i % 12 == 0 and h1_idx < len(raw_h1_df):
                stream_calc.update_h1(raw_h1_df.iloc[h1_idx])
            
            # Get complete feature set
            if m5_indicators is not None:
                features = stream_calc.get_current_features(candle_row, timestamp)
                
                if features is not None:
                    # Add OHLCV data
                    features.update({
                        'Open': candle_row['Open'],
                        'High': candle_row['High'], 
                        'Low': candle_row['Low'],
                        'Volume': candle_row['Volume']
                    })
                    
                    batch_data.append(features)
        
        # Convert batch to DataFrame and append
        if batch_data:
            batch_df = pd.DataFrame(batch_data, index=raw_m5_df.index[start_idx:start_idx+len(batch_data)])
            processed_data.append(batch_df)
            
            # Periodic memory cleanup
            if len(processed_data) % 5 == 0:
                logging.info(f"Processed {len(processed_data) * batch_size} candles, merging batches...")
                # Combine batches to free memory
                combined_df = pd.concat(processed_data, ignore_index=False)
                processed_data = [combined_df]
        
        start_idx = end_idx
    
    # Final combination
    if processed_data:
        logging.info("Combining final results...")
        final_df = pd.concat(processed_data, ignore_index=False)
        
        # Clean up any remaining NaN values
        final_df = final_df.replace([np.inf, -np.inf], np.nan)
        final_df = final_df.dropna()
        
        # Ensure correct column order
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume', 'ATR', 'RSI', 'MACD_line', 'MACD_signal', 'MACD_hist',
            'ADX', 'stoch_k', 'stoch_d', 'upper_wick', 'lower_wick', 'body_size', 'H1_Close', 'H1_EMA',
            'H1_RSI', 'H1_ADX', 'RSI_Z', 'MACD_line_Z', 'MACD_hist_Z', 'ADX_Z', 'stoch_k_Z', 'stoch_d_Z',
            'upper_wick_Z', 'lower_wick_Z', 'body_size_Z', 'H1_RSI_Z', 'H1_ADX_Z', 'hour_sin', 'hour_cos',
            'day_sin', 'day_cos'
        ]
        
        # Reorder columns and fill any missing ones
        for col in feature_columns:
            if col not in final_df.columns:
                final_df[col] = 0.0
        
        final_df = final_df[feature_columns]
        
        # Save with compression
        final_df.to_parquet(output_path, compression='snappy')
        
        logging.info(f"Successfully processed {len(final_df)} realistic candles")
        logging.info(f"Final dataset saved to {output_path}")
        logging.info(f"Memory usage: {final_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    else:
        logging.error("No data was processed successfully")

if __name__ == '__main__':
    main()