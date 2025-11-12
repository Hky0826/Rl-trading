# File: live_trader.py
# Description: Live trading with FULL feature set support
# Key change: Updated feature_columns in get_live_observation to match full feature set
# =============================================================================
import logging
import time
from datetime import datetime, timedelta
import math
import numpy as np
import pandas as pd
import json
import os
import traceback
import csv
from stable_baselines3 import PPO
import config
import data_handler
import trade_executor
import MetaTrader5 as mt5
from replay_buffer import ReplayBuffer
from portfolio_tracker import PortfolioTracker
from performance_logger import PerformanceLogger

# ---------------------------------------------------------------------------
# Battery-optimized PerformanceLogger (used by BatteryOptimizedLiveTrader)
# ---------------------------------------------------------------------------
class BatteryOptimizedPerformanceLogger(PerformanceLogger):
    """Buffer equity updates and save in batches to reduce disk I/O in power-save mode."""
    def __init__(self):
        super().__init__()
        self.pending_equity_updates = []
        self.last_batch_save = datetime.now()
        if not hasattr(self, 'lock'):
            self.lock = DummyLock()

    def log_equity(self, current_equity):
        """Buffer equity updates for batch processing"""
        if getattr(config, 'POWER_SAVE_MODE', False):
            try:
                self.pending_equity_updates.append({
                    "time": datetime.now().isoformat(),
                    "equity": float(current_equity)
                })
                if len(self.pending_equity_updates) >= 50 or \
                   (datetime.now() - self.last_batch_save).total_seconds() > getattr(config, 'POWER_SAVE_EQUITY_LOG_INTERVAL_SECONDS', 300):
                    self._flush_equity_batch()
            except Exception:
                pass
        else:
            super().log_equity(current_equity)

    def _flush_equity_batch(self):
        """Flush batched equity updates"""
        try:
            with getattr(self, 'lock', DummyLock()):
                if 'equity_history' not in self.stats:
                    self.stats['equity_history'] = []
                self.stats['equity_history'].extend(self.pending_equity_updates)
                self.pending_equity_updates.clear()
                self.last_batch_save = datetime.now()
                try:
                    self.save_stats()
                except Exception:
                    pass
        except Exception:
            pass

class DummyLock:
    """Fallback no-op lock if PerformanceLogger lacks lock attribute."""
    def __enter__(self): return None
    def __exit__(self, exc_type, exc, tb): return False

# ---------------------------------------------------------------------------
# Core LiveTrader
# ---------------------------------------------------------------------------
class LiveTrader:
    def __init__(self):
        """Initialize the live trader with all components."""
        self.logger = self._setup_logging()
        self.logger.info("--- Starting Pure RL Live Trading Bot ---")

        self.mt5_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = getattr(config, 'MAX_RECONNECT_ATTEMPTS', 10)

        self.rl_model = None
        self.perf_logger = None
        self.tracker = None
        self.last_candle_times = {ticker: None for ticker in config.TICKERS}

        self.consecutive_errors = 0
        self.max_consecutive_errors = getattr(config, 'MAX_CONSECUTIVE_ERRORS', 50)

        self.status_update_interval = getattr(config, 'STATUS_UPDATE_INTERVAL', 30)
        self.last_status_update = datetime.now()

        self.wake_interval_minutes = getattr(config, 'WAKE_INTERVAL_MINUTES', 5)

        self._last_stats_save_time = datetime.min
        self._last_equity_log_time = datetime.min

        self._last_restart_attempt = None
        self._restart_backoff_seconds = 1
        self._restart_backoff_max_seconds = getattr(config, 'RESTART_BACKOFF_MAX_SECONDS', 300)

    def _setup_logging(self):
        """Setup comprehensive logging."""
        log_dir = getattr(config, 'LOG_DIR', "logs")
        os.makedirs(log_dir, exist_ok=True)

        logger = logging.getLogger('LiveTrader')
        logger.setLevel(logging.DEBUG)

        if logger.handlers:
            return logger

        fh = logging.FileHandler(
            os.path.join(log_dir, f'live_trader_{datetime.now().strftime("%Y%m%d")}.log')
        )
        fh.setLevel(logging.DEBUG)

        eh = logging.FileHandler(
            os.path.join(log_dir, f'live_trader_errors_{datetime.now().strftime("%Y%m%d")}.log')
        )
        eh.setLevel(logging.ERROR)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        eh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(eh)
        logger.addHandler(ch)

        return logger

    def connect_mt5(self):
        """Connect to MT5 with retry logic."""
        try:
            try:
                term_info = mt5.terminal_info()
            except Exception:
                term_info = None

            if term_info and getattr(term_info, 'connected', False):
                self.logger.info("Already connected to MT5")
                self.mt5_connected = True
                self.reconnect_attempts = 0
                return True

            self.logger.info("Attempting to connect to MT5...")

            if not mt5.initialize():
                self.logger.error("MT5 initialization failed")
                self.mt5_connected = False
                return False

            account_info = mt5.account_info()
            if account_info:
                self.logger.info(f"Connected to MT5 - Account: {account_info.login}, Server: {account_info.server}")
                self.mt5_connected = True
                self.reconnect_attempts = 0
                return True
            else:
                self.logger.error("MT5 connected but cannot get account info")
                self.mt5_connected = False
                return False

        except Exception as e:
            self.logger.error(f"Error connecting to MT5: {e}")
            self.mt5_connected = False
            return False

    def reconnect_mt5(self):
        """Attempt to reconnect to MT5 with exponential backoff."""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            self.logger.critical("Max reconnection attempts reached. Exiting reconnect attempts.")
            return False

        self.reconnect_attempts += 1
        wait_time = min(2 ** self.reconnect_attempts, 300)

        self.logger.warning(f"Reconnection attempt {self.reconnect_attempts}/{self.max_reconnect_attempts} in {wait_time} seconds...")
        time.sleep(wait_time)

        try:
            mt5.shutdown()
        except Exception:
            pass

        return self.connect_mt5()

    def smart_mt5_check(self):
        """Less frequent MT5 connection checks (default behavior)."""
        if not hasattr(self, '_last_mt5_check'):
            self._last_mt5_check = datetime.now()

        check_interval = getattr(config, 'MT5_CHECK_INTERVAL', 10)

        if (datetime.now() - self._last_mt5_check).total_seconds() > check_interval:
            self._last_mt5_check = datetime.now()
            try:
                return not self.mt5_connected or not mt5.terminal_info()
            except Exception:
                return True
        return False

    def load_model(self):
        """Load the RL model with error handling."""
        try:
            model_path = os.path.join(config.RL_MODEL_DIR, f"{config.TICKERS[0]}.zip")

            if not os.path.exists(model_path):
                alt_path = os.path.join(config.RL_MODEL_DIR, f"rl_agent_{config.TICKERS[0]}.zip")
                if os.path.exists(alt_path):
                    model_path = alt_path

            if not os.path.exists(model_path):
                self.logger.critical(f"Model file not found at {model_path}")
                return False

            self.rl_model = PPO.load(model_path, device="cpu")
            self.logger.info(f"Successfully loaded RL model from {model_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def initialize_components(self):
        """Initialize all tracking components."""
        try:
            if self.perf_logger is None:
                if getattr(config, 'POWER_SAVE_MODE', False):
                    self.perf_logger = BatteryOptimizedPerformanceLogger()
                else:
                    self.perf_logger = PerformanceLogger()
            if self.tracker is None:
                self.tracker = PortfolioTracker()
            self.logger.info("Successfully initialized performance logger and portfolio tracker")
            return True
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def update_status_file(self, status, account_info=None, positions=None):
        """Update status file with error handling. Create file if missing."""
        try:
            os.makedirs(os.path.dirname(config.GUI_STATUS_FILE), exist_ok=True)

            if not os.path.exists(config.GUI_STATUS_FILE):
                base_data = {
                    'status': 'Initializing',
                    'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'mt5_connected': False,
                    'consecutive_errors': 0,
                    'account_info': {},
                    'positions': [],
                    'performance_summary': {}
                }
                with open(config.GUI_STATUS_FILE, 'w') as f:
                    json.dump(base_data, f, indent=4)

            data = {
                'status': status,
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'mt5_connected': self.mt5_connected,
                'consecutive_errors': self.consecutive_errors,
                'account_info': {},
                'positions': []
            }

            if account_info:
                data['account_info'] = {
                    'login': getattr(account_info, 'login', None),
                    'server': getattr(account_info, 'server', None),
                    'currency': getattr(account_info, 'currency', None),
                    'equity': f"{getattr(account_info, 'equity', 0):,.2f}",
                    'balance': f"{getattr(account_info, 'balance', 0):,.2f}",
                    'profit': f"{getattr(account_info, 'profit', 0):,.2f}",
                    'margin': f"{getattr(account_info, 'margin', 0):,.2f}" if hasattr(account_info, 'margin') else "0.00",
                    'free_margin': f"{getattr(account_info, 'margin_free', 0):,.2f}" if hasattr(account_info, 'margin_free') else "0.00"
                }

            if positions:
                for pos in positions:
                    try:
                        if getattr(pos, 'magic', None) == config.MAGIC_NUMBER:
                            data['positions'].append({
                                'ticket': getattr(pos, 'ticket', None),
                                'symbol': getattr(pos, 'symbol', None),
                                'type': "BUY" if getattr(pos, 'type', None) == mt5.ORDER_TYPE_BUY else "SELL",
                                'volume': getattr(pos, 'volume', None),
                                'open_price': getattr(pos, 'price_open', None),
                                'current_price': getattr(pos, 'price_current', None),
                                'sl': getattr(pos, 'sl', None),
                                'tp': getattr(pos, 'tp', None),
                                'profit': f"{getattr(pos, 'profit', 0):,.2f}",
                                'open_time': datetime.fromtimestamp(getattr(pos, 'time', 0)).strftime('%Y-%m-%d %H:%M:%S')
                            })
                    except Exception:
                        continue

            try:
                if self.perf_logger:
                    data['performance_summary'] = self.perf_logger.get_summary()
            except Exception:
                pass

            with open(config.GUI_STATUS_FILE, 'w') as f:
                json.dump(data, f, indent=4)

            self.last_status_update = datetime.now()

        except Exception as e:
            self.logger.error(f"Error updating status file: {e}")
            self.logger.error(traceback.format_exc())

    def get_live_observation(self, df, open_positions, account_info, live_drawdown):
        """Get observation for the model with FULL feature set - ALIGNED with environment."""
        try:
            lookback_df = df.iloc[-config.RL_LOOKBACK_WINDOW:]

            # ✅ FULL FEATURE SET - matches environment and requirement
            feature_columns = [
                'Open', 'High', 'Low', 'Close', 'Volume', 'ATR', 'RSI', 'MACD_line', 'MACD_hist',
                'ADX', 'stoch_k', 'stoch_d', 'upper_wick', 'lower_wick', 'body_size', 'H1_Close', 'H1_EMA',
                'H1_RSI', 'H1_ADX', 'RSI_Z', 'MACD_line_Z', 'MACD_hist_Z', 'ADX_Z', 'stoch_k_Z', 'stoch_d_Z',
                'upper_wick_Z', 'lower_wick_Z', 'body_size_Z', 'H1_RSI_Z', 'H1_ADX_Z', 'hour_sin', 'hour_cos',
                'day_sin', 'day_cos'
            ]

            # Validate all columns exist
            missing_cols = [col for col in feature_columns if col not in lookback_df.columns]
            if missing_cols:
                self.logger.error(f"Missing columns in dataframe: {missing_cols}")
                return None

            features_df = lookback_df[feature_columns]

            # Fill NaNs exactly like environment
            if features_df.isnull().any().any():
                self.logger.warning("NaN values found in features, filling with zeros")
                features_df = features_df.fillna(0)

            # Market features - exactly like environment format
            market_features = features_df.values.astype(np.float32)

            # Return dictionary observation - EXACTLY like environment
            observation = {
                'market_features': market_features
            }

            # Validate shapes
            expected_market_shape = (config.RL_LOOKBACK_WINDOW, len(feature_columns))

            if observation['market_features'].shape != expected_market_shape:
                self.logger.error(f"Invalid market_features shape: {observation['market_features'].shape}, expected {expected_market_shape}")
                return None

            return observation

        except Exception as e:
            self.logger.error(f"Error creating observation: {e}")
            self.logger.error(traceback.format_exc())
            return None

    def process_closed_deals(self, account_info):
        """Process recently closed deals with error handling."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)

            deals = mt5.history_deals_get(start_date, end_date)

            if not deals:
                return

            for deal in deals:
                try:
                    if getattr(deal, 'magic', None) == config.MAGIC_NUMBER and getattr(deal, 'entry', None) == 1:
                        if deal.ticket not in getattr(self.perf_logger, 'session_processed_deals', set()):
                            params = self.tracker.remove_open_trade(getattr(deal, 'order', None))

                            if params:
                                self.perf_logger.log_trade(
                                    deal,
                                    params.get('risk_level'),
                                    params.get('rr_profile_index')
                                )
                                self.logger.info(f"Processed closed deal {deal.ticket}: P&L = {getattr(deal, 'profit', 0):.2f}")
                except Exception as e:
                    self.logger.error(f"Error processing deal {getattr(deal, 'ticket', 'unknown')}: {e}")
                    self.logger.error(traceback.format_exc())

        except Exception as e:
            self.logger.error(f"Error getting deal history: {e}")
            self.logger.error(traceback.format_exc())

    def cleanup_stale_positions(self):
        """Clean up stale positions from tracker."""
        try:
            all_positions = mt5.positions_get()
            if all_positions:
                active_tickets = [p.ticket for p in all_positions if getattr(p, 'magic', None) == config.MAGIC_NUMBER]
                self.tracker.cleanup_stale_trades(active_tickets)
        except Exception as e:
            self.logger.error(f"Error cleaning up stale positions: {e}")
            self.logger.error(traceback.format_exc())

    def save_candle_data(self, observation, save_path="live_candle_data.csv", append=True):
        """Save the current observation (market features) to a CSV file."""
        try:
            if observation is None or 'market_features' not in observation:
                self.logger.warning("No valid observation to save.")
                return

            market_features = observation['market_features']

            # Full feature column names
            df = pd.DataFrame(
                market_features,
                columns=[
                    'Open', 'High', 'Low', 'Close', 'Volume', 'ATR', 'RSI', 'MACD_line', 'MACD_signal', 'MACD_hist',
                    'ADX', 'stoch_k', 'stoch_d', 'upper_wick', 'lower_wick', 'body_size', 'H1_Close', 'H1_EMA',
                    'H1_RSI', 'H1_ADX', 'RSI_Z', 'MACD_line_Z', 'MACD_hist_Z', 'ADX_Z', 'stoch_k_Z', 'stoch_d_Z',
                    'upper_wick_Z', 'lower_wick_Z', 'body_size_Z', 'H1_RSI_Z', 'H1_ADX_Z', 'hour_sin', 'hour_cos',
                    'day_sin', 'day_cos'
                ]
            )

            # Add full timestamp
            current_time = pd.Timestamp.now()
            df['timestamp'] = current_time

            # Also add separate date + time columns
            df['date'] = current_time.date()
            df['time'] = current_time.time()

            # Save to file
            if append and os.path.exists(save_path):
                df.to_csv(save_path, mode='a', header=False, index=False)
            else:
                df.to_csv(save_path, index=False)

            self.logger.info(f"Saved candle data to {save_path} ({len(df)} rows).")

        except Exception as e:
            self.logger.error(f"Error saving candle data: {e}")
            self.logger.error(traceback.format_exc())


    def handle_new_candle(self, ticker, live_data, positions_for_symbol, account_info, live_drawdown):
        """Process a new candle for trading decisions."""
        try:
            self.logger.info(f"New {config.PRIMARY_TIMEFRAME_STRING} candle for {ticker}. Analyzing...")

            current_obs = self.get_live_observation(live_data, positions_for_symbol, account_info, live_drawdown)

            if current_obs is None:
                self.logger.error("Failed to create observation, skipping this candle")
                return

            action, _ = self.rl_model.predict(current_obs, deterministic=True)

            try:
                action_type, risk_index, rr_profile_index = action
            except Exception:
                self.logger.error(f"Unexpected model action format: {action}")
                return

            self.logger.debug(f"Model decision: action_type={action_type}, risk={risk_index}, rr={rr_profile_index}")
            
            if action_type in [1, 2]:
                risk_percent = config.RISK_LEVELS[int(risk_index)]
                latest_candle = live_data.iloc[-2]
                self.save_candle_data(latest_candle)
                atr = latest_candle['ATR']
                current_price = latest_candle['Close']

                if np.isnan(atr) or atr <= 0:
                    self.logger.warning(f"Invalid ATR value: {atr}, skipping trade")
                    return

                sl_multiplier, tp_multiplier = config.RR_PROFILES[int(rr_profile_index)]

                if action_type == 1:
                    sl = current_price - (atr * sl_multiplier)
                    tp = current_price + (atr * tp_multiplier)
                    trade_type = 'LONG'
                else:
                    sl = current_price + (atr * sl_multiplier)
                    tp = current_price - (atr * tp_multiplier)
                    trade_type = 'SHORT'

                lot_size = trade_executor.calculate_lot_size(ticker, sl, risk_percent)

                if lot_size and lot_size > 0:
                    self.logger.info(f"Placing {trade_type} trade: lot={lot_size:.2f}, SL={sl:.5f}, TP={tp:.5f}")

                    result = trade_executor.place_trade(trade_type, ticker, lot_size, sl, tp)

                    if result and hasattr(result, 'order'):
                        params_to_save = {
                            'risk_level': float(risk_percent),
                            'rr_profile_index': int(rr_profile_index)
                        }
                        self.tracker.add_open_trade(result.order, params_to_save)
                        self.logger.info(f"Trade placed successfully: Order {result.order}")
                    else:
                        self.logger.warning("Trade execution failed")
                else:
                    self.logger.warning(f"Invalid lot size calculated: {lot_size}")
            else:
                self.logger.debug("Model decision: HOLD")

        except Exception as e:
            self.logger.error(f"Error handling new candle for {ticker}: {e}")
            self.logger.error(traceback.format_exc())
            self.consecutive_errors += 1

    def _seconds_until_next_aligned_tick(self):
        """Return seconds until the next clock-aligned wake based on wake_interval_minutes."""
        try:
            now = datetime.now()
            interval = max(1, int(self.wake_interval_minutes))
            minute = now.minute
            second = now.second
            
            if minute % interval == 0 and second == 0:
                return 0
            
            mins_since_multiple = minute % interval
            mins_to_add = (interval - mins_since_multiple) % interval
            
            if mins_to_add == 0:
                mins_to_add = interval
            
            next_tick = (now.replace(second=0, microsecond=0) + timedelta(minutes=mins_to_add))
            return max(0.0, (next_tick - now).total_seconds())
        except Exception as e:
            self.logger.error(f"Error computing aligned tick: {e}")
            return getattr(config, 'DEFAULT_SLEEP_SECONDS', 15)

    def calculate_sleep_duration(self):
        """Calculate how long to sleep until next wake aligned to WAKE_INTERVAL_MINUTES."""
        try:
            secs = self._seconds_until_next_aligned_tick()
            
            if secs == 0:
                return 0.5

            base_min = 1.0

            if getattr(config, 'POWER_SAVE_MODE', False):
                min_sleep = getattr(config, 'POWER_SAVE_MIN_SLEEP_SECONDS', 30)
                secs = max(secs, min_sleep)

            return max(secs, base_min)

        except Exception as e:
            self.logger.error(f"Error calculating sleep duration: {e}")
            self.logger.error(traceback.format_exc())
            return getattr(config, 'DEFAULT_SLEEP_SECONDS', 15)

    def _attempt_restart(self, reason="unspecified"):
        """Try to restart core components instead of full exit. Backoff between attempts."""
        try:
            now = datetime.now()
            if self._last_restart_attempt:
                since_last = (now - self._last_restart_attempt).total_seconds()
                if since_last < self._restart_backoff_seconds:
                    time.sleep(self._restart_backoff_seconds - since_last)

            self._last_restart_attempt = now
            self.logger.warning(f"Attempting restart due to: {reason}. Backoff: {self._restart_backoff_seconds}s")

            try:
                if self.perf_logger:
                    self.perf_logger.save_stats()
            except Exception:
                pass
            try:
                if self.tracker:
                    self.tracker.force_save()
            except Exception:
                pass

            try:
                if self.mt5_connected:
                    mt5.shutdown()
            except Exception:
                pass
            self.mt5_connected = False
            self.reconnect_attempts = 0

            if not self.connect_mt5():
                self.logger.error("Restart attempt: Failed to connect to MT5")
                self._restart_backoff_seconds = min(self._restart_backoff_seconds * 2, self._restart_backoff_max_seconds)
                return False

            if self.rl_model is None:
                if not self.load_model():
                    self.logger.error("Restart attempt: Failed to load model")
                    self._restart_backoff_seconds = min(self._restart_backoff_seconds * 2, self._restart_backoff_max_seconds)
                    return False

            if not self.initialize_components():
                self.logger.error("Restart attempt: Failed to initialize components")
                self._restart_backoff_seconds = min(self._restart_backoff_seconds * 2, self._restart_backoff_max_seconds)
                return False

            self.consecutive_errors = 0
            self._restart_backoff_seconds = 1
            self.logger.info("Restart successful")
            return True

        except Exception as e:
            self.logger.error(f"Error during restart attempt: {e}")
            self.logger.error(traceback.format_exc())
            self._restart_backoff_seconds = min(self._restart_backoff_seconds * 2, self._restart_backoff_max_seconds)
            return False

    def run(self):
        """Main trading loop with comprehensive error handling."""
        while True:
            try:
                if not self.connect_mt5():
                    self.logger.critical("Failed to connect to MT5 at startup. Will retry with backoff.")
                    if not self._attempt_restart("initial_mt5_connect_failed"):
                        time.sleep(10)
                        continue

                if not self.load_model():
                    self.logger.critical("Failed to load model at startup. Will retry with backoff.")
                    if not self._attempt_restart("initial_model_load_failed"):
                        time.sleep(10)
                        continue

                if not self.initialize_components():
                    self.logger.critical("Failed to initialize components at startup. Will retry with backoff.")
                    if not self._attempt_restart("initial_init_components_failed"):
                        time.sleep(10)
                        continue

                symbol_missing = False
                for ticker in config.TICKERS:
                    symbol_info = mt5.symbol_info(ticker)
                    if symbol_info is None:
                        self.logger.critical(f"Symbol {ticker} not found in MT5. Please check symbol name.")
                        symbol_missing = True
                        break
                    if not symbol_info.visible:
                        self.logger.info(f"Subscribing to {ticker}")
                        if not mt5.symbol_select(ticker, True):
                            self.logger.critical(f"Failed to subscribe to {ticker}")
                            symbol_missing = True
                            break
                if symbol_missing:
                    self._attempt_restart("symbol_missing")
                    time.sleep(10)
                    continue

                self.logger.info("=== Live Trading Bot Successfully Initialized ===")

                while True:
                    try:
                        if self.smart_mt5_check():
                            self.logger.warning("MT5 connection lost or stale, attempting to reconnect...")
                            if not self.reconnect_mt5():
                                self.logger.error("Failed to reconnect to MT5 in smart check. Attempting restart.")
                                if not self._attempt_restart("mt5_smart_check_reconnect_failed"):
                                    break

                        account_info = mt5.account_info()
                        if not account_info:
                            self.logger.error("Could not get account info")
                            time.sleep(5)
                            if not self.reconnect_mt5():
                                if not self._attempt_restart("no_account_info"):
                                    break
                            continue

                        all_positions = mt5.positions_get()

                        try:
                            now = datetime.now()
                            if getattr(config, 'POWER_SAVE_MODE', False):
                                eq_interval = getattr(config, 'POWER_SAVE_EQUITY_LOG_INTERVAL_SECONDS', 300)
                                if (now - self._last_equity_log_time).total_seconds() >= eq_interval:
                                    self.tracker.update_peak_equity(account_info.equity)
                                    try:
                                        self.perf_logger.log_equity(account_info.equity)
                                    except Exception:
                                        pass
                                    self._last_equity_log_time = now
                                else:
                                    try:
                                        self.tracker.update_peak_equity(account_info.equity)
                                    except Exception:
                                        pass
                            else:
                                self.tracker.update_peak_equity(account_info.equity)
                                try:
                                    self.perf_logger.log_equity(account_info.equity)
                                except Exception:
                                    pass
                        except Exception:
                            pass

                        live_drawdown = self.tracker.get_current_drawdown(account_info.equity)

                        try:
                            self.process_closed_deals(account_info)
                        except Exception:
                            pass

                        if datetime.now().minute % 10 == 0:
                            try:
                                self.cleanup_stale_positions()
                            except Exception:
                                pass

                        open_bot_positions = [
                            p for p in all_positions
                            if getattr(p, 'magic', None) == config.MAGIC_NUMBER
                        ] if all_positions else []

                        if len(open_bot_positions) >= config.MAX_TOTAL_POSITIONS:
                            self.logger.debug(f"Max total positions reached ({len(open_bot_positions)}) - skipping symbol processing this cycle")
                            sleep_seconds = self.calculate_sleep_duration()
                            total = sleep_seconds
                            seg = 5
                            while total > 0:
                                time.sleep(min(seg, total))
                                total -= seg
                            continue

                        for ticker in config.TICKERS:
                            try:
                                positions_for_symbol = [
                                    p for p in open_bot_positions if getattr(p, 'symbol', None) == ticker
                                ]
                                if len(positions_for_symbol) >= config.MAX_POSITIONS_PER_SYMBOL:
                                    continue

                                m5_data, h1_data = data_handler.fetch_live_data(
                                    ticker, num_candles=config.INDICATOR_LOOKBACK_CANDLES
                                )
                                if m5_data is None or len(m5_data) < 200 or h1_data is None:
                                    self.logger.warning(f"Insufficient data for {ticker}")
                                    continue
                                try:
                                    live_data = data_handler.prepare_indicators(m5_data, h1_data)
                                except Exception as e:
                                    self.logger.error(f"Error preparing indicators for {ticker}: {e}")
                                    continue

                                if live_data is None or live_data.empty:
                                    continue

                                current_candle_time = live_data.index[-1]
                                if (self.last_candle_times.get(ticker) is None or
                                        current_candle_time > self.last_candle_times[ticker]):
                                    self.last_candle_times[ticker] = current_candle_time
                                    self.handle_new_candle(
                                        ticker, live_data, positions_for_symbol, account_info, live_drawdown
                                    )

                            except Exception as e:
                                self.logger.error(f"Error processing ticker {ticker}: {e}")
                                self.logger.error(traceback.format_exc())
                                self.consecutive_errors += 1

                        if self.consecutive_errors >= self.max_consecutive_errors:
                            self.logger.critical(f"Too many consecutive errors ({self.consecutive_errors}), attempting restart")
                            if not self._attempt_restart("consecutive_errors_threshold"):
                                break

                        if self.consecutive_errors > 0:
                            self.consecutive_errors = max(0, self.consecutive_errors - 1)

                        now = datetime.now()
                        save_interval = getattr(config, 'POWER_SAVE_SAVE_STATS_INTERVAL_SECONDS', 600) if getattr(config, 'POWER_SAVE_MODE', False) else 300
                        if (now - self._last_stats_save_time).total_seconds() >= save_interval:
                            try:
                                self.perf_logger.save_stats()
                            except Exception:
                                pass
                            try:
                                self.tracker.force_save()
                            except Exception:
                                pass
                            self._last_stats_save_time = now

                        sleep_duration = self.calculate_sleep_duration()
                        self.logger.debug(f"Sleeping for {sleep_duration:.1f} seconds (until next aligned wake)")
                        remaining = sleep_duration
                        heartbeat_interval = 5
                        while remaining > 0:
                            to_sleep = min(heartbeat_interval, remaining)
                            time.sleep(to_sleep)
                            remaining -= to_sleep

                    except KeyboardInterrupt:
                        self.logger.info("KeyboardInterrupt detected. Shutting down gracefully...")
                        raise
                    except Exception as e:
                        self.logger.error(f"Error in main loop iteration: {e}")
                        self.logger.error(traceback.format_exc())
                        self.consecutive_errors += 1
                        time.sleep(5)
                        if self.consecutive_errors >= self.max_consecutive_errors:
                            self.logger.critical("Consecutive errors exceeded during iteration; breaking to outer restart loop")
                            break

                self.logger.warning("Inner loop ended; preparing to restart (outer loop iteration)")

            except KeyboardInterrupt:
                self.logger.info("KeyboardInterrupt: shutting down from outer loop")
                break
            except Exception as e:
                self.logger.error(f"Unexpected fatal error in run outer loop: {e}")
                self.logger.error(traceback.format_exc())
                if not self._attempt_restart("unexpected_fatal"):
                    time.sleep(10)
                    continue
            finally:
                self.logger.info("Performing per-iteration cleanup...")
                try:
                    if self.perf_logger:
                        self.perf_logger.save_stats()
                except Exception:
                    pass
                try:
                    if self.tracker:
                        self.tracker.force_save()
                except Exception:
                    pass
                try:
                    self.update_status_file('Stopped', mt5.account_info() if self.mt5_connected else None, None)
                except Exception:
                    pass
                if self.mt5_connected:
                    self.logger.info("Shutting down MT5 connection")
                    try:
                        mt5.shutdown()
                    except Exception:
                        pass
                self.rl_model = None
                self.perf_logger = None
                self.tracker = None
                self.mt5_connected = False
                time.sleep(2)

        self.logger.info("=== Live Trading Bot Shutdown Complete ===")


# ---------------------------------------------------------------------------
# Battery-optimized subclass
# ---------------------------------------------------------------------------
class BatteryOptimizedLiveTrader(LiveTrader):
    """LiveTrader variant with battery/power-saving features."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.power_save_enabled = getattr(config, "POWER_SAVE_MODE", False)

        if self.power_save_enabled:
            self.reduced_logging = True
            self.extended_quiet = True
            self.quiet_hours = getattr(config, "QUIET_HOURS", [(22, 6)])
        else:
            self.reduced_logging = False
            self.extended_quiet = False
            self.quiet_hours = []

    def _log(self, message):
        """Conditional logging based on power save mode"""
        if self.power_save_enabled and self.reduced_logging:
            print(f"[LOG] {message}")
        else:
            logging.info(message)

    def _maybe_sleep(self):
        if self.power_save_enabled and self.extended_quiet:
            now = datetime.utcnow().hour
            for start, end in self.quiet_hours:
                if start < end:
                    in_quiet = start <= now < end
                else:
                    in_quiet = (now >= start or now < end)

                if in_quiet:
                    self._log("Quiet hours active, extending sleep...")
                    time.sleep(2)
                    break

    def calculate_sleep_duration(self):
        """Calculate sleep duration with power optimization"""
        base_sleep = super().calculate_sleep_duration()

        if self.power_save_enabled:
            current_hour = datetime.utcnow().hour

            if self.extended_quiet and self.quiet_hours:
                for start_hour, end_hour in self.quiet_hours:
                    if start_hour <= end_hour:
                        in_range = (start_hour <= current_hour <= end_hour)
                    else:
                        in_range = (current_hour >= start_hour or current_hour <= end_hour)
                    if in_range:
                        return min(base_sleep * 2, 60)

            return max(base_sleep, getattr(config, 'POWER_SAVE_MIN_SLEEP_SECONDS', 30))

        return base_sleep

    def update_status_file(self, status, account_info=None, positions=None):
        """Update status file with power optimization (reduced frequency)."""
        if self.power_save_enabled and self.reduced_logging:
            if (datetime.now() - self.last_status_update).total_seconds() < 120:
                return
        super().update_status_file(status, account_info, positions)

    def _observation_cache_valid(self, df):
        """Return True if the cached observation is valid for the provided df."""
        if not hasattr(self, '_cached_observation'):
            return False
        if not hasattr(self, '_last_cache_time'):
            return False
        if (datetime.now() - self._last_cache_time).total_seconds() > getattr(config, 'OBS_CACHE_TTL_SECONDS', 10):
            return False
        try:
            last_cached_index = getattr(self, '_cached_last_index', None)
            last_df_index = df.index[-1]
            return last_cached_index == last_df_index
        except Exception:
            return False

    def get_live_observation(self, df, open_positions, account_info, live_drawdown):
        """Optimized observation creation with caching to save CPU."""
        try:
            if self.power_save_enabled and hasattr(self, '_cached_observation'):
                if self._observation_cache_valid(df):
                    return self._cached_observation

            observation = super().get_live_observation(df, open_positions, account_info, live_drawdown)

            if self.power_save_enabled and observation is not None:
                self._cached_observation = observation
                self._last_cache_time = datetime.now()
                try:
                    self._cached_last_index = df.index[-1]
                except Exception:
                    self._cached_last_index = None

            return observation
        except Exception as e:
            self.logger.error(f"Error creating optimized observation: {e}")
            self.logger.error(traceback.format_exc())
            return None

    def smart_mt5_check(self):
        """Less frequent MT5 connection checks in power-save mode."""
        if not hasattr(self, '_last_mt5_check'):
            self._last_mt5_check = datetime.now()

        check_interval = 60 if self.power_save_enabled else getattr(config, 'MT5_CHECK_INTERVAL', 10)

        if (datetime.now() - self._last_mt5_check).total_seconds() > check_interval:
            self._last_mt5_check = datetime.now()
            try:
                return not self.mt5_connected or not mt5.terminal_info()
            except Exception:
                return True
        return False


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main_live():
    """Entry point for the live trader. Chooses optimized or base class based on config."""
    if getattr(config, 'POWER_SAVE_MODE', False):
        trader = BatteryOptimizedLiveTrader()
    else:
        trader = LiveTrader()
    trader.run()

if __name__ == '__main__':
    main_live()