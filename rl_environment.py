import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from collections import deque
import config
import logging
import math

_logger = logging.getLogger(__name__)

# Small constants to prevent divides by zero and uncontrolled growth
_EPS = 1e-9
_MAX_POSITION_SIZE = 1e8
_MAX_PNL = 1e12
_MAX_OBS_VALUE = 1e9

def safe_div(numer, denom, default=0.0):
    """Safe division that avoids divide-by-zero and NaN/Inf results."""
    if numer == 0:
        return 0.0
    try:
        denom_safe = denom if denom is not None else 0.0
        if denom_safe == 0 or np.isnan(denom_safe) or np.isinf(denom_safe):
            return default
        res = numer / denom_safe
        if np.isnan(res) or np.isinf(res):
            return default
        return res
    except Exception:
        return default

def safe_value(x, default=0.0, clip_min=-_MAX_OBS_VALUE, clip_max=_MAX_OBS_VALUE):
    """Replace NaN/Inf with default and clip extreme values."""
    if x is None:
        return float(default)
    if np.isnan(x) or np.isinf(x):
        return float(default)
    return float(np.clip(x, clip_min, clip_max))


class TradingEnv(gym.Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df.copy()
        self.lookback_window = config.RL_LOOKBACK_WINDOW

        # FULL FEATURE SET - matches your requirement
        self.feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume', 'ATR', 'RSI', 'MACD_line', 'MACD_hist',
            'ADX', 'stoch_k', 'stoch_d', 'upper_wick', 'lower_wick', 'body_size', 'H1_Close', 'H1_EMA',
            'H1_RSI', 'H1_ADX', 'RSI_Z', 'MACD_line_Z', 'MACD_hist_Z', 'ADX_Z', 'stoch_k_Z', 'stoch_d_Z',
            'upper_wick_Z', 'lower_wick_Z', 'body_size_Z', 'H1_RSI_Z', 'H1_ADX_Z', 'hour_sin', 'hour_cos',
            'day_sin', 'day_cos'
        ]

        # Validate all columns exist in dataframe
        missing_cols = [col for col in self.feature_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in dataframe: {missing_cols}")

        # Pre-compute and cache arrays for better performance
        self.features_array = self.df[self.feature_columns].astype(np.float64).values
        self.open_prices = self.df['Open'].astype(np.float64).values
        self.high_prices = self.df['High'].astype(np.float64).values
        self.low_prices = self.df['Low'].astype(np.float64).values
        self.close_prices = self.df['Close'].astype(np.float64).values
        self.atr_values = self.df['ATR'].astype(np.float64).values

        # Action space: (hold,buy,sell) x risk x rr_profile
        self.action_space = spaces.MultiDiscrete([
            3,  # Hold, Buy, Sell
            len(config.RISK_LEVELS),
            len(config.RR_PROFILES)
        ])

        market_features_shape = (self.lookback_window, len(self.feature_columns))

        self.observation_space = spaces.Dict({
            'market_features': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=market_features_shape,
                dtype=np.float32
            )  
        })

        self.market_obs_buffer = np.zeros(market_features_shape, dtype=np.float64)

        self.activity_log = deque(maxlen=config.ACTIVITY_MEMORY_WINDOW)
        self.action_usage_counts = {}
        self.recent_returns = deque(maxlen=self.lookback_window)

        # Efficient tracking for MORL rewards
        self.win_history = deque(maxlen=config.REWARD_METRIC_WINDOW)
        self.rr_history = deque(maxlen=config.REWARD_METRIC_WINDOW)
        self.risk_history = deque(maxlen=config.REWARD_METRIC_WINDOW)

        self._last_floating_pnl = 0.0
        self._last_current_equity = config.INITIAL_EQUITY

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.lookback_window
        self.equity = float(config.INITIAL_EQUITY)
        self.peak_equity = float(config.INITIAL_EQUITY)
        self.open_positions = []
        self.consecutive_holds = 0
        self.action_usage_counts.clear()
        self.recent_returns.clear()
        self._last_floating_pnl = 0.0
        self._last_current_equity = float(config.INITIAL_EQUITY)
        
        # Clear histories on reset
        self.win_history.clear()
        self.rr_history.clear()
        self.risk_history.clear()

        self._just_opened_trade = False
        obs = self._next_observation()
        return obs, {}

    def _calculate_floating_pnl(self):
        if not self.open_positions:
            self._last_floating_pnl = 0.0
            return 0.0

        idx = max(0, min(len(self.close_prices) - 1, int(self.current_step - 1)))
        current_price = safe_value(self.close_prices[idx], default=self.close_prices[idx] if len(self.close_prices) else 0.0)

        floating_pnl = 0.0
        for p in self.open_positions:
            price = safe_value(p.get('price', 0.0))
            size = safe_value(p.get('size', 0.0), default=0.0, clip_min=0.0, clip_max=_MAX_POSITION_SIZE)
            if size <= 0:
                continue
            if p.get('type') == 'LONG':
                pnl = (current_price - price) * size
            else:
                pnl = (price - current_price) * size
            floating_pnl += float(np.clip(pnl, -_MAX_PNL, _MAX_PNL))

        floating_pnl = float(np.clip(floating_pnl, -_MAX_PNL, _MAX_PNL))
        self._last_floating_pnl = floating_pnl
        return floating_pnl

    def _calculate_reward(self, total_pnl_closed_this_step=0, num_closed_trades=0, action=None):
        """
        Scalarization-Based MORL Reward Calculation.
        Combines multiple objectives into a single scalar using adjustable weights.
        """
        # 1. PnL Component (Normalized)
        pnl_norm = safe_div(total_pnl_closed_this_step, max(self._last_current_equity, 1.0), default=0.0)
        r_pnl = safe_value(pnl_norm, default=0.0, clip_min=-1.0, clip_max=1.0)

        # 2. Drawdown Component (Continuous)
        current_equity = safe_value(self._last_current_equity, default=self.equity)
        if self.peak_equity <= 0:
            drawdown = 0.0
        else:
            drawdown = max(0.0, safe_div((self.peak_equity - current_equity), self.peak_equity, default=0.0))
        r_drawdown = -(drawdown ** 2) # Penalty

        # 3. Metrics Components (Only applied when trades close to avoid living bonus)
        r_winrate = 0.0
        r_avg_rr = 0.0
        r_avg_risk = 0.0

        if num_closed_trades > 0:
            # Win Rate
            if len(self.win_history) > 0:
                r_winrate = sum(self.win_history) / len(self.win_history)
            
            # Average R:R
            if len(self.rr_history) > 0:
                r_avg_rr = sum(self.rr_history) / len(self.rr_history)
            
            # Average Risk
            if len(self.risk_history) > 0:
                r_avg_risk = sum(self.risk_history) / len(self.risk_history)

        # Combine with Weights
        weights = config.REWARD_WEIGHTS
        
        reward = (
            weights['pnl'] * r_pnl +
            weights['drawdown'] * r_drawdown +
            weights['winrate'] * r_winrate +
            weights['avg_rr'] * r_avg_rr +
            weights['avg_risk'] * r_avg_risk
        )

        return safe_value(reward, default=0.0, clip_min=-1e9, clip_max=1e9)

    def _update_portfolio(self):
        """
        Update portfolio by checking if SL or TP hit on existing positions.
        
        CRITICAL: Positions are ONLY closed when SL or TP is hit.
        No other exit conditions exist (no time-based exits, no manual closes, etc.)
        """
        if not self.open_positions:
            return 0.0, 0

        total_pnl = 0.0
        positions_to_close = []

        idx = max(0, min(len(self.close_prices) - 1, int(self.current_step - 1)))
        current_price = safe_value(self.close_prices[idx], default=0.0)

        for position in list(self.open_positions):
            sl = safe_value(position.get('sl', np.nan), default=np.nan)
            tp = safe_value(position.get('tp', np.nan), default=np.nan)
            price = safe_value(position.get('price', 0.0), default=0.0)
            size = safe_value(position.get('size', 0.0), default=0.0, clip_min=0.0, clip_max=_MAX_POSITION_SIZE)

            if size <= 0:
                positions_to_close.append(position)
                continue

            # Check ONLY for SL/TP hits - no other exit conditions
            # Data is assumed to be BID prices
            
            if position.get('type') == 'LONG':
                # Long Exit: Sell at Bid
                # Check SL hit (Bid <= SL)
                if not np.isnan(sl) and self.low_prices[idx] <= sl + _EPS:
                    close_price = sl
                    pnl = (close_price - price) * size
                    total_pnl += pnl
                    positions_to_close.append(position)
                    denom = price * size
                    pct = safe_div(pnl, denom, default=0.0)
                    self.recent_returns.append(safe_value(pct))
                # Check TP hit (Bid >= TP)
                elif not np.isnan(tp) and self.high_prices[idx] >= tp - _EPS:
                    close_price = tp
                    pnl = (close_price - price) * size
                    total_pnl += pnl
                    positions_to_close.append(position)
                    denom = price * size
                    pct = safe_div(pnl, denom, default=0.0)
                    self.recent_returns.append(safe_value(pct))
                
            else:  # SHORT
                # Short Exit: Buy at Ask (Bid + Spread)
                # Ask High = High + Spread
                # Ask Low = Low + Spread
                
                spread_cost = config.SPREAD_PIPS * config.PIP_VALUE
                ask_high = self.high_prices[idx] + spread_cost
                ask_low = self.low_prices[idx] + spread_cost
                
                # Check SL hit (Ask >= SL) -> (High + Spread >= SL)
                if not np.isnan(sl) and ask_high >= sl - _EPS:
                    close_price = sl
                    pnl = (price - close_price) * size
                    total_pnl += pnl
                    positions_to_close.append(position)
                    denom = price * size
                    pct = safe_div(pnl, denom, default=0.0)
                    self.recent_returns.append(safe_value(pct))
                # Check TP hit (Ask <= TP) -> (Low + Spread <= TP)
                elif not np.isnan(tp) and ask_low <= tp + _EPS:
                    close_price = tp
                    pnl = (price - close_price) * size
                    total_pnl += pnl
                    positions_to_close.append(position)
                    denom = price * size
                    pct = safe_div(pnl, denom, default=0.0)
                    self.recent_returns.append(safe_value(pct))

        total_pnl = float(np.clip(total_pnl, -_MAX_PNL, _MAX_PNL))

        # Only remove positions that hit SL or TP
        if positions_to_close:
            self.equity = float(np.clip(self.equity + total_pnl, -_MAX_PNL, _MAX_PNL))
            self.open_positions = [p for p in self.open_positions if p not in positions_to_close]
            
            # Update MORL metrics
            for p in positions_to_close:
                # Win/Loss Logic
                p_type = p.get('type')
                p_tp = safe_value(p.get('tp', 0.0))
                
                is_win = 0.0
                if p_type == 'LONG':
                    if not np.isnan(p_tp) and current_price >= p_tp - _EPS:
                        is_win = 1.0
                else: # SHORT
                    if not np.isnan(p_tp) and current_price <= p_tp + _EPS:
                        is_win = 1.0
                
                self.win_history.append(is_win)
                
                # R:R
                rr_idx = int(p.get('rr_profile_index', 0))
                if 0 <= rr_idx < len(config.RR_PROFILES):
                    sl_mult, tp_mult = config.RR_PROFILES[rr_idx]
                    rr_val = safe_div(tp_mult, sl_mult, default=0.0)
                    self.rr_history.append(rr_val)
                
                # Risk
                risk_idx = int(p.get('risk_level', 0))
                if 0 <= risk_idx < len(config.RISK_LEVELS):
                    risk_val = config.RISK_LEVELS[risk_idx]
                    self.risk_history.append(risk_val)

        return total_pnl, len(positions_to_close)

    def _execute_action(self, action_type, risk_index, rr_profile_index):
        """
        Execute trading action on CURRENT candle (realistic timing).
        
        Key behavior:
        - Observation was created from candles 0 to current_step-1 (completed)
        - Action is executed on candle at current_step (forming/current candle)
        - In live trading: you observe completed candles, then execute on next candle
        - Uses current_step-1 for price data since current_step hasn't fully formed yet
        """
        try:
            action_type = int(action_type)
        except Exception:
            action_type = 0
        risk_index = int(np.clip(int(risk_index), 0, max(0, len(config.RISK_LEVELS) - 1)))
        rr_profile_index = int(np.clip(int(rr_profile_index), 0, max(0, len(config.RR_PROFILES) - 1)))

        if (len(self.open_positions) >= config.MAX_POSITIONS_PER_SYMBOL or action_type == 0):
            if action_type == 0:
                self.consecutive_holds += 1
            return

        self.consecutive_holds = 0
        
        # Use the last COMPLETED candle for ATR reference (prediction)
        # But execute on the CURRENT candle Open (execution)
        idx_prev = max(0, min(len(self.close_prices) - 1, int(self.current_step - 1)))
        idx_curr = max(0, min(len(self.open_prices) - 1, int(self.current_step)))
        
        # Execution Price is OPEN of current candle
        current_open_price = safe_value(self.open_prices[idx_curr], default=0.0)
        
        # ATR from previous candle (for SL/TP distance)
        atr = safe_value(self.atr_values[idx_prev], default=np.nan)

        if np.isnan(atr) or atr <= 0:
            return

        risk_percent = safe_value(
            config.RISK_LEVELS[risk_index],
            default=min(config.RISK_LEVELS),
            clip_min=min(config.RISK_LEVELS),
            clip_max=max(config.RISK_LEVELS)
        )

        sl_multiplier, tp_multiplier = config.RR_PROFILES[rr_profile_index]
        spread_val = config.SPREAD_PIPS * config.PIP_VALUE

        if action_type == 1: # LONG
            # Buy at Ask = Open + Spread
            entry_price = current_open_price + spread_val
            
            # SL/TP based on Entry Price (or pattern? usually based on entry)
            sl = entry_price - (atr * sl_multiplier)
            tp = entry_price + (atr * tp_multiplier)
            pos_type = 'LONG'
            
        else: # SHORT
            # Sell at Bid = Open
            entry_price = current_open_price
            
            sl = entry_price + (atr * sl_multiplier)
            tp = entry_price - (atr * tp_multiplier)
            pos_type = 'SHORT'

        stop_distance = abs(entry_price - sl)
        stop_distance = max(stop_distance, _EPS)

        if stop_distance > 0 and self.equity > 0:
            # Lot Size Calculation
            risk_amount = self.equity * (risk_percent / 100.0)
            
            # Standard Forex Lot Calculation:
            # Risk = (Entry - SL) * Lots * Standard_Lot_Size
            # Lots = Risk / ((Entry - SL) * Standard_Lot_Size)
            
            raw_lots = risk_amount / (stop_distance * config.STANDARD_LOT_SIZE)
            
            # Round to Lot Step (0.01)
            lots = math.floor(raw_lots / config.LOT_STEP) * config.LOT_STEP
            lots = max(lots, config.MIN_LOT_SIZE)
            
            # Convert back to units for PnL calculation
            size = lots * config.STANDARD_LOT_SIZE
            
            size = safe_value(size, default=0.0, clip_min=0.0, clip_max=_MAX_POSITION_SIZE)
            
            if size > 0 and np.isfinite(size):
                new_position = {
                    'type': pos_type,
                    'price': float(entry_price),
                    'sl': float(sl),
                    'tp': float(tp),
                    'size': float(size),
                    'lots': float(lots),
                    'risk_level': int(risk_index),
                    'rr_profile_index': int(rr_profile_index)
                }
                self.open_positions.append(new_position)
                self._just_opened_trade = True
            else:
                _logger.debug("Rejected position due to invalid size: %s (raw_lots=%s)", size, raw_lots)

    def _next_observation(self):
        """
        Create observation from COMPLETED candles only (realistic timing).
        
        Key behavior:
        - At step N, we observe candles 0 to N-1 (completed candles)
        - This matches live trading where you can only see completed candles
        - The action taken will be executed on candle N (current/forming candle)
        """
        # Get completed candles up to (but not including) current step
        end = int(self.current_step)
        start = max(0, end - self.lookback_window)

        # Extract features from completed candles
        slice_arr = self.features_array[start:end]
        
        # Pad if we don't have enough history yet
        if slice_arr.shape[0] < self.lookback_window:
            pad = np.zeros((self.lookback_window - slice_arr.shape[0], slice_arr.shape[1]), dtype=np.float64)
            slice_arr = np.vstack((pad, slice_arr))

        np.copyto(self.market_obs_buffer, slice_arr[-self.lookback_window:])

        floating_pnl = self._calculate_floating_pnl()
        current_equity = safe_value(self.equity + floating_pnl, default=self.equity, clip_min=-_MAX_PNL, clip_max=_MAX_PNL)
        self._last_current_equity = current_equity

        if np.isfinite(current_equity) and current_equity > self.peak_equity:
            self.peak_equity = current_equity

        market_features_out = np.nan_to_num(self.market_obs_buffer.astype(np.float32),
                                            nan=0.0, posinf=_MAX_OBS_VALUE, neginf=-_MAX_OBS_VALUE)
        np.clip(market_features_out, -_MAX_OBS_VALUE, _MAX_OBS_VALUE, out=market_features_out)

        return {'market_features': market_features_out.copy()}

    def step(self, action):
        if self.equity <= 0:
            self.current_step += 1
            obs = self._next_observation()
            return obs, float(config.CONSTRAINT_VIOLATION_PENALTY), True, False, {}

        try:
            action_type, risk_index, rr_profile_index = action
        except Exception:
            action_type, risk_index, rr_profile_index = 0, 0, 0

        self._execute_action(action_type, risk_index, rr_profile_index)

        self.current_step += 1
        terminated = bool(self.current_step >= len(self.df))

        pnl_from_closed_trades, num_closed = self._update_portfolio()
        reward = self._calculate_reward(total_pnl_closed_this_step=pnl_from_closed_trades, num_closed_trades=num_closed, action=action)

        if not np.isfinite(self.equity):
            self.equity = safe_value(self.equity, default=0.0)
        if self.equity <= 0:
            terminated = True
            reward += float(config.CONSTRAINT_VIOLATION_PENALTY)

        obs = self._next_observation()
        obs['market_features'] = np.nan_to_num(obs['market_features'], nan=0.0, posinf=_MAX_OBS_VALUE, neginf=-_MAX_OBS_VALUE)

        return obs, float(safe_value(reward, default=0.0)), terminated, False, {}