# File: rl_environment.py
# Description: Performance-optimized trading environment with FULL feature set
# Modified to accumulate rewards episode-by-episode instead of returning per-step rewards.

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from collections import deque
import config
import logging

_logger = logging.getLogger(__name__)

# Small constants to prevent divides by zero and uncontrolled growth
_EPS = 1e-9
_MAX_POSITION_SIZE = 1e8
_MAX_PNL = 1e12
_MAX_OBS_VALUE = 1e9

def safe_div(numer, denom, default=0.0):
    """Safe division that avoids divide-by-zero and NaN/Inf results."""
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

        self._last_floating_pnl = 0.0
        self._last_current_equity = config.INITIAL_EQUITY

        # Episode-level reward accumulator (changed behavior)
        self._episode_reward_sum = 0.0

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

        # Reset episode accumulator
        self._episode_reward_sum = 0.0

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

    def _calculate_reward(self, total_pnl_closed_this_step=0, action=None):
        """
        Compute the same per-step reward as before. We will accumulate these per-step rewards
        into an episode-level sum (self._episode_reward_sum) and return the episode reward at
        the end of the episode (when terminated=True).
        """
        pnl_norm = safe_div(total_pnl_closed_this_step, max(self._last_current_equity, 1.0), default=0.0)
        return_reward = safe_value(pnl_norm, default=0.0, clip_min=-1e6, clip_max=1e6)

        if action is None or len(action) < 3:
            # If action is not provided, return the normalized pnl-based reward only
            # (keeps behavior reasonable if called without an action)
            return return_reward

        try:
            _, risk_index, rr_profile_index = action
            risk_index = int(np.clip(int(risk_index), 0, len(config.RISK_LEVELS) - 1))
            rr_profile_index = int(np.clip(int(rr_profile_index), 0, len(config.RR_PROFILES) - 1))
        except Exception:
            return return_reward

        # === Base RR profile scores (customizable) ===
        rr_scores = [1.2, 1.3, 1.4, 1.2, 1.3,
                    1.4, 1.2, 1.3, 1.4, 1.2]

        rr_score = rr_scores[rr_profile_index]
        risk_multiplier = config.RISK_LEVELS[risk_index]

        # === Determine PnL direction ===
        if total_pnl_closed_this_step > 0:
            direction = 1.0   # profitable
        elif total_pnl_closed_this_step < 0:
            direction = -1.0  # losing trade
        else:
            direction = 0.0   # no trade closed

        # === Reward composition ===
        pnl_reward = rr_score * risk_multiplier * direction

        current_equity = safe_value(self._last_current_equity, default=self.equity)
        if self.peak_equity <= 0 or np.isnan(self.peak_equity) or np.isinf(self.peak_equity):
            drawdown = 0.0
        else:
            drawdown = max(0.0, safe_div((self.peak_equity - current_equity), self.peak_equity, default=0.0))
        drawdown_penalty = -(drawdown ** 2) * 0.18 #config.DRAWDOWN_PENALTY_SCALAR

        exposure_penalty = -0.001 * len(self.open_positions)

        if getattr(self, "_just_opened_trade", False):
            entry_penalty = -0.0009
            self._just_opened_trade = False
        else:
            entry_penalty = 0.0

        reward = pnl_reward + return_reward + drawdown_penalty + exposure_penalty + entry_penalty
        reward = safe_value(reward, default=0.0, clip_min=-1e9, clip_max=1e9)

        return reward

    def _update_portfolio(self):
        """
        Update portfolio by checking if SL or TP hit on existing positions.
        CRITICAL: Positions are ONLY closed when SL or TP is hit.
        No other exit conditions exist (no time-based exits, no manual closes, etc.)
        """
        if not self.open_positions:
            return 0.0

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
            if position.get('type') == 'LONG':
                # Check SL hit (price dropped to/below SL)
                if not np.isnan(sl) and current_price <= sl + _EPS:
                    close_price = sl
                    pnl = (close_price - price) * size
                    total_pnl += pnl
                    positions_to_close.append(position)
                    denom = price * size
                    pct = safe_div(pnl, denom, default=0.0)
                    self.recent_returns.append(safe_value(pct))
                # Check TP hit (price rose to/above TP)
                elif not np.isnan(tp) and current_price >= tp - _EPS:
                    close_price = tp
                    pnl = (close_price - price) * size
                    total_pnl += pnl
                    positions_to_close.append(position)
                    denom = price * size
                    pct = safe_div(pnl, denom, default=0.0)
                    self.recent_returns.append(safe_value(pct))
                # Otherwise, position remains open
                
            else:  # SHORT
                # Check SL hit (price rose to/above SL)
                if not np.isnan(sl) and current_price >= sl - _EPS:
                    close_price = sl
                    pnl = (price - close_price) * size
                    total_pnl += pnl
                    positions_to_close.append(position)
                    denom = price * size
                    pct = safe_div(pnl, denom, default=0.0)
                    self.recent_returns.append(safe_value(pct))
                # Check TP hit (price dropped to/below TP)
                elif not np.isnan(tp) and current_price <= tp + _EPS:
                    close_price = tp
                    pnl = (price - close_price) * size
                    total_pnl += pnl
                    positions_to_close.append(position)
                    denom = price * size
                    pct = safe_div(pnl, denom, default=0.0)
                    self.recent_returns.append(safe_value(pct))
                # Otherwise, position remains open

        total_pnl = float(np.clip(total_pnl, -_MAX_PNL, _MAX_PNL))

        # Only remove positions that hit SL or TP
        if abs(total_pnl) > 0:
            self.equity = float(np.clip(self.equity + total_pnl, -_MAX_PNL, _MAX_PNL))
            self.open_positions = [p for p in self.open_positions if p not in positions_to_close]

        return total_pnl

    def _execute_action(self, action_type, risk_index, rr_profile_index):
        """
        Execute trading action on CURRENT candle (realistic timing).
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
        
        # Use the last COMPLETED candle for price/ATR reference
        idx = max(0, min(len(self.close_prices) - 1, int(self.current_step - 1)))
        current_price = safe_value(self.close_prices[idx], default=0.0)
        atr = safe_value(self.atr_values[idx], default=np.nan)

        if np.isnan(atr) or atr <= 0:
            return

        risk_percent = safe_value(
            config.RISK_LEVELS[risk_index],
            default=min(config.RISK_LEVELS),
            clip_min=min(config.RISK_LEVELS),
            clip_max=max(config.RISK_LEVELS)
        )

        sl_multiplier, tp_multiplier = config.RR_PROFILES[rr_profile_index]

        if action_type == 1:
            sl = current_price - (atr * sl_multiplier)
            tp = current_price + (atr * tp_multiplier)
            pos_type = 'LONG'
        else:
            sl = current_price + (atr * sl_multiplier)
            tp = current_price - (atr * tp_multiplier)
            pos_type = 'SHORT'

        stop_distance = abs(current_price - sl)
        stop_distance = max(stop_distance, _EPS)

        if stop_distance > 0 and self.equity > 0:
            raw_size = (self.equity * (risk_percent / 100.0)) / stop_distance
            size = safe_value(raw_size, default=0.0, clip_min=0.0, clip_max=_MAX_POSITION_SIZE)
            if size > 0 and np.isfinite(size):
                new_position = {
                    'type': pos_type,
                    'price': float(current_price),
                    'sl': float(sl),
                    'tp': float(tp),
                    'size': float(size),
                    'risk_level': int(risk_index),
                    'rr_profile_index': int(rr_profile_index)
                }
                self.open_positions.append(new_position)
                self._just_opened_trade = True
            else:
                _logger.debug("Rejected position due to invalid size: %s (raw_size=%s)", size, raw_size)

    def _next_observation(self):
        """
        Create observation from COMPLETED candles only (realistic timing).
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
        # If equity already <= 0 at the start of step, end episode and return accumulated episode reward + penalty
        if self.equity <= 0:
            terminated = True
            reward = float(safe_value(self._episode_reward_sum, default=0.0)) + float(config.CONSTRAINT_VIOLATION_PENALTY)
            obs = self._next_observation()
            return obs, reward, True, False, {}

        try:
            action_type, risk_index, rr_profile_index = action
        except Exception:
            action_type, risk_index, rr_profile_index = 0, 0, 0

        self._execute_action(action_type, risk_index, rr_profile_index)

        self.current_step += 1
        terminated = bool(self.current_step >= len(self.df))

        pnl_from_closed_trades = self._update_portfolio()

        # Calculate the per-step reward but do NOT return it immediately. Instead accumulate it
        # and only return the episode total when the episode terminates.
        reward_step = self._calculate_reward(total_pnl_closed_this_step=pnl_from_closed_trades, action=action)
        # Accumulate
        self._episode_reward_sum += float(safe_value(reward_step, default=0.0))

        # If we hit invalid equity states, handle and mark termination
        if not np.isfinite(self.equity):
            self.equity = safe_value(self.equity, default=0.0)
        if self.equity <= 0:
            terminated = True
            # apply constraint penalty to episode sum
            self._episode_reward_sum += float(config.CONSTRAINT_VIOLATION_PENALTY)

        # If episode terminated, return the accumulated episode reward. Otherwise return 0.0
        if terminated:
            reward_out = float(safe_value(self._episode_reward_sum, default=0.0))
        else:
            reward_out = 0.0

        obs = self._next_observation()
        obs['market_features'] = np.nan_to_num(obs['market_features'], nan=0.0, posinf=_MAX_OBS_VALUE, neginf=-_MAX_OBS_VALUE)

        return obs, reward_out, terminated, False, {}
