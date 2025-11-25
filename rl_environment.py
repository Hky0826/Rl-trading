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
    def __init__(self, df, phase=1):
        """
        Initialize the trading environment with curriculum learning support.
        
        Args:
            df: DataFrame with market data
            phase: Curriculum phase (1, 2, or 3)
                   Phase 1: Direction Learning (Win Rate)
                   Phase 2: R:R Strategy Learning
                   Phase 3: Full Risk Management
        """
        super(TradingEnv, self).__init__()
        self.df = df.copy()
        self.lookback_window = config.RL_LOOKBACK_WINDOW
        
        # Curriculum Phase
        self.phase = phase
        _logger.info(f"Initializing TradingEnv with Phase {self.phase}")

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

        # Action space: ALWAYS (direction, rr_profile, risk_level) - UNCHANGED across phases
        self.action_space = spaces.MultiDiscrete([
            3,  # 0=BUY, 1=SELL, 2=HOLD
            len(config.RR_PROFILES),  # RR Profile index
            len(config.RISK_LEVELS)   # Risk Level index
        ])

        market_features_shape = (self.lookback_window, len(self.feature_columns))

        # Observation space - UNCHANGED across phases
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

        # Efficient tracking for rewards
        self.win_history = deque(maxlen=config.REWARD_METRIC_WINDOW)
        self.rr_history = deque(maxlen=config.REWARD_METRIC_WINDOW)
        self.risk_history = deque(maxlen=config.REWARD_METRIC_WINDOW)

        self._last_floating_pnl = 0.0
        self._last_current_equity = config.INITIAL_EQUITY

        self.reset()

    def set_phase(self, phase_id):
        """
        Change the curriculum phase at runtime.
        
        Args:
            phase_id: Integer 1, 2, or 3
        """
        if phase_id not in [1, 2, 3]:
            raise ValueError(f"Invalid phase_id {phase_id}. Must be 1, 2, or 3.")
        
        _logger.info(f"Switching from Phase {self.phase} to Phase {phase_id}")
        self.phase = phase_id

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
        """Calculate unrealized P&L for open positions using lot-based system."""
        if not self.open_positions:
            self._last_floating_pnl = 0.0
            return 0.0

        idx = max(0, min(len(self.close_prices) - 1, int(self.current_step - 1)))
        current_price = safe_value(self.close_prices[idx], default=self.close_prices[idx] if len(self.close_prices) else 0.0)

        floating_pnl = 0.0
        for p in self.open_positions:
            entry_price = safe_value(p.get('price', 0.0))
            lot_size = safe_value(p.get('lot_size', 0.0), default=0.0, clip_min=0.0, clip_max=100.0)
            
            if lot_size <= 0:
                continue
            
            contract_size = getattr(config, 'CONTRACT_SIZE', 100000)
            
            if p.get('type') == 'LONG':
                pnl = (current_price - entry_price) * lot_size * contract_size
            else:  # SHORT
                pnl = (entry_price - current_price) * lot_size * contract_size
            
            floating_pnl += float(np.clip(pnl, -_MAX_PNL, _MAX_PNL))

        floating_pnl = float(np.clip(floating_pnl, -_MAX_PNL, _MAX_PNL))
        self._last_floating_pnl = floating_pnl
        return floating_pnl

    def _calculate_reward(self, total_pnl_closed_this_step=0, num_closed_trades=0, 
                         closed_positions=None, action=None):
        """
        Calculate reward based on curriculum phase.
        
        Phase 1: Binary win/loss reward (+1 for profit, -1 for loss)
        Phase 2: Normalized by risk (PnL / risk_amount) - teaches R-multiples
        Phase 3: Full PnL-based reward with drawdown penalties
        
        Args:
            total_pnl_closed_this_step: Total realized P&L from closed trades
            num_closed_trades: Number of trades closed this step
            closed_positions: List of position dictionaries that closed
            action: The action taken this step
        """
        if self.phase == 1:
            # PHASE 1: Direction Learning - Binary Win/Loss Reward
            if num_closed_trades == 0:
                return 0.0
            
            # Binary reward for each closed trade
            reward = 0.0
            if closed_positions:
                for pos in closed_positions:
                    pnl = pos.get('realized_pnl', 0.0)
                    if pnl > 0:
                        reward += 1.0  # Win
                    elif pnl < 0:
                        reward -= 1.0  # Loss
            
            return safe_value(reward, default=0.0, clip_min=-10.0, clip_max=10.0)
        
        elif self.phase == 2:
            # PHASE 2: R:R Strategy Learning - Reward Normalized by Risk
            if num_closed_trades == 0:
                return 0.0
            
            reward = 0.0
            if closed_positions:
                for pos in closed_positions:
                    pnl = pos.get('realized_pnl', 0.0)
                    risk_amount = pos.get('risk_amount', 1.0)  # Stored during position opening
                    
                    if risk_amount > 0:
                        # Normalize PnL by risk - this is the R-multiple
                        r_multiple = safe_div(pnl, risk_amount, default=0.0)
                        reward += r_multiple
                    else:
                        # Fallback to simple PnL
                        reward += safe_div(pnl, max(self._last_current_equity, 1.0), default=0.0)
            
            return safe_value(reward, default=0.0, clip_min=-10.0, clip_max=10.0)
        
        else:  # Phase 3
            # PHASE 3: Full Risk Management - Original reward with drawdown penalties
            
            # 1. PnL Component (Normalized)
            pnl_norm = safe_div(total_pnl_closed_this_step, max(self._last_current_equity, 1.0), default=0.0)
            r_pnl = safe_value(pnl_norm, default=0.0, clip_min=-1.0, clip_max=1.0)

            # 2. Drawdown Component (Continuous)
            current_equity = safe_value(self._last_current_equity, default=self.equity)
            if self.peak_equity <= 0:
                drawdown = 0.0
            else:
                drawdown = max(0.0, safe_div((self.peak_equity - current_equity), self.peak_equity, default=0.0))
            r_drawdown = -(drawdown ** 2)  # Penalty

            # 3. Win Rate Component
            r_winrate = 0.0
            if num_closed_trades > 0 and len(self.win_history) > 0:
                r_winrate = sum(self.win_history) / len(self.win_history)

            # 4. RR-PnL Component
            try:
                _, risk_index, rr_profile_index = action
                risk_index = int(np.clip(int(risk_index), 0, len(config.RISK_LEVELS) - 1))
                rr_profile_index = int(np.clip(int(rr_profile_index), 0, len(config.RR_PROFILES) - 1))
            except Exception:
                return 0.0

            rr_scores = [1.2, 1.3, 1.4, 1.2, 1.3, 1.4, 1.2, 1.3, 1.4, 1.2]
            rr_score = rr_scores[rr_profile_index]
            risk_multiplier = config.RISK_LEVELS[risk_index]

            if total_pnl_closed_this_step > 0:
                direction = 1.0
            elif total_pnl_closed_this_step < 0:
                direction = -1.0
            else:
                direction = 0.0

            rr_pnl = rr_score * risk_multiplier * direction

            # Combine with Weights
            weights = config.REWARD_WEIGHTS
            
            reward = (
                weights['pnl'] * r_pnl +
                weights['drawdown'] * r_drawdown +
                weights['winrate'] * r_winrate +
                weights['rrpnl'] * rr_pnl
            )

            return safe_value(reward, default=0.0, clip_min=-1e9, clip_max=1e9)

    def _update_portfolio(self):
        """
        Update portfolio by checking if SL or TP hit on existing positions.
        Returns closed positions for reward calculation.
        """
        if not self.open_positions:
            return 0.0, 0, []

        total_pnl = 0.0
        positions_to_close = []
        closed_positions = []  # For reward calculation

        idx = max(0, min(len(self.close_prices) - 1, int(self.current_step - 1)))
        current_price = safe_value(self.close_prices[idx], default=0.0)
        
        contract_size = getattr(config, 'CONTRACT_SIZE', 100000)

        for position in list(self.open_positions):
            sl = safe_value(position.get('sl', np.nan), default=np.nan)
            tp = safe_value(position.get('tp', np.nan), default=np.nan)
            entry_price = safe_value(position.get('price', 0.0), default=0.0)
            lot_size = safe_value(position.get('lot_size', 0.0), default=0.0, clip_min=0.0, clip_max=100.0)

            if lot_size <= 0:
                positions_to_close.append(position)
                continue

            if position.get('type') == 'LONG':
                # Long Exit: Check SL/TP
                if not np.isnan(sl) and self.low_prices[idx] <= sl + _EPS:
                    close_price = sl
                    pnl = (close_price - entry_price) * lot_size * contract_size
                    total_pnl += pnl
                    positions_to_close.append(position)
                    position['realized_pnl'] = pnl
                    position['exit_price'] = close_price
                    position['exit_reason'] = 'SL'
                    closed_positions.append(position.copy())
                    denom = entry_price * lot_size * contract_size
                    pct = safe_div(pnl, denom, default=0.0)
                    self.recent_returns.append(safe_value(pct))
                    
                elif not np.isnan(tp) and self.high_prices[idx] >= tp - _EPS:
                    close_price = tp
                    pnl = (close_price - entry_price) * lot_size * contract_size
                    total_pnl += pnl
                    positions_to_close.append(position)
                    position['realized_pnl'] = pnl
                    position['exit_price'] = close_price
                    position['exit_reason'] = 'TP'
                    closed_positions.append(position.copy())
                    denom = entry_price * lot_size * contract_size
                    pct = safe_div(pnl, denom, default=0.0)
                    self.recent_returns.append(safe_value(pct))
                
            else:  # SHORT
                spread_cost = config.SPREAD_PIPS * config.PIP_VALUE
                ask_high = self.high_prices[idx] + spread_cost
                ask_low = self.low_prices[idx] + spread_cost
                
                if not np.isnan(sl) and ask_high >= sl - _EPS:
                    close_price = sl
                    pnl = (entry_price - close_price) * lot_size * contract_size
                    total_pnl += pnl
                    positions_to_close.append(position)
                    position['realized_pnl'] = pnl
                    position['exit_price'] = close_price
                    position['exit_reason'] = 'SL'
                    closed_positions.append(position.copy())
                    denom = entry_price * lot_size * contract_size
                    pct = safe_div(pnl, denom, default=0.0)
                    self.recent_returns.append(safe_value(pct))
                    
                elif not np.isnan(tp) and ask_low <= tp + _EPS:
                    close_price = tp
                    pnl = (entry_price - close_price) * lot_size * contract_size
                    total_pnl += pnl
                    positions_to_close.append(position)
                    position['realized_pnl'] = pnl
                    position['exit_price'] = close_price
                    position['exit_reason'] = 'TP'
                    closed_positions.append(position.copy())
                    denom = entry_price * lot_size * contract_size
                    pct = safe_div(pnl, denom, default=0.0)
                    self.recent_returns.append(safe_value(pct))

        total_pnl = float(np.clip(total_pnl, -_MAX_PNL, _MAX_PNL))

        if positions_to_close:
            new_equity = self.equity + total_pnl
            self.equity = float(np.clip(new_equity, -_MAX_PNL, _MAX_PNL))
            self.open_positions = [p for p in self.open_positions if p not in positions_to_close]
            
            # Update metrics
            for p in positions_to_close:
                exit_reason = p.get('exit_reason', 'UNKNOWN')
                is_win = 1.0 if exit_reason == 'TP' else 0.0
                self.win_history.append(is_win)
                
                rr_idx = int(p.get('rr_profile_index', 0))
                if 0 <= rr_idx < len(config.RR_PROFILES):
                    sl_mult, tp_mult = config.RR_PROFILES[rr_idx]
                    rr_val = safe_div(tp_mult, sl_mult, default=0.0)
                    self.rr_history.append(rr_val)
                
                risk_idx = int(p.get('risk_level', 0))
                if 0 <= risk_idx < len(config.RISK_LEVELS):
                    risk_val = config.RISK_LEVELS[risk_idx]
                    self.risk_history.append(risk_val)

        return total_pnl, len(positions_to_close), closed_positions

    def _execute_action(self, action_type, risk_index, rr_profile_index):
        """
        Execute trading action with curriculum phase overrides.
        
        Phase 1: Override to use RR_PROFILES[0] and RISK_LEVELS[0]
        Phase 2: Override to use RISK_LEVELS[0], respect agent's RR choice
        Phase 3: Respect all agent choices
        """
        try:
            action_type = int(action_type)
        except Exception:
            action_type = 0
        
        # Original agent choices
        original_risk_index = int(np.clip(int(risk_index), 0, max(0, len(config.RISK_LEVELS) - 1)))
        original_rr_index = int(np.clip(int(rr_profile_index), 0, max(0, len(config.RR_PROFILES) - 1)))
        
        # Apply curriculum phase overrides
        if self.phase == 1:
            # Phase 1: Force simplest configuration
            risk_index = 0  # Force lowest risk
            rr_profile_index = 0  # Force simplest R:R
            _logger.debug(f"Phase 1 Override: RR={rr_profile_index}, Risk={risk_index}")
        
        elif self.phase == 2:
            # Phase 2: Force lowest risk, respect R:R choice
            risk_index = 0  # Force lowest risk
            rr_profile_index = original_rr_index  # Respect agent's R:R choice
            _logger.debug(f"Phase 2 Override: Risk forced to 0, RR={rr_profile_index} (agent choice)")
        
        else:  # Phase 3
            # Phase 3: Full autonomy - respect all choices
            risk_index = original_risk_index
            rr_profile_index = original_rr_index
        
        # Continue with execution using (potentially overridden) values
        if (len(self.open_positions) >= config.MAX_POSITIONS_PER_SYMBOL or action_type == 2):
            if action_type == 2:
                self.consecutive_holds += 1
            return

        self.consecutive_holds = 0
        
        idx_prev = max(0, min(len(self.close_prices) - 1, int(self.current_step - 1)))
        idx_curr = max(0, min(len(self.open_prices) - 1, int(self.current_step)))
        
        current_open_price = safe_value(self.open_prices[idx_curr], default=0.0)
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

        if action_type == 0:  # BUY/LONG
            entry_price = current_open_price + spread_val
            sl = entry_price - (atr * sl_multiplier)
            tp = entry_price + (atr * tp_multiplier)
            pos_type = 'LONG'
            
        else:  # SELL/SHORT (action_type == 1)
            entry_price = current_open_price
            sl = entry_price + (atr * sl_multiplier)
            tp = entry_price - (atr * tp_multiplier)
            pos_type = 'SHORT'

        stop_distance = abs(entry_price - sl)
        stop_distance = max(stop_distance, _EPS)

        if stop_distance > 0 and self.equity > 0:
            # Calculate risk amount for Phase 2 reward normalization
            risk_amount = self.equity * (risk_percent / 100.0)
            
            contract_size = getattr(config, 'CONTRACT_SIZE', 100000)
            raw_lot_size = risk_amount / (stop_distance * contract_size)
            
            lot_step = getattr(config, 'LOT_STEP', 0.01)
            lot_size = math.floor(raw_lot_size / lot_step) * lot_step
            
            min_lot = getattr(config, 'MIN_LOT_SIZE', 0.01)
            max_lot = getattr(config, 'MAX_LOT_SIZE', 100.0)
            lot_size = max(min_lot, min(lot_size, max_lot))
            
            if lot_size > 0 and np.isfinite(lot_size):
                new_position = {
                    'type': pos_type,
                    'price': float(entry_price),
                    'sl': float(sl),
                    'tp': float(tp),
                    'lot_size': float(lot_size),
                    'risk_level': int(risk_index),
                    'rr_profile_index': int(rr_profile_index),
                    'risk_amount': float(risk_amount)  # Store for Phase 2 reward
                }
                self.open_positions.append(new_position)
                self._just_opened_trade = True
                _logger.debug(f"Phase {self.phase} - Opened {pos_type}: {lot_size:.2f} lots, RR={rr_profile_index}, Risk={risk_index}")

    def _next_observation(self):
        """Create observation from COMPLETED candles only."""
        end = int(self.current_step)
        start = max(0, end - self.lookback_window)

        slice_arr = self.features_array[start:end]
        
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
        """
        Execute one step with curriculum phase logic.
        Action is ALWAYS parsed as [direction, rr_profile, risk_level]
        but may be overridden internally based on phase.
        """
        if self.equity <= 0:
            self.current_step += 1
            obs = self._next_observation()
            return obs, float(config.CONSTRAINT_VIOLATION_PENALTY), True, False, {}

        try:
            action_type, rr_index, risk_index = action
        except Exception:
            action_type, rr_index, risk_index = 2, 0, 0  # Default to HOLD

        # Execute action (with potential phase overrides inside)
        self._execute_action(action_type, risk_index, rr_index)

        self.current_step += 1
        terminated = bool(self.current_step >= len(self.df))

        # Update portfolio and get closed positions
        pnl_from_closed_trades, num_closed, closed_positions = self._update_portfolio()
        
        # Calculate reward based on phase
        reward = self._calculate_reward(
            total_pnl_closed_this_step=pnl_from_closed_trades,
            num_closed_trades=num_closed,
            closed_positions=closed_positions,
            action=action
        )

        if not np.isfinite(self.equity):
            self.equity = safe_value(self.equity, default=0.0)
        if self.equity <= 0:
            terminated = True
            reward += float(config.CONSTRAINT_VIOLATION_PENALTY)

        obs = self._next_observation()
        obs['market_features'] = np.nan_to_num(obs['market_features'], nan=0.0, posinf=_MAX_OBS_VALUE, neginf=-_MAX_OBS_VALUE)

        return obs, float(safe_value(reward, default=0.0)), terminated, False, {}