# File: config_optimized.py
# Description: Optimized configuration for better performance
# =============================================================================
# pip install stable-baselines3[extra] gymnasium torch pandas pyarrow MetaTrader5 tqdm matplotlib numba Flask Flask-HTTPAuth opencv-python-headless psutil

#import MetaTrader5 as mt5
import logging
import psutil
import torch
from stable_baselines3.common.utils import ConstantSchedule

# --- Performance Optimization Settings ---
ENABLE_TORCH_COMPILE = True  # Use torch.compile for faster inference (PyTorch 2.0+)
PREFETCH_FACTOR = 2  # Data loading optimization
PIN_MEMORY = True  # Faster GPU transfers
PERSISTENT_WORKERS = True  # Keep data loaders alive

# --- Instruments and Timeframe ---
TICKERS = ['EURUSDc']
START_DATE = "2023-01-01"
END_DATE = "2025-08-01"

# --- Live Trading Configuration ---
MAGIC_NUMBER = 123456
MAX_TOTAL_POSITIONS = 10
MAX_POSITIONS_PER_SYMBOL = MAX_TOTAL_POSITIONS
GUI_STATUS_FILE = "live_status.json"

# --- General Risk Management ---
RISK_LEVELS = [0.5, 1.0, 2.0] 
MIN_LOT_SIZE = 0.01
MAX_DRAWDOWN_PERCENT = 100.0
MAX_VALIDATION_DRAWDOWN_PERCENT = 80.0

# --- Optimized RL Configuration ---
RL_MODEL_DIR = "rl_models"
TRADE_LOG_DIR = "trade_logs"
PROFIT_FACTOR_THRESHOLD = 1.10
REPLAY_BUFFER_FILE = "replay_buffer.joblib"
INDICATOR_LOOKBACK_CANDLES = 300 

# Training settings optimized for speed
INITIAL_TRAINING_TIMESTEPS = 5_000_000  # Reduced for faster initial training
CONTINUOUS_TRAINING_TIMESTEPS = 5_000_000  # Smaller incremental training
RL_LOOKBACK_WINDOW = 48  # Reduced from 60 for faster processing
INITIAL_EQUITY = 200.00

# Dynamic CPU usage based on system resources
def get_optimal_cpu_count():
    """Determines optimal number of CPUs to use"""
    total_cpus = psutil.cpu_count()
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    
    # Conservative approach: don't use all CPUs to leave room for system
    max_envs_by_cpu = max(1, int(total_cpus))
    
    # Memory-based limit (rough estimate: 1.5GB per environment)
    max_envs_by_memory = max(1, int(available_memory_gb // 1.5))
    
    # Take the minimum to avoid resource exhaustion
    optimal_count = min(max_envs_by_cpu, max_envs_by_memory, 12)  # Cap at 12
    optimal_count = 8
    return optimal_count

NUM_CPU_TO_USE = get_optimal_cpu_count()

# Optimized PPO hyperparameters
def get_device_optimized_hyperparams():
    """Returns hyperparameters optimized for the current device"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        # GPU optimizations
        batch_size = 256
        n_steps = 4096
        n_epochs = 10
        net_arch = {"pi": [128, 128, 64], "vf": [128, 128, 64]}
    else:
        # CPU optimizations - smaller networks and batches
        batch_size = 256
        n_steps = 4096
        n_epochs = 8
        net_arch = {"pi": [128, 128, 64], "vf": [128, 128, 64]}

    """# CPU optimizations - smaller networks and batches
        batch_size = 128
        n_steps = 2048
        n_epochs = 8
        net_arch = {"pi": [64, 64], "vf": [64, 64]}"""
    
    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": ConstantSchedule(0.1),
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "learning_rate": 0.0001, #initial training rate, 9e-6 
        "policy_kwargs": {
            "net_arch": net_arch,
            "activation_fn": torch.nn.ReLU,
            "ortho_init": True,
        }
    }

PPO_HYPERPARAMS = get_device_optimized_hyperparams()

# --- Timeframe Configuration ---
PRIMARY_TIMEFRAME_STRING = "M5"
TREND_TIMEFRAME_STRING = "M30"
PRIMARY_TIMEFRAME_MT5 = "M5"
TREND_TIMEFRAME_MT5 = "M30"

# Calculate candles per day
minutes_per_candle = 0
if PRIMARY_TIMEFRAME_STRING.startswith("M"):
    minutes_per_candle = int(PRIMARY_TIMEFRAME_STRING.replace("M", ""))
elif PRIMARY_TIMEFRAME_STRING.startswith("H"):
    minutes_per_candle = int(PRIMARY_TIMEFRAME_STRING.replace("H", "")) * 60
if minutes_per_candle > 0:
    CANDLES_PER_DAY = (24 * 60) // minutes_per_candle
else:
    CANDLES_PER_DAY = 288

# --- Simplified Reward Function Configuration ---
# Reduced complexity for faster computation
DRAWDOWN_PENALTY_SCALAR = 0.1  # Increased impact
ACTION_NOVELTY_SCALAR = 0.001
TRADE_QUALITY_REWARD_SCALER = 0.0  # Disabled for speed
CVAR_WINDOW = 24  # Reduced window size
CVAR_ALPHA = 0.05
CVAR_PENALTY_SCALAR = 0.0  # Disabled for speed

# Simplified frequency targeting
TARGET_TRADE_FREQUENCY_UPPER_RANGE = 0.10
TARGET_TRADE_FREQUENCY_LOWER_RANGE = 0.05
FREQUENCY_PENALTY_SCALAR = 0.0  # Disabled for speed
ACTIVITY_MEMORY_WINDOW = 864  # Reduced from 864

# --- Technical Analysis Configuration ---
M5_ATR_PERIOD = 14
ADX_PERIOD = 14
RSI_PERIOD = 14
MA_FAST_PERIOD = 12
MA_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9
H1_EMA_PERIOD = 50
NORMALIZATION_WINDOW = 50  # Reduced from 100 for speed
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3

RR_PROFILES = [
    (1.0, 2.0), (1.0, 3.0), (1.0, 4.0),
    (1.5, 3.0), (1.5, 4.5), (1.5, 6.0),
    (2.0, 4.0), (2.0, 6.0), (2.0, 8.0),
    (2.5, 5.0)
]


# Hard constraints - simplified
MAX_DRAWDOWN = 0.50
MIN_TRADE_FREQUENCY = 0.05
MAX_TRADE_FREQUENCY = 0.20
CONSTRAINT_VIOLATION_PENALTY = -10.0  # Reduced penalty magnitude

# Power Save Mode
POWER_SAVE_MODE = True
QUIET_HOURS = [(22, 23), (12, 13)]

# --- Memory Management Settings ---
TORCH_MEMORY_CLEANUP_INTERVAL = 1000  # Steps between memory cleanup
MAX_MEMORY_USAGE_PCT = 85  # Maximum memory usage before cleanup

# --- Data Processing Optimizations ---
USE_NUMBA_ACCELERATION = True
VECTORIZED_OPERATIONS = True
CACHE_INDICATORS = True
PRECOMPUTE_FEATURES = True

# Dashboard Configuration
DASHBOARD_USERNAME = "your_username"
DASHBOARD_PASSWORD = "your_strong_password"
DASHBOARD_STALE_THRESHOLD_MINUTES = 120
