"""
Manual Phase Control Trainer
Gives you full control over each phase of curriculum learning
"""
import logging
import os
import pandas as pd
import numpy as np
import time
import psutil
import torch
import gc
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

import config
from rl_environment import TradingEnv

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PhaseCallback(BaseCallback):
    """Enhanced callback with memory management and detailed logging"""
    
    def __init__(self, phase, save_freq, save_dir, ticker):
        super().__init__()
        self.phase = phase
        self.save_freq = save_freq
        self.save_dir = save_dir
        self.ticker = ticker
        self.last_save = 0
        self.last_memory_check = 0
        
    def _on_step(self) -> bool:
        # Memory cleanup every 10k steps
        if self.n_calls - self.last_memory_check > 10000:
            memory_pct = psutil.virtual_memory().percent
            if memory_pct > config.MAX_MEMORY_USAGE_PCT:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info(f"Memory cleanup at {memory_pct:.1f}% usage")
            self.last_memory_check = self.n_calls
        
        # Save checkpoints
        if (self.n_calls % self.save_freq == 0 and 
            self.n_calls > self.last_save + 5000):
            
            model_path = Path(self.save_dir) / f"phase{self.phase}_{self.ticker}_{self.n_calls}.zip"
            
            # Background save
            import threading
            def save_model():
                try:
                    self.model.save(str(model_path))
                    print(f"✅ Phase {self.phase} checkpoint: {model_path.name}")
                except Exception as e:
                    logger.error(f"Save failed: {e}")
            
            threading.Thread(target=save_model, daemon=True).start()
            self.last_save = self.n_calls
        
        # Progress reporting
        if self.n_calls % 10000 == 0:
            elapsed = time.time() - self.training_start_time
            speed = self.n_calls / elapsed if elapsed > 0 else 0
            memory_pct = psutil.virtual_memory().percent
            
            print(f"Phase {self.phase} | Step {self.n_calls:,} | Speed: {speed:.1f} steps/s | Memory: {memory_pct:.1f}%")
        
        return True
    
    def _on_training_start(self):
        self.training_start_time = time.time()


def load_data(ticker):
    """Load preprocessed data"""
    data_path = Path("processed_data") / f"{ticker}_processed.parquet"
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found at {data_path}. "
            "Run preprocess_data.py first."
        )
    
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    
    # Basic validation and cleaning
    df = df.dropna().sort_index()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].astype(np.float32)
    
    logger.info(f"Data loaded: {len(df):,} samples, {df.shape[1]} features")
    return df


def create_phase_env(df, phase, num_envs):
    """Create vectorized environment for specific phase"""
    vec_env_cls = DummyVecEnv if num_envs <= 4 else SubprocVecEnv
    
    env = make_vec_env(
        TradingEnv,
        n_envs=num_envs,
        seed=42,
        vec_env_cls=vec_env_cls,
        env_kwargs={'df': df, 'phase': phase}
    )
    
    logger.info(f"Created Phase {phase} environment with {num_envs} workers ({vec_env_cls.__name__})")
    return env


def train_phase_1(ticker, df, num_envs, timesteps):
    """
    PHASE 1: Direction Learning
    
    Goal: Teach the agent to get market direction right
    - Forces simplest R:R profile (1:2)
    - Forces lowest risk (0.5%)
    - Binary reward: +1 for wins, -1 for losses
    """
    print("\n" + "="*80)
    print("🎯 PHASE 1: DIRECTION LEARNING")
    print("="*80)
    print("Goal: Learn to predict market direction")
    print("Overrides: RR=1:2 (forced), Risk=0.5% (forced)")
    print("Reward: Binary (+1 win, -1 loss)")
    print("="*80 + "\n")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create Phase 1 environment
    env = create_phase_env(df, phase=1, num_envs=num_envs)
    
    # Create new model
    logger.info("Creating new PPO model for Phase 1")
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        device=device,
        tensorboard_log="./ppo_phase1_tensorboard/",
        **config.PPO_HYPERPARAMS
    )
    
    # Create callback
    save_freq = max(2048 * 80 // num_envs, 1000)
    callback = PhaseCallback(
        phase=1,
        save_freq=save_freq,
        save_dir=config.RL_MODEL_DIR,
        ticker=ticker
    )
    
    # Training info
    logger.info(f"Training timesteps: {timesteps:,}")
    logger.info(f"Batch size: {config.PPO_HYPERPARAMS.get('batch_size')}")
    logger.info(f"Learning rate: {config.PPO_HYPERPARAMS.get('learning_rate')}")
    
    # Train
    print(f"\n🚀 Starting Phase 1 training...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=timesteps,
        progress_bar=True,
        callback=callback
    )
    
    # Save final model
    os.makedirs(config.RL_MODEL_DIR, exist_ok=True)
    final_path = Path(config.RL_MODEL_DIR) / f"phase1_{ticker}_final.zip"
    model.save(str(final_path))
    
    training_time = time.time() - start_time
    print(f"\n✅ Phase 1 completed in {training_time/3600:.2f} hours")
    print(f"📦 Model saved: {final_path}")
    
    env.close()
    return final_path


def train_phase_2(ticker, df, num_envs, timesteps, phase1_model_path):
    """
    PHASE 2: R:R Strategy Learning
    
    Goal: Teach the agent to select optimal R:R profiles
    - Respects agent's R:R choice (learns to pick from 10 profiles)
    - Forces lowest risk (0.5%)
    - Reward: PnL / risk_amount (teaches R-multiples)
    """
    print("\n" + "="*80)
    print("📊 PHASE 2: R:R STRATEGY LEARNING")
    print("="*80)
    print("Goal: Learn to select optimal R:R profiles")
    print("Overrides: Risk=0.5% (forced), RR=Agent Choice (10 options)")
    print("Reward: Normalized by risk (R-multiples)")
    print("="*80 + "\n")
    
    # Verify Phase 1 model exists
    if not phase1_model_path.exists():
        raise FileNotFoundError(
            f"Phase 1 model not found at {phase1_model_path}. "
            "Please train Phase 1 first."
        )
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Create Phase 2 environment
    env = create_phase_env(df, phase=2, num_envs=num_envs)
    
    # Load Phase 1 weights
    logger.info(f"Loading Phase 1 weights from: {phase1_model_path}")
    model = PPO.load(
        str(phase1_model_path),
        env=env,
        device=device
    )
    
    logger.info("✅ Phase 1 weights loaded successfully")
    logger.info("🔄 Continuing training with Phase 2 reward structure")
    
    # Create callback
    save_freq = max(2048 * 80 // num_envs, 1000)
    callback = PhaseCallback(
        phase=2,
        save_freq=save_freq,
        save_dir=config.RL_MODEL_DIR,
        ticker=ticker
    )
    
    # Training info
    logger.info(f"Training timesteps: {timesteps:,}")
    
    # Train
    print(f"\n🚀 Starting Phase 2 training...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=timesteps,
        progress_bar=True,
        callback=callback,
        reset_num_timesteps=False  # Continue counting
    )
    
    # Save final model
    final_path = Path(config.RL_MODEL_DIR) / f"phase2_{ticker}_final.zip"
    model.save(str(final_path))
    
    training_time = time.time() - start_time
    print(f"\n✅ Phase 2 completed in {training_time/3600:.2f} hours")
    print(f"📦 Model saved: {final_path}")
    
    env.close()
    return final_path


def train_phase_3(ticker, df, num_envs, timesteps, phase2_model_path):
    """
    PHASE 3: Full Risk Management
    
    Goal: Full autonomy - manage everything
    - Respects agent's R:R choice (10 profiles)
    - Respects agent's risk choice (3 levels: 0.5%, 1%, 2%)
    - Reward: Full multi-objective (PnL + drawdown penalties)
    """
    print("\n" + "="*80)
    print("🎓 PHASE 3: FULL RISK MANAGEMENT")
    print("="*80)
    print("Goal: Complete autonomy in trading decisions")
    print("Overrides: NONE - Full agent control")
    print("Choices: Direction (3), R:R (10), Risk (3) = 90 combinations")
    print("Reward: Full multi-objective with drawdown penalties")
    print("="*80 + "\n")
    
    # Verify Phase 2 model exists
    if not phase2_model_path.exists():
        raise FileNotFoundError(
            f"Phase 2 model not found at {phase2_model_path}. "
            "Please train Phase 2 first."
        )
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Create Phase 3 environment
    env = create_phase_env(df, phase=3, num_envs=num_envs)
    
    # Load Phase 2 weights
    logger.info(f"Loading Phase 2 weights from: {phase2_model_path}")
    model = PPO.load(
        str(phase2_model_path),
        env=env,
        device=device
    )
    
    logger.info("✅ Phase 2 weights loaded successfully")
    logger.info("🔄 Continuing training with Phase 3 full autonomy")
    
    # Create callback
    save_freq = max(2048 * 80 // num_envs, 1000)
    callback = PhaseCallback(
        phase=3,
        save_freq=save_freq,
        save_dir=config.RL_MODEL_DIR,
        ticker=ticker
    )
    
    # Training info
    logger.info(f"Training timesteps: {timesteps:,}")
    logger.info("Reward weights:")
    for key, value in config.REWARD_WEIGHTS.items():
        logger.info(f"  {key}: {value}")
    
    # Train
    print(f"\n🚀 Starting Phase 3 training...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=timesteps,
        progress_bar=True,
        callback=callback,
        reset_num_timesteps=False  # Continue counting
    )
    
    # Save final model
    final_path = Path(config.RL_MODEL_DIR) / f"phase3_{ticker}_final.zip"
    model.save(str(final_path))
    
    training_time = time.time() - start_time
    print(f"\n✅ Phase 3 completed in {training_time/3600:.2f} hours")
    print(f"📦 Model saved: {final_path}")
    
    env.close()
    return final_path


def main():
    """
    Main function for manual phase control training
    
    Controlled by config.CURRICULUM_PHASE:
    - 'all': Train all phases sequentially (1 → 2 → 3)
    - 1: Train only Phase 1
    - 2: Train only Phase 2 (requires Phase 1)
    - 3: Train only Phase 3 (requires Phase 2)
    """
    print("="*80)
    print("🎮 MANUAL PHASE CONTROL TRAINER")
    print("="*80)
    
    ticker = config.TICKERS[0]
    num_envs = config.NUM_CPU_TO_USE
    
    # Get phase selection from config
    phase_to_train = getattr(config, 'CURRICULUM_PHASE', 'all')
    
    # Validate phase selection
    valid_phases = ['all', 1, 2, 3]
    if phase_to_train not in valid_phases:
        raise ValueError(
            f"Invalid CURRICULUM_PHASE in config: {phase_to_train}. "
            f"Must be one of: {valid_phases}"
        )
    
    print(f"📋 Training Mode: {phase_to_train}")
    print("="*80 + "\n")
    
    # Load data once
    df = load_data(ticker)
    
    # Get timesteps from config (with defaults)
    phase_timesteps = {
        1: getattr(config, 'PHASE1_TIMESTEPS', 2_000_000),
        2: getattr(config, 'PHASE2_TIMESTEPS', 3_000_000),
        3: getattr(config, 'PHASE3_TIMESTEPS', 5_000_000)
    }
    
    # Model paths
    phase1_path = Path(config.RL_MODEL_DIR) / f"phase1_{ticker}_final.zip"
    phase2_path = Path(config.RL_MODEL_DIR) / f"phase2_{ticker}_final.zip"
    phase3_path = Path(config.RL_MODEL_DIR) / f"phase3_{ticker}_final.zip"
    
    # ----- TRAIN BASED ON CONFIG -----
    
    if phase_to_train == 'all':
        # Train all phases sequentially
        print("🎯 Mode: Training ALL phases (1 → 2 → 3)")
        print("="*80 + "\n")
        
        # Phase 1
        if phase1_path.exists():
            print(f"\n⏭️  Phase 1 model already exists: {phase1_path}")
            print("Skipping Phase 1 training...")
        else:
            phase1_path = train_phase_1(
                ticker=ticker,
                df=df,
                num_envs=num_envs,
                timesteps=phase_timesteps[1]
            )
        
        # Phase 2
        if phase2_path.exists():
            print(f"\n⏭️  Phase 2 model already exists: {phase2_path}")
            print("Skipping Phase 2 training...")
        else:
            phase2_path = train_phase_2(
                ticker=ticker,
                df=df,
                num_envs=num_envs,
                timesteps=phase_timesteps[2],
                phase1_model_path=phase1_path
            )
        
        # Phase 3
        if phase3_path.exists():
            print(f"\n⏭️  Phase 3 model already exists: {phase3_path}")
            print("Skipping Phase 3 training...")
        else:
            phase3_path = train_phase_3(
                ticker=ticker,
                df=df,
                num_envs=num_envs,
                timesteps=phase_timesteps[3],
                phase2_model_path=phase2_path
            )
        
        # Summary
        print("\n" + "="*80)
        print("✅ ALL PHASES TRAINING COMPLETE!")
        print("="*80)
        print(f"📦 Phase 1 Model: {phase1_path}")
        print(f"📦 Phase 2 Model: {phase2_path}")
        print(f"📦 Phase 3 Model: {phase3_path}")
    
    elif phase_to_train == 1:
        # Train only Phase 1
        print("🎯 Mode: Training PHASE 1 ONLY")
        print("="*80 + "\n")
        
        phase1_path = train_phase_1(
            ticker=ticker,
            df=df,
            num_envs=num_envs,
            timesteps=phase_timesteps[1]
        )
        
        print("\n" + "="*80)
        print("✅ PHASE 1 TRAINING COMPLETE!")
        print("="*80)
        print(f"📦 Phase 1 Model: {phase1_path}")
    
    elif phase_to_train == 2:
        # Train only Phase 2
        print("🎯 Mode: Training PHASE 2 ONLY")
        print("="*80 + "\n")
        
        if not phase1_path.exists():
            raise FileNotFoundError(
                f"❌ Phase 1 model not found at {phase1_path}\n"
                "Phase 2 requires Phase 1 weights. Please:\n"
                "1. Set CURRICULUM_PHASE = 1 in config\n"
                "2. Train Phase 1 first\n"
                "3. Then train Phase 2"
            )
        
        phase2_path = train_phase_2(
            ticker=ticker,
            df=df,
            num_envs=num_envs,
            timesteps=phase_timesteps[2],
            phase1_model_path=phase1_path
        )
        
        print("\n" + "="*80)
        print("✅ PHASE 2 TRAINING COMPLETE!")
        print("="*80)
        print(f"📦 Phase 2 Model: {phase2_path}")
    
    elif phase_to_train == 3:
        # Train only Phase 3
        print("🎯 Mode: Training PHASE 3 ONLY")
        print("="*80 + "\n")
        
        if not phase2_path.exists():
            raise FileNotFoundError(
                f"❌ Phase 2 model not found at {phase2_path}\n"
                "Phase 3 requires Phase 2 weights. Please:\n"
                "1. Set CURRICULUM_PHASE = 2 in config\n"
                "2. Train Phase 2 first (which requires Phase 1)\n"
                "3. Then train Phase 3"
            )
        
        phase3_path = train_phase_3(
            ticker=ticker,
            df=df,
            num_envs=num_envs,
            timesteps=phase_timesteps[3],
            phase2_model_path=phase2_path
        )
        
        print("\n" + "="*80)
        print("✅ PHASE 3 TRAINING COMPLETE!")
        print("="*80)
        print(f"📦 Phase 3 Model: {phase3_path}")
    
    # Final next steps
    print("\n🎯 Next Steps:")
    print("1. Run backtester.py to evaluate model(s)")
    print("2. Compare performance")
    print("3. Use best model for live trading")
    print("="*80)


# ============================================================================
# EXAMPLE USAGE SCENARIOS
# ============================================================================

def example_train_only_phase_1():
    """Example: Train only Phase 1"""
    ticker = config.TICKERS[0]
    df = load_data(ticker)
    
    train_phase_1(
        ticker=ticker,
        df=df,
        num_envs=config.NUM_CPU_TO_USE,
        timesteps=2_000_000
    )


def example_train_phase_2_only():
    """Example: Train only Phase 2 (requires existing Phase 1 model)"""
    ticker = config.TICKERS[0]
    df = load_data(ticker)
    
    phase1_model = Path(config.RL_MODEL_DIR) / f"phase1_{ticker}_final.zip"
    
    if not phase1_model.exists():
        print("❌ Phase 1 model not found. Train Phase 1 first!")
        return
    
    train_phase_2(
        ticker=ticker,
        df=df,
        num_envs=config.NUM_CPU_TO_USE,
        timesteps=3_000_000,
        phase1_model_path=phase1_model
    )


def example_custom_timesteps():
    """Example: Use custom timesteps for each phase"""
    ticker = config.TICKERS[0]
    df = load_data(ticker)
    num_envs = config.NUM_CPU_TO_USE
    
    # Custom timesteps
    phase1_path = train_phase_1(ticker, df, num_envs, timesteps=1_000_000)
    phase2_path = train_phase_2(ticker, df, num_envs, timesteps=2_000_000, phase1_model_path=phase1_path)
    phase3_path = train_phase_3(ticker, df, num_envs, timesteps=4_000_000, phase2_model_path=phase2_path)


if __name__ == '__main__':
    # Run main curriculum training
    main()
    
    # Or uncomment to run specific examples:
    # example_train_only_phase_1()
    # example_train_phase_2_only()
    # example_custom_timesteps()