"""
Manual Phase Control Trainer (LSTM Version)
Gives you full control over each phase of curriculum learning
Supports resuming from current phase checkpoints
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

# CHANGED: Use RecurrentPPO
from sb3_contrib import RecurrentPPO
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
        self.training_start_time = time.time()
        
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


def train_phase_1(ticker, df, num_envs, timesteps, resume_path=None):
    """
    PHASE 1: Direction Learning
    """
    print("\n" + "="*80)
    print("🎯 PHASE 1: DIRECTION LEARNING")
    print("="*80)
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Create Phase 1 environment
    env = create_phase_env(df, phase=1, num_envs=num_envs)
    
    # Logic to resume or start new
    if resume_path and os.path.exists(resume_path):
        logger.info(f"🔄 Resuming Phase 1 from: {resume_path}")
        model = RecurrentPPO.load(
            str(resume_path),
            env=env,
            device=device
        )
        reset_timesteps = False
    else:
        logger.info("🆕 Creating new RecurrentPPO model for Phase 1")
        model = RecurrentPPO(
            "MultiInputLstmPolicy",
            env,
            verbose=1,
            device=device,
            tensorboard_log="./ppo_phase1_tensorboard/",
            **config.PPO_HYPERPARAMS
        )
        reset_timesteps = True
    
    # Create callback
    save_freq = max(2048 * 80 // num_envs, 1000)
    callback = PhaseCallback(
        phase=1,
        save_freq=save_freq,
        save_dir=config.RL_MODEL_DIR,
        ticker=ticker
    )
    
    # Train
    print(f"\n🚀 Starting Phase 1 training...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=timesteps,
        progress_bar=True,
        callback=callback,
        reset_num_timesteps=reset_timesteps
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


def train_phase_2(ticker, df, num_envs, timesteps, phase1_model_path, resume_path=None):
    """
    PHASE 2: R:R Strategy Learning
    """
    print("\n" + "="*80)
    print("📊 PHASE 2: R:R STRATEGY LEARNING")
    print("="*80)
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create Phase 2 environment
    env = create_phase_env(df, phase=2, num_envs=num_envs)
    
    # Logic to resume from Phase 2 OR load Phase 1
    if resume_path and os.path.exists(resume_path):
        logger.info(f"🔄 Resuming Phase 2 from existing Phase 2 model: {resume_path}")
        model = RecurrentPPO.load(
            str(resume_path),
            env=env,
            device=device
        )
    else:
        if not phase1_model_path.exists():
            raise FileNotFoundError(f"Phase 1 model not found at {phase1_model_path}")
            
        logger.info(f"🆕 Starting Phase 2 using Phase 1 weights: {phase1_model_path}")
        model = RecurrentPPO.load(
            str(phase1_model_path),
            env=env,
            device=device
        )
    
    # Create callback
    save_freq = max(2048 * 80 // num_envs, 1000)
    callback = PhaseCallback(
        phase=2,
        save_freq=save_freq,
        save_dir=config.RL_MODEL_DIR,
        ticker=ticker
    )
    
    # Train
    print(f"\n🚀 Starting Phase 2 training...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=timesteps,
        progress_bar=True,
        callback=callback,
        reset_num_timesteps=False
    )
    
    # Save final model
    final_path = Path(config.RL_MODEL_DIR) / f"phase2_{ticker}_final.zip"
    model.save(str(final_path))
    
    training_time = time.time() - start_time
    print(f"\n✅ Phase 2 completed in {training_time/3600:.2f} hours")
    print(f"📦 Model saved: {final_path}")
    
    env.close()
    return final_path


def train_phase_3(ticker, df, num_envs, timesteps, phase2_model_path, resume_path=None):
    """
    PHASE 3: Full Risk Management
    """
    print("\n" + "="*80)
    print("🎓 PHASE 3: FULL RISK MANAGEMENT")
    print("="*80)
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create Phase 3 environment
    env = create_phase_env(df, phase=3, num_envs=num_envs)
    
    # Logic to resume from Phase 3 OR load Phase 2
    if resume_path and os.path.exists(resume_path):
        logger.info(f"🔄 Resuming Phase 3 from existing Phase 3 model: {resume_path}")
        model = RecurrentPPO.load(
            str(resume_path),
            env=env,
            device=device
        )
    else:
        if not phase2_model_path.exists():
            raise FileNotFoundError(f"Phase 2 model not found at {phase2_model_path}")

        logger.info(f"🆕 Starting Phase 3 using Phase 2 weights: {phase2_model_path}")
        model = RecurrentPPO.load(
            str(phase2_model_path),
            env=env,
            device=device
        )
    
    # Create callback
    save_freq = max(2048 * 80 // num_envs, 1000)
    callback = PhaseCallback(
        phase=3,
        save_freq=save_freq,
        save_dir=config.RL_MODEL_DIR,
        ticker=ticker
    )
    
    # Train
    print(f"\n🚀 Starting Phase 3 training...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=timesteps,
        progress_bar=True,
        callback=callback,
        reset_num_timesteps=False
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
    Automatically detects if a previous model exists for the current phase 
    and resumes from it if available.
    """
    print("="*80)
    print("🎮 MANUAL PHASE CONTROL TRAINER (LSTM & RESUME SUPPORT)")
    print("="*80)
    
    ticker = config.TICKERS[0]
    num_envs = config.NUM_CPU_TO_USE
    
    # Get phase selection from config
    phase_to_train = getattr(config, 'CURRICULUM_PHASE', 'all')
    
    # Load data once
    df = load_data(ticker)
    
    # Get timesteps from config (with defaults)
    phase_timesteps = {
        1: getattr(config, 'PHASE1_TIMESTEPS', 2_000_000),
        2: getattr(config, 'PHASE2_TIMESTEPS', 3_000_000),
        3: getattr(config, 'PHASE3_TIMESTEPS', 5_000_000)
    }
    
    # Model paths
    phase1_final = Path(config.RL_MODEL_DIR) / f"phase1_{ticker}_final.zip"
    phase2_final = Path(config.RL_MODEL_DIR) / f"phase2_{ticker}_final.zip"
    phase3_final = Path(config.RL_MODEL_DIR) / f"phase3_{ticker}_final.zip"
    
    # ----- TRAIN BASED ON CONFIG -----
    
    if phase_to_train == 'all':
        print("🎯 Mode: Training ALL phases (Sequential)")
        
        # Train 1
        phase1_final = train_phase_1(ticker, df, num_envs, phase_timesteps[1], resume_path=phase1_final)
        # Train 2
        phase2_final = train_phase_2(ticker, df, num_envs, phase_timesteps[2], phase1_final, resume_path=phase2_final)
        # Train 3
        phase3_final = train_phase_3(ticker, df, num_envs, phase_timesteps[3], phase2_final, resume_path=phase3_final)
    
    elif phase_to_train == 1:
        print("🎯 Mode: Training PHASE 1 ONLY")
        train_phase_1(
            ticker=ticker,
            df=df,
            num_envs=num_envs,
            timesteps=phase_timesteps[1],
            resume_path=phase1_final  # Will resume if this file exists
        )
    
    elif phase_to_train == 2:
        print("🎯 Mode: Training PHASE 2 ONLY")
        
        train_phase_2(
            ticker=ticker,
            df=df,
            num_envs=num_envs,
            timesteps=phase_timesteps[2],
            phase1_model_path=phase1_final,
            resume_path=phase2_final # Will resume if this file exists
        )
    
    elif phase_to_train == 3:
        print("🎯 Mode: Training PHASE 3 ONLY")
        
        train_phase_3(
            ticker=ticker,
            df=df,
            num_envs=num_envs,
            timesteps=phase_timesteps[3],
            phase2_model_path=phase2_final,
            resume_path=phase3_final # Will resume if this file exists
        )
    
    print("\n🎯 Next Steps:")
    print("1. Run backtester.py to evaluate model(s)")
    print("2. Compare performance")
    print("3. Use best model for live trading")
    print("="*80)


if __name__ == '__main__':
    main()