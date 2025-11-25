"""
Curriculum Learning Trainer for Trading Agent
Progressively trains through 3 phases while maintaining same architecture
"""
import logging
import os
import pandas as pd
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

import config
from rl_environment import TradingEnv  # Your refactored environment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PhaseTransitionCallback(BaseCallback):
    """Callback to monitor training and potentially trigger phase transitions"""
    
    def __init__(self, phase, save_freq, save_dir, ticker):
        super().__init__()
        self.phase = phase
        self.save_freq = save_freq
        self.save_dir = save_dir
        self.ticker = ticker
        self.last_save = 0
        
    def _on_step(self) -> bool:
        # Save checkpoints
        if (self.n_calls % self.save_freq == 0 and 
            self.n_calls > self.last_save + 5000):
            
            model_path = Path(self.save_dir) / f"phase{self.phase}_{self.ticker}_{self.n_calls}.zip"
            self.model.save(str(model_path))
            logger.info(f"Phase {self.phase} checkpoint saved: {model_path.name}")
            self.last_save = self.n_calls
        
        # Progress reporting
        if self.n_calls % 10000 == 0:
            elapsed = time.time() - self.training_start_time
            speed = self.n_calls / elapsed if elapsed > 0 else 0
            logger.info(f"Phase {self.phase} | Step {self.n_calls:,} | Speed: {speed:.1f} steps/s")
        
        return True
    
    def _on_training_start(self):
        self.training_start_time = time.time()


class CurriculumTrainer:
    """Trains agent through curriculum phases"""
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.model_dir = Path(config.RL_MODEL_DIR)
        self.model_dir.mkdir(exist_ok=True)
        
    def load_data(self):
        """Load preprocessed data"""
        data_path = Path("processed_data") / f"{self.ticker}_processed.parquet"
        if not data_path.exists():
            raise FileNotFoundError(f"Data not found: {data_path}")
        
        df = pd.read_parquet(data_path)
        df = df.dropna().sort_index()
        logger.info(f"Loaded data: {len(df):,} samples")
        return df
    
    def create_phase_env(self, df, phase, num_envs):
        """Create vectorized environment for specific phase"""
        vec_env_cls = DummyVecEnv if num_envs <= 4 else SubprocVecEnv
        
        env = make_vec_env(
            TradingEnv,
            n_envs=num_envs,
            seed=42,
            vec_env_cls=vec_env_cls,
            env_kwargs={'df': df, 'phase': phase}
        )
        
        logger.info(f"Created Phase {phase} environment with {num_envs} workers")
        return env
    
    def train_phase(self, phase, df, num_envs, timesteps, prev_model_path=None):
        """
        Train a single curriculum phase.
        
        Args:
            phase: Phase number (1, 2, or 3)
            df: Training data
            num_envs: Number of parallel environments
            timesteps: Training timesteps for this phase
            prev_model_path: Path to model from previous phase (for transfer learning)
        """
        logger.info("="*80)
        logger.info(f"STARTING PHASE {phase} TRAINING")
        logger.info("="*80)
        
        # Create phase-specific environment
        env = self.create_phase_env(df, phase, num_envs)
        
        # Load or create model
        if prev_model_path and prev_model_path.exists():
            logger.info(f"Loading weights from previous phase: {prev_model_path}")
            model = PPO.load(
                str(prev_model_path),
                env=env,
                device="cuda" if config.PPO_HYPERPARAMS else "cpu"
            )
            # Keep the same architecture, just continue training
        else:
            logger.info("Creating new model from scratch")
            model = PPO(
                "MultiInputPolicy",
                env,
                verbose=1,
                device="cuda" if config.PPO_HYPERPARAMS else "cpu",
                tensorboard_log=f"./ppo_curriculum_phase{phase}/",
                **config.PPO_HYPERPARAMS
            )
        
        # Create callback
        save_freq = max(2048 * 80 // num_envs, 1000)
        callback = PhaseTransitionCallback(
            phase=phase,
            save_freq=save_freq,
            save_dir=str(self.model_dir),
            ticker=self.ticker
        )
        
        # Display phase-specific info
        phase_descriptions = {
            1: "Direction Learning - Binary win/loss reward",
            2: "R:R Strategy Learning - Risk-normalized rewards",
            3: "Full Risk Management - Complete autonomy"
        }
        
        logger.info(f"Phase {phase} Focus: {phase_descriptions[phase]}")
        logger.info(f"Training timesteps: {timesteps:,}")
        logger.info(f"Environments: {num_envs}")
        
        # Train
        start_time = time.time()
        model.learn(
            total_timesteps=timesteps,
            progress_bar=True,
            callback=callback,
            reset_num_timesteps=False  # Continue counting timesteps
        )
        
        # Save final model for this phase
        final_path = self.model_dir / f"phase{phase}_{self.ticker}_final.zip"
        model.save(str(final_path))
        
        training_time = time.time() - start_time
        logger.info(f"Phase {phase} completed in {training_time/3600:.2f} hours")
        logger.info(f"Phase {phase} model saved: {final_path}")
        
        env.close()
        return final_path
    
    def train_curriculum(self):
        """Execute full 3-phase curriculum training"""
        print("="*80)
        print("CURRICULUM LEARNING TRAINER")
        print("="*80)
        
        # Load data once
        df = self.load_data()
        num_envs = config.NUM_CPU_TO_USE
        
        # Phase-specific timesteps (adjust as needed)
        phase_timesteps = {
            1: 2_000_000,  # Phase 1: Direction learning
            2: 3_000_000,  # Phase 2: R:R strategy learning  
            3: 5_000_000   # Phase 3: Full risk management
        }
        
        # Train Phase 1
        phase1_model = self.train_phase(
            phase=1,
            df=df,
            num_envs=num_envs,
            timesteps=phase_timesteps[1],
            prev_model_path=None
        )
        
        # Train Phase 2 (load Phase 1 weights)
        phase2_model = self.train_phase(
            phase=2,
            df=df,
            num_envs=num_envs,
            timesteps=phase_timesteps[2],
            prev_model_path=phase1_model
        )
        
        # Train Phase 3 (load Phase 2 weights)
        phase3_model = self.train_phase(
            phase=3,
            df=df,
            num_envs=num_envs,
            timesteps=phase_timesteps[3],
            prev_model_path=phase2_model
        )
        
        print("\n" + "="*80)
        print("✅ CURRICULUM TRAINING COMPLETE!")
        print("="*80)
        print(f"Phase 1 Model: {phase1_model}")
        print(f"Phase 2 Model: {phase2_model}")
        print(f"Phase 3 Model: {phase3_model}")
        print("\nNext steps:")
        print("1. Backtest all three phase models")
        print("2. Compare performance across phases")
        print("3. Use Phase 3 model for live trading")
        print("="*80)
        
        return phase1_model, phase2_model, phase3_model


def main():
    """Main entry point"""
    ticker = config.TICKERS[0]
    trainer = CurriculumTrainer(ticker)
    trainer.train_curriculum()


if __name__ == '__main__':
    main()