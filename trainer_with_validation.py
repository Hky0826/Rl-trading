# File: trainer.py
# Description: (CORRECTED) Uses the 'save all candidates' callback and enables GPU.
# =============================================================================
import logging
import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.logger import configure as configure_logger

import config
import data_handler
from rl_environment import TradingEnv
from rl_callbacks import ValidationCallback

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    ticker = config.TICKERS[0]
    logging.info(f"--- Starting Full Training Pipeline for {ticker} ---")

    processed_data_path = os.path.join("processed_data", f"{ticker}_processed.parquet")
    if not os.path.exists(processed_data_path):
        logging.critical(f"Processed data not found at {processed_data_path}. Please run preprocess_data.py first.")
        return
        
    logging.info(f"Loading pre-processed data from {processed_data_path}...")
    full_df = pd.read_parquet(processed_data_path)

    n_validation_period = config.CANDLES_PER_DAY * 10
    validation_df_1 = full_df.iloc[-n_validation_period:]
    training_df = full_df.iloc[:-n_validation_period]
    val_start_2 = int(len(training_df) * 0.25); validation_df_2 = training_df.iloc[val_start_2 : val_start_2 + n_validation_period]
    val_start_3 = int(len(training_df) * 0.50); validation_df_3 = training_df.iloc[val_start_3 : val_start_3 + n_validation_period]
    val_start_4 = int(len(training_df) * 0.75); validation_df_4 = training_df.iloc[val_start_4 : val_start_4 + n_validation_period]
    validation_sets = [validation_df_1, validation_df_2, validation_df_3, validation_df_4]
    logging.info(f"Using {len(validation_sets)} validation sets.")

    final_model_path = os.path.join(config.RL_MODEL_DIR, f"rl_agent_{ticker}_final.zip")

    env_kwargs = {'df': training_df}
    env = make_vec_env(TradingEnv, n_envs=config.NUM_CPU_TO_USE, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs)
    
    validation_freq_per_cpu = 2048 * 20
    check_freq = max(validation_freq_per_cpu // config.NUM_CPU_TO_USE, 1)

    callbacks = [
        ValidationCallback(
            validation_dfs=validation_sets,
            ticker=ticker,
            save_dir=config.RL_MODEL_DIR,
            check_freq=check_freq,
            verbose=1
        )
    ]
    # Set device to "auto" to use GPU if available
    device = "auto"
    
    if os.path.exists(final_model_path):
        logging.info(f"Found existing model at {final_model_path}. Loading and continuing training.")
        model = PPO.load(final_model_path, env=env, device=device)
        new_logger = configure_logger(tensorboard_log="./ppo_trading_tensorboard/", tb_log_name="PPO")
        model.set_logger(new_logger)
        total_timesteps = config.CONTINUOUS_TRAINING_TIMESTEPS
    else:
        logging.info("No existing model found. Creating a new PPO agent.")
        model = PPO("MlpPolicy", env, verbose=1, device=device, tensorboard_log="./ppo_trading_tensorboard/", **config.PPO_HYPERPARAMS)
        total_timesteps = config.INITIAL_TRAINING_TIMESTEPS
    
    logging.info(f"Training on device: {model.device}")
    
    try:
        model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=callbacks)
    finally:
        if not os.path.exists(config.RL_MODEL_DIR):
            os.makedirs(config.RL_MODEL_DIR)
        model.save(final_model_path)
        logging.info(f"Training session finished. Final model saved to {final_model_path}")
        logging.info(f"Candidate models saved in '{config.RL_MODEL_DIR}'. Run backtester.py to find the best one.")

if __name__ == '__main__':
    main()