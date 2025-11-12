# File: rl_callbacks.py
# Description: (FINAL) The validation callback uses the synchronized backtester.
# =============================================================================
import os
import logging
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import config

from backtester import run_rl_backtest
from reporting import calculate_metrics

class ValidationCallback(BaseCallback):
    def __init__(self, validation_dfs, ticker, save_dir, check_freq, verbose=1):
        super(ValidationCallback, self).__init__(verbose)
        self.validation_dfs = validation_dfs
        self.ticker = ticker
        self.save_dir = save_dir
        self.check_freq = check_freq

    def _init_callback(self) -> None:
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            logging.info(f"--- Running validation at step {self.n_calls} ---")
            
            all_metrics = []
            for i, val_df in enumerate(self.validation_dfs):
                # This now calls the correct, synchronized backtester
                equity_curve, trades, max_drawdown = run_rl_backtest(
                    data_df=val_df, model=self.model, ticker=self.ticker
                )
                if not equity_curve.empty:
                    metrics = calculate_metrics(equity_curve, trades, max_drawdown)
                    all_metrics.append(metrics)
            
            if not all_metrics:
                logging.warning("Validation backtests produced no results. Skipping.")
                return True

            metric_values = [m["Profit Factor"] for m in all_metrics]
            max_drawdowns = [m["Max Drawdown (%)"] for m in all_metrics]

            if any(dd > config.MAX_VALIDATION_DRAWDOWN_PERCENT for dd in max_drawdowns):
                logging.warning(f"Model FAILED safety check. Drawdowns: {[f'{dd:.1f}%' for dd in max_drawdowns]}. Rejecting.")
                return True

            mean_pf = np.mean(metric_values)
            median_pf = np.median(metric_values)
            
            logging.info(f"Mean PF: {mean_pf:.2f}, Median PF: {median_pf:.2f} (Threshold: {config.PROFIT_FACTOR_THRESHOLD})")

            if mean_pf > config.PROFIT_FACTOR_THRESHOLD:
                save_path = os.path.join(self.save_dir, f"model_mean_{mean_pf:.2f}_step_{self.n_calls}.zip")
                logging.info(f"*** Candidate model (Mean) found! Saving to {save_path} ***")
                self.model.save(save_path)
            
            if median_pf > config.PROFIT_FACTOR_THRESHOLD:
                save_path = os.path.join(self.save_dir, f"model_median_{median_pf:.2f}_step_{self.n_calls}.zip")
                logging.info(f"*** Candidate model (Median) found! Saving to {save_path} ***")
                self.model.save(save_path)
        
        return True