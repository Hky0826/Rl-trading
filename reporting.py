# File: reporting.py
# Description: (CORRECTED) Calmar Ratio calculation is now fixed.
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime
import config

def calculate_metrics(equity_curve, trades, max_drawdown):
    initial_equity = config.INITIAL_EQUITY
    total_trades = 0; win_rate = 0.0; avg_win = 0.0; avg_loss = 0.0; profit_factor = 0.0
    if trades is not None and not trades.empty:
        total_trades = len(trades); wins = trades[trades['pnl'] > 0]; losses = trades[trades['pnl'] <= 0]
        if total_trades > 0: win_rate = (len(wins) / total_trades) * 100
        if not wins.empty: avg_win = wins['pnl'].mean()
        if not losses.empty:
            avg_loss = losses['pnl'].mean()
            if losses['pnl'].sum() != 0: profit_factor = abs(wins['pnl'].sum() / losses['pnl'].sum())

    final_equity = initial_equity; total_return = 0.0; sharpe_ratio = 0.0; sortino_ratio = 0.0; winning_months_pct = 0.0; calmar_ratio = 0.0

    if equity_curve is not None and not equity_curve.empty and len(equity_curve) > 1:
        final_equity = equity_curve['Equity'].iloc[-1]
        total_return = ((final_equity - initial_equity) / initial_equity) * 100
        returns = equity_curve['Equity'].pct_change().dropna()
        if returns.std() != 0:
            annualization_factor = np.sqrt(252 * config.CANDLES_PER_DAY)
            sharpe_ratio = (returns.mean() / returns.std()) * annualization_factor
        
        monthly_returns = equity_curve.set_index('Time')['Equity'].resample('M').last().pct_change().dropna()
        if not monthly_returns.empty:
            winning_months_pct = (monthly_returns > 0).sum() / len(monthly_returns) * 100
            downside_returns = monthly_returns[monthly_returns < 0]
            downside_std = downside_returns.std()
            if downside_std is not None and downside_std > 0:
                annualized_mean_return = monthly_returns.mean() * 12
                annualized_downside_std = downside_std * np.sqrt(12)
                sortino_ratio = annualized_mean_return / annualized_downside_std
        
        # --- UPDATED: Correct Calmar Ratio Calculation ---
        start_date = equity_curve['Time'].iloc[0]
        end_date = equity_curve['Time'].iloc[-1]
        num_years = (end_date - start_date).days / 365.25
        if num_years > 0:
            annualized_return = ((final_equity / initial_equity) ** (1 / num_years) - 1) * 100
            if max_drawdown > 0:
                calmar_ratio = annualized_return / max_drawdown

    return {
        "Final Equity": final_equity, "Total Return (%)": total_return, "Max Drawdown (%)": max_drawdown,
        "Sharpe Ratio (Annualized)": sharpe_ratio, "Sortino Ratio (Annualized)": sortino_ratio,
        "Winning Months (%)": winning_months_pct, "Calmar Ratio": calmar_ratio, "Total Trades": total_trades,
        "Win Rate (%)": win_rate, "Average Win ($)": avg_win, "Average Loss ($)": avg_loss, "Profit Factor": profit_factor
    }

def print_report(ticker, metrics):
    # This function is unchanged
    print("\n" + "="*80 + f"\nBACKTEST RESULTS FOR: {ticker}\n" + "="*80)
    print(f"{'Final Equity':<30}: ${metrics['Final Equity']:,.2f}"); print(f"{'Total Return (%)':<30}: {metrics['Total Return (%)']:.2f}%"); print(f"{'Max Drawdown (%)':<30}: {metrics['Max Drawdown (%)']:.2f}%")
    print("-" * 40); print(f"{'Sharpe Ratio (Annualized)':<30}: {metrics['Sharpe Ratio (Annualized)']:.2f}"); print(f"{'Sortino Ratio (Annualized)':<30}: {metrics['Sortino Ratio (Annualized)']:.2f}"); print(f"{'Calmar Ratio':<30}: {metrics['Calmar Ratio']:.2f}")
    print("-" * 40); print(f"{'Winning Months (%)':<30}: {metrics['Winning Months (%)']:.2f}%"); print(f"{'Total Trades':<30}: {metrics['Total Trades']}"); print(f"{'Win Rate (%)':<30}: {metrics['Win Rate (%)']:.2f}%")
    print(f"{'Average Win ($)':<30}: ${metrics['Average Win ($)']:.2f}"); print(f"{'Average Loss ($)':<30}: ${metrics['Average Loss ($)']:.2f}"); print(f"{'Profit Factor':<30}: {metrics['Profit Factor']:.2f}"); print("="*80 + "\n")

def plot_equity_curve(ticker, equity_curve):
    if equity_curve is None or equity_curve.empty: return
    plt.style.use('seaborn-v0_8-darkgrid'); fig, ax = plt.subplots(figsize=(15, 7)); ax.plot(equity_curve['Time'], equity_curve['Equity'], label='Equity Curve', color='royalblue')
    ax.set_title(f'Equity Curve for {ticker}', fontsize=16); ax.set_xlabel('Date'); ax.set_ylabel('Equity ($)')
    ax.legend(); ax.grid(True, which='both', linestyle='--', linewidth=0.5); plt.tight_layout(); plt.show()

def save_trades_to_csv(trades_df, ticker):
    if trades_df is None or trades_df.empty: logging.warning(f"No trades to save for {ticker}."); return
    if not os.path.exists(config.TRADE_LOG_DIR): os.makedirs(config.TRADE_LOG_DIR)
    log_path = os.path.join(config.TRADE_LOG_DIR, f"tradelog_{ticker}.csv"); trades_df.to_csv(log_path, index=False)
    logging.info(f"Trade log saved to {log_path}")

# NEW: This function generates the detailed breakdown report
def format_performance_breakdown(trades_df):
    """Creates a formatted string report for performance by risk and RR profile."""
    if trades_df.empty:
        return "No trades to analyze for breakdown."

    # --- Breakdown by Risk Level ---
    risk_breakdown = []
    if 'risk_level' in trades_df.columns:
        risk_groups = trades_df.groupby('risk_level')
        for risk, group in risk_groups:
            win_rate = (group['pnl'] > 0).sum() / len(group) * 100
            risk_breakdown.append({
                "Risk Level (%)": risk,
                "Trades": len(group),
                "Win Rate (%)": f"{win_rate:.2f}"
            })
    
    # --- Breakdown by RR Profile ---
    rr_breakdown = []
    if 'rr_profile_index' in trades_df.columns:
        rr_groups = trades_df.groupby('rr_profile_index')
        for index, group in rr_groups:
            win_rate = (group['pnl'] > 0).sum() / len(group) * 100
            sl_mult, tp_mult = config.RR_PROFILES[index]
            rr_ratio = f"1:{tp_mult/sl_mult:.1f}"
            rr_breakdown.append({
                "RR Profile (SL, TP)": f"({sl_mult}, {tp_mult})",
                "RR Ratio": rr_ratio,
                "Trades": len(group),
                "Win Rate (%)": f"{win_rate:.2f}"
            })

    # --- Format into a nice string ---
    report_str = "\n" + "-"*80 + "\nPERFORMANCE BREAKDOWN\n" + "-"*80 + "\n"
    if risk_breakdown:
        report_str += "By Risk Level:\n"
        report_str += pd.DataFrame(risk_breakdown).to_string(index=False) + "\n\n"
    if rr_breakdown:
        report_str += "By Risk/Reward Profile:\n"
        report_str += pd.DataFrame(rr_breakdown).to_string(index=False)
    report_str += "\n" + "="*80
    
    return report_str