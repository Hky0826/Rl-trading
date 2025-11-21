# File: reporting.py
# Description: Updated reporting module with lot-based metrics and winning days calculation
# =============================================================================
import pandas as pd
import numpy as np
import config


def calculate_metrics(equity_curve, trades, max_drawdown, max_consecutive_dd_days=0):
    """
    Calculate comprehensive trading metrics with lot-based system support.
    
    Args:
        equity_curve: DataFrame with 'Time' and 'Equity' columns
        trades: DataFrame with trade details (using lot_size instead of size)
        max_drawdown: Maximum drawdown percentage
        max_consecutive_dd_days: Maximum consecutive drawdown days
    
    Returns:
        Dictionary of calculated metrics
    """
    metrics = {}
    
    # Basic Equity Metrics
    initial_equity = config.INITIAL_EQUITY
    final_equity = equity_curve['Equity'].iloc[-1] if not equity_curve.empty else initial_equity
    
    total_return = ((final_equity - initial_equity) / initial_equity) * 100
    metrics['Total Return (%)'] = round(total_return, 2)
    metrics['Final Equity'] = round(final_equity, 2)
    metrics['Max Drawdown (%)'] = round(max_drawdown, 2)
    metrics['Max Consecutive DD Days'] = int(max_consecutive_dd_days)
    
    # Trade Metrics
    if not trades.empty:
        total_trades = len(trades)
        winning_trades = (trades['pnl'] > 0).sum()
        losing_trades = (trades['pnl'] < 0).sum()
        
        metrics['Total Trades'] = total_trades
        metrics['Winning Trades'] = winning_trades
        metrics['Losing Trades'] = losing_trades
        metrics['Win Rate (%)'] = round((winning_trades / total_trades * 100), 2) if total_trades > 0 else 0
        
        # P&L Metrics
        total_pnl = trades['pnl'].sum()
        avg_win = trades[trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(trades[trades['pnl'] < 0]['pnl'].mean()) if losing_trades > 0 else 0
        
        metrics['Total P&L'] = round(total_pnl, 2)
        metrics['Average Win'] = round(avg_win, 2)
        metrics['Average Loss'] = round(avg_loss, 2)
        
        # Profit Factor
        gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
        metrics['Profit Factor'] = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0
        
        # Expectancy
        expectancy = (winning_trades / total_trades * avg_win) - (losing_trades / total_trades * avg_loss) if total_trades > 0 else 0
        metrics['Expectancy'] = round(expectancy, 2)
        
    else:
        metrics['Total Trades'] = 0
        metrics['Winning Trades'] = 0
        metrics['Losing Trades'] = 0
        metrics['Win Rate (%)'] = 0
        metrics['Total P&L'] = 0
        metrics['Average Win'] = 0
        metrics['Average Loss'] = 0
        metrics['Profit Factor'] = 0
        metrics['Expectancy'] = 0
    
    # Winning Days Calculation (replaces Winning Months)
    if not equity_curve.empty and len(equity_curve) > 1:
        equity_curve = equity_curve.copy()
        equity_curve['Date'] = pd.to_datetime(equity_curve['Time']).dt.date
        
        # Group by date and calculate daily returns
        daily_equity = equity_curve.groupby('Date')['Equity'].last()
        daily_returns = daily_equity.pct_change().dropna()
        
        winning_days = (daily_returns > 0).sum()
        total_days = len(daily_returns)
        
        metrics['Winning Days'] = f"{winning_days}/{total_days}"
        metrics['Winning Days (%)'] = round((winning_days / total_days * 100), 2) if total_days > 0 else 0
    else:
        metrics['Winning Days'] = "0/0"
        metrics['Winning Days (%)'] = 0
    
    # Risk-Adjusted Metrics
    if not equity_curve.empty and len(equity_curve) > 1:
        equity_curve = equity_curve.copy()
        equity_curve['returns'] = equity_curve['Equity'].pct_change().fillna(0)
        
        # Sharpe Ratio (Annualized)
        # Assume 252 trading days per year (standard for daily data)
        # For intraday data (M5), we have 288 candles per day (24h * 60min / 5min)
        periods_per_year = 252 * 288  # For M5 candles
        
        avg_return = equity_curve['returns'].mean()
        std_return = equity_curve['returns'].std()
        
        if std_return > 0:
            sharpe_ratio = (avg_return / std_return) * np.sqrt(periods_per_year)
        else:
            sharpe_ratio = 0
        
        metrics['Sharpe Ratio (Annualized)'] = round(sharpe_ratio, 2)
        
        # Sortino Ratio (using downside deviation)
        downside_returns = equity_curve['returns'][equity_curve['returns'] < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        
        if downside_std > 0:
            sortino_ratio = (avg_return / downside_std) * np.sqrt(periods_per_year)
        else:
            sortino_ratio = 0
        
        metrics['Sortino Ratio (Annualized)'] = round(sortino_ratio, 2)
    else:
        metrics['Sharpe Ratio (Annualized)'] = 0
        metrics['Sortino Ratio (Annualized)'] = 0
    
    # Calmar Ratio (CORRECTED CALCULATION)
    # Calmar Ratio = Annualized Return / Max Drawdown
    if not equity_curve.empty and len(equity_curve) > 1:
        # Calculate time period in years
        start_time = pd.to_datetime(equity_curve['Time'].iloc[0])
        end_time = pd.to_datetime(equity_curve['Time'].iloc[-1])
        days_elapsed = (end_time - start_time).days
        years_elapsed = days_elapsed / 365.25 if days_elapsed > 0 else 1
        
        # Annualized return
        if years_elapsed > 0:
            annualized_return = ((final_equity / initial_equity) ** (1 / years_elapsed) - 1) * 100
        else:
            annualized_return = total_return
        
        # Calmar Ratio = Annualized Return / Max Drawdown
        if max_drawdown > 0:
            calmar_ratio = annualized_return / max_drawdown
        else:
            calmar_ratio = 0
        
        metrics['Annualized Return (%)'] = round(annualized_return, 2)
        metrics['Calmar Ratio'] = round(calmar_ratio, 2)
    else:
        metrics['Annualized Return (%)'] = 0
        metrics['Calmar Ratio'] = 0
    
    # Recovery Factor
    if max_drawdown > 0:
        recovery_factor = total_return / max_drawdown
        metrics['Recovery Factor'] = round(recovery_factor, 2)
    else:
        metrics['Recovery Factor'] = 0
    
    return metrics


def format_performance_breakdown(trades_df):
    """
    Format a detailed performance breakdown by risk level and RR profile.
    
    Args:
        trades_df: DataFrame with trade details (using lot_size)
    
    Returns:
        Formatted string report
    """
    if trades_df.empty:
        return "No trades to analyze."
    
    report = []
    report.append("\n" + "="*80)
    report.append("📊 DETAILED PERFORMANCE BREAKDOWN")
    report.append("="*80)
    
    # Overall Summary
    total_trades = len(trades_df)
    winning_trades = (trades_df['pnl'] > 0).sum()
    losing_trades = (trades_df['pnl'] < 0).sum()
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    report.append(f"\n📈 OVERALL STATISTICS")
    report.append(f"  Total Trades: {total_trades}")
    report.append(f"  Winning Trades: {winning_trades} ({win_rate:.2f}%)")
    report.append(f"  Losing Trades: {losing_trades}")
    report.append(f"  Total P&L: ${trades_df['pnl'].sum():,.2f}")
    
    # Breakdown by Risk Level
    if 'risk_level' in trades_df.columns:
        report.append(f"\n💼 BREAKDOWN BY RISK LEVEL")
        report.append("-" * 80)
        
        risk_groups = trades_df.groupby('risk_level')
        for risk_level, group in risk_groups:
            wins = (group['pnl'] > 0).sum()
            losses = (group['pnl'] < 0).sum()
            total = len(group)
            win_rate_risk = (wins / total * 100) if total > 0 else 0
            total_pnl = group['pnl'].sum()
            avg_pnl = group['pnl'].mean()
            
            # Get risk percentage from config
            risk_idx = int(risk_level)
            if risk_idx < len(config.RISK_LEVELS):
                risk_pct = config.RISK_LEVELS[risk_idx]
            else:
                risk_pct = 0
            
            report.append(f"\n  Risk Level {risk_level} ({risk_pct}% per trade):")
            report.append(f"    Trades: {total} | Wins: {wins} | Losses: {losses}")
            report.append(f"    Win Rate: {win_rate_risk:.2f}%")
            report.append(f"    Total P&L: ${total_pnl:,.2f}")
            report.append(f"    Avg P&L: ${avg_pnl:,.2f}")
    
    # Breakdown by RR Profile
    if 'rr_profile_index' in trades_df.columns:
        report.append(f"\n🎯 BREAKDOWN BY RISK:REWARD PROFILE")
        report.append("-" * 80)
        
        rr_groups = trades_df.groupby('rr_profile_index')
        for rr_index, group in rr_groups:
            wins = (group['pnl'] > 0).sum()
            losses = (group['pnl'] < 0).sum()
            total = len(group)
            win_rate_rr = (wins / total * 100) if total > 0 else 0
            total_pnl = group['pnl'].sum()
            avg_pnl = group['pnl'].mean()
            
            # Get RR profile from config
            rr_idx = int(rr_index)
            if rr_idx < len(config.RR_PROFILES):
                sl_mult, tp_mult = config.RR_PROFILES[rr_idx]
                rr_ratio = f"1:{tp_mult/sl_mult:.1f}" if sl_mult != 0 else "N/A"
            else:
                rr_ratio = "Unknown"
            
            report.append(f"\n  RR Profile {rr_index} (Ratio {rr_ratio}):")
            report.append(f"    Trades: {total} | Wins: {wins} | Losses: {losses}")
            report.append(f"    Win Rate: {win_rate_rr:.2f}%")
            report.append(f"    Total P&L: ${total_pnl:,.2f}")
            report.append(f"    Avg P&L: ${avg_pnl:,.2f}")
    
    # Exit Reason Analysis
    if 'exit_reason' in trades_df.columns:
        report.append(f"\n🚪 EXIT REASON ANALYSIS")
        report.append("-" * 80)
        
        exit_groups = trades_df.groupby('exit_reason')
        for exit_reason, group in exit_groups:
            total = len(group)
            total_pnl = group['pnl'].sum()
            avg_pnl = group['pnl'].mean()
            pct_of_total = (total / len(trades_df) * 100)
            
            report.append(f"\n  {exit_reason}:")
            report.append(f"    Count: {total} ({pct_of_total:.1f}% of all trades)")
            report.append(f"    Total P&L: ${total_pnl:,.2f}")
            report.append(f"    Avg P&L: ${avg_pnl:,.2f}")
    
    report.append("\n" + "="*80)
    
    return "\n".join(report)


def generate_trade_summary_csv(trades_df, output_path="trade_summary.csv"):
    """
    Generate a CSV summary of all trades.
    
    Args:
        trades_df: DataFrame with trade details
        output_path: Path to save the CSV file
    """
    if trades_df.empty:
        print("No trades to export.")
        return
    
    try:
        # Select relevant columns for export
        export_columns = [
            'close_time', 'type', 'price', 'close_price', 'sl', 'tp',
            'lot_size', 'pnl', 'pnl_percent', 'exit_reason',
            'risk_level', 'rr_profile_index'
        ]
        
        # Filter to only existing columns
        available_columns = [col for col in export_columns if col in trades_df.columns]
        
        trades_export = trades_df[available_columns].copy()
        
        # Format numeric columns
        if 'pnl' in trades_export.columns:
            trades_export['pnl'] = trades_export['pnl'].round(2)
        if 'pnl_percent' in trades_export.columns:
            trades_export['pnl_percent'] = (trades_export['pnl_percent'] * 100).round(2)
        
        trades_export.to_csv(output_path, index=False)
        print(f"✅ Trade summary saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error generating trade summary: {e}")