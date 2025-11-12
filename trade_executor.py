# File: trade_executor.py
# Description: (CORRECTED) Handles all live order execution with MT5.
# =============================================================================
import MetaTrader5 as mt5
import logging
import config
import time

def calculate_lot_size(symbol, stop_loss_price, risk_percent):
    account_info = mt5.account_info()
    if account_info is None: return None
    equity = account_info.equity
    risk_amount = equity * (risk_percent / 100)
    tick = mt5.symbol_info_tick(symbol)
    if tick is None: return None
    is_buy = stop_loss_price < tick.bid
    request_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
    entry_price = tick.ask if is_buy else tick.bid
    loss_for_one_lot = mt5.order_calc_profit(request_type, symbol, 1.0, entry_price, stop_loss_price)
    if loss_for_one_lot is None or loss_for_one_lot >= 0: return None
    risk_per_lot = abs(loss_for_one_lot)
    if risk_per_lot == 0: return None
    lot_size = risk_amount / risk_per_lot
    volume_step = mt5.symbol_info(symbol).volume_step
    lot_size = (lot_size // volume_step) * volume_step
    return lot_size

def place_trade(trade_type, symbol, lot_size, stop_loss, take_profit):
    order_type = mt5.ORDER_TYPE_BUY if trade_type == 'LONG' else mt5.ORDER_TYPE_SELL
    price = mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid
    request = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": lot_size,
        "type": order_type, "price": price, "sl": stop_loss, "tp": take_profit,
        "magic": config.MAGIC_NUMBER, "comment": "RL Bot Trade",
        "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_FOK
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error(f"Order Send Failed. Retcode: {result.retcode}, Comment: {result.comment}")
        return None
    logging.info(f"Trade placed successfully! Ticket: {result.order}")
    time.sleep(1)
    # UPDATED: Return the entire result object, not just the order number
    return result

def count_open_positions_for_symbol(symbol):
    positions = mt5.positions_get(symbol=symbol)
    if positions is None or len(positions) == 0: return 0
    return len([p for p in positions if p.magic == config.MAGIC_NUMBER])

def get_total_open_positions():
    positions = mt5.positions_get()
    if positions is None: return 0
    return len([p for p in positions if p.magic == config.MAGIC_NUMBER])