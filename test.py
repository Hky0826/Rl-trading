import MetaTrader5 as mt5

mt5.initialize(login=1100123044, server="JustMarkets-Demo2", password="Kahyuen2608@")

info = mt5.symbol_info("EURUSD.m")
print("Symbol info:", info)

if info is None:
    print("EURUSD.m not available in this account!")
else:
    print("EURUSD.m available, subscribed =", info.visible)
    if not info.visible:
        mt5.symbol_select("EURUSD.m", True)
