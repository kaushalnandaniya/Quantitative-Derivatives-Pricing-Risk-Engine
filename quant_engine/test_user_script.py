import pandas as pd
import numpy as np
from services.pine_executor import run_pine_backtest

df = pd.DataFrame({
    'open': np.random.randn(100),
    'high': np.random.randn(100),
    'low': np.random.randn(100),
    'close': np.random.randn(100),
    'volume': np.random.randint(100, 1000, 100).astype(float)
}, index=pd.date_range('2025-01-01', periods=100))

script = """
//@version=5
strategy("Enhanced Delta Neutral Volume Strategy", overlay=true, default_qty_type=strategy.percent_of_equity, default_qty_value=10)

// Input for Volume Analysis
volumeThreshold = input.int(50000, title="Volume Threshold", tooltip="Minimum volume to consider for trading")
volumeLookback = input.int(10, title="Volume Lookback Period", tooltip="Period to average volume")
deltaNeutralRatio = input.float(1.0, title="Delta Neutral Ratio", tooltip="Ratio for hedging positions")

// Input for Moving Average Filter
maLength = input.int(50, title="Moving Average Length", tooltip="Period for the moving average filter")
maFilter = ta.sma(close, maLength)

// Calculate average volume
avgVolume = ta.sma(volume, volumeLookback)

// Long and short conditions based on volume and moving average filter
longCondition = volume > avgVolume and close > maFilter
shortCondition = volume < avgVolume and close < maFilter

// Execute strategy
if (longCondition and strategy.opentrades < 2)
    strategy.entry("Long", strategy.long, qty = deltaNeutralRatio)

if (shortCondition and strategy.opentrades < 2)
    strategy.entry("Short", strategy.short, qty = deltaNeutralRatio)

// Exit conditions based on volume reversion or MA crossing
exitLongCondition = shortCondition or ta.crossunder(close, maFilter)
exitShortCondition = longCondition or ta.crossover(close, maFilter)

if (exitLongCondition)
    strategy.close("Long")

if (exitShortCondition)
    strategy.close("Short")

// Plot the volume, average volume, and moving average for reference
plot(volume, color=color.blue, title="Volume")
plot(avgVolume, color=color.red, title="Average Volume")
plot(maFilter, color=color.green, title="MA Filter")
"""

try:
    results = run_pine_backtest(script, df)
    print("Success:", results['summary'])
except Exception as e:
    import traceback
    traceback.print_exc()
