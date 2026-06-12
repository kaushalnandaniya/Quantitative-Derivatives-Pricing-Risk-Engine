"""Tests for Pine Script Parser and Executor."""
import pytest
import numpy as np
import pandas as pd
from services.pine_parser import tokenize, PineParser, parse_pine_script
from services.pine_executor import (
    run_pine_backtest, _ta_sma, _ta_ema, _ta_rsi,
    _ta_crossover, _ta_crossunder
)


def _make_ohlcv(n=200, base=100.0):
    """Generate a simple test OHLCV DataFrame."""
    rng = np.random.default_rng(42)
    close = np.cumsum(rng.normal(0.1, 1.0, n)) + base
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    open_ = low + (high - low) * rng.uniform(0.3, 0.7, n)
    volume = rng.integers(100000, 1000000, n).astype(float)

    dates = pd.date_range('2025-01-01', periods=n, freq='B')
    return pd.DataFrame({
        'open': open_, 'high': high, 'low': low,
        'close': close, 'volume': volume
    }, index=dates)


# ===== Parser Tests =====

def test_tokenize_simple():
    tokens = tokenize("fast = ta.sma(close, 10)")
    names = [t.type.name for t in tokens if t.type.name != 'NEWLINE' and t.type.name != 'EOF']
    assert 'IDENTIFIER' in names
    assert 'ASSIGN' in names


def test_parse_sma_crossover():
    script = """
//@version=5
strategy("Test", overlay=true)
fast = ta.sma(close, 10)
slow = ta.sma(close, 30)
if ta.crossover(fast, slow)
    strategy.entry("Long", strategy.long)
if ta.crossunder(fast, slow)
    strategy.close("Long")
"""
    ast, config = parse_pine_script(script)
    assert len(ast) > 0
    assert config.get('title') == 'Test'


def test_parse_for_loop():
    script = """
//@version=5
strategy("Test")
sum = 0.0
for i = 1 to 10 by 2
    sum := sum + i
"""
    ast, config = parse_pine_script(script)
    from services.pine_parser import ForLoop
    for_loops = [n for n in ast if isinstance(n, ForLoop)]
    assert len(for_loops) == 1
    assert for_loops[0].var_name == 'i'


def test_parse_tuple_assignment():
    script = """
//@version=5
strategy("Test")
[macdLine, signalLine, histLine] = ta.macd(close, 12, 26, 9)
"""
    ast, config = parse_pine_script(script)
    from services.pine_parser import TupleAssignment
    tuples = [n for n in ast if isinstance(n, TupleAssignment)]
    assert len(tuples) == 1
    assert tuples[0].names == ['macdLine', 'signalLine', 'histLine']


# ===== Indicator Tests =====

def test_ta_sma():
    data = np.arange(20, dtype=float)
    result = _ta_sma(data, 5)
    assert np.isnan(result[3])
    assert result[4] == pytest.approx(2.0, abs=0.01)


def test_ta_ema():
    data = np.ones(20)
    result = _ta_ema(data, 10)
    assert result[-1] == pytest.approx(1.0, abs=0.01)


def test_ta_rsi():
    data = np.linspace(10, 30, 50)  # Steadily rising
    result = _ta_rsi(data, 14)
    assert result[-1] > 90  # Strong uptrend = high RSI


def test_ta_crossover():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
    result = _ta_crossover(a, b)
    assert result[3] == True  # a crosses above b at index 3


def test_ta_crossunder():
    a = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    b = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
    result = _ta_crossunder(a, b)
    assert result[3] == True  # a crosses below b at index 3


# ===== Backtest Execution Tests =====

def test_sma_crossover_backtest():
    df = _make_ohlcv(200)
    script = """
//@version=5
strategy("SMA Cross", overlay=true)
fast = ta.sma(close, 10)
slow = ta.sma(close, 30)
if ta.crossover(fast, slow)
    strategy.entry("Long", strategy.long)
if ta.crossunder(fast, slow)
    strategy.close("Long")
"""
    results = run_pine_backtest(script, df)
    assert 'trades' in results
    assert 'summary' in results
    assert 'equity_curve' in results
    assert results['summary']['total_trades'] >= 0


def test_rsi_strategy_backtest():
    df = _make_ohlcv(200)
    script = """
//@version=5
strategy("RSI Strategy", overlay=true)
rsiVal = ta.rsi(close, 14)
if rsiVal < 30
    strategy.entry("Long", strategy.long)
if rsiVal > 70
    strategy.close("Long")
"""
    results = run_pine_backtest(script, df)
    assert 'trades' in results
    assert results['summary']['total_trades'] >= 0
def test_tuple_assignment_backtest():
    df = _make_ohlcv(200)
    script = """
//@version=5
strategy("MACD Tuple Cross", overlay=true)
[macdLine, signalLine, histLine] = ta.macd(close, 12, 26, 9)
if ta.crossover(macdLine, signalLine)
    strategy.entry("Long", strategy.long)
if ta.crossunder(macdLine, signalLine)
    strategy.close("Long")
"""
    results = run_pine_backtest(script, df)
    assert 'trades' in results
    assert results['summary']['total_trades'] >= 0


def test_for_loop_backtest():
    df = _make_ohlcv(200)
    script = """
//@version=5
strategy("For Loop", overlay=true)
sumVal = 0.0
for i = 1 to 5
    sumVal := sumVal + 1
if sumVal == 5
    strategy.entry("Long", strategy.long)
"""
    results = run_pine_backtest(script, df)
    assert 'trades' in results
    assert len(results['trades']) >= 0


def test_custom_function_backtest():
    df = _make_ohlcv(200)
    script = """
//@version=5
strategy("Custom Func", overlay=true)

my_func(src, len) =>
    sum = 0.0
    for i = 1 to len
        sum := sum + src
    sum / len

val = my_func(close, 5)
if val > close
    strategy.entry("Long", strategy.long)
"""
    results = run_pine_backtest(script, df)
    assert 'trades' in results
    assert len(results['trades']) >= 0
