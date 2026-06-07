"""
Pine Script Executor
======================
Executes a parsed Pine Script AST against OHLCV data to generate
trade signals, compute P&L, and produce backtest results.
"""

import logging
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

from services.pine_parser import (
    ASTNode, NumberLiteral, StringLiteral, BoolLiteral, NALiteral,
    Identifier, BinaryOp, UnaryOp, FunctionCall, MemberAccess,
    HistoryRef, Assignment, IfStatement, StrategyCall, TernaryOp,
    InputCall, parse_pine_script,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Built-in Indicator Functions
# =============================================================================

def _ta_sma(series: np.ndarray, length: int) -> np.ndarray:
    result = np.full_like(series, np.nan)
    for i in range(length - 1, len(series)):
        result[i] = np.mean(series[i - length + 1:i + 1])
    return result


def _ta_ema(series: np.ndarray, length: int) -> np.ndarray:
    result = np.full_like(series, np.nan)
    alpha = 2.0 / (length + 1)
    result[0] = series[0]
    for i in range(1, len(series)):
        if np.isnan(result[i - 1]):
            result[i] = series[i]
        else:
            result[i] = alpha * series[i] + (1 - alpha) * result[i - 1]
    return result


def _ta_rsi(series: np.ndarray, length: int) -> np.ndarray:
    result = np.full_like(series, np.nan)
    deltas = np.diff(series, prepend=series[0])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.full_like(series, np.nan)
    avg_loss = np.full_like(series, np.nan)

    if length < len(series):
        avg_gain[length] = np.mean(gains[1:length + 1])
        avg_loss[length] = np.mean(losses[1:length + 1])

        for i in range(length + 1, len(series)):
            avg_gain[i] = (avg_gain[i - 1] * (length - 1) + gains[i]) / length
            avg_loss[i] = (avg_loss[i - 1] * (length - 1) + losses[i]) / length

        for i in range(length, len(series)):
            if avg_loss[i] == 0:
                result[i] = 100.0
            else:
                rs = avg_gain[i] / avg_loss[i]
                result[i] = 100.0 - 100.0 / (1.0 + rs)

    return result


def _ta_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    return _ta_sma(tr, length)


def _ta_stdev(series: np.ndarray, length: int) -> np.ndarray:
    result = np.full_like(series, np.nan)
    for i in range(length - 1, len(series)):
        result[i] = np.std(series[i - length + 1:i + 1], ddof=0)
    return result


def _ta_highest(series: np.ndarray, length: int) -> np.ndarray:
    result = np.full_like(series, np.nan)
    for i in range(length - 1, len(series)):
        result[i] = np.max(series[i - length + 1:i + 1])
    return result


def _ta_lowest(series: np.ndarray, length: int) -> np.ndarray:
    result = np.full_like(series, np.nan)
    for i in range(length - 1, len(series)):
        result[i] = np.min(series[i - length + 1:i + 1])
    return result


def _ta_macd(series: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = _ta_ema(series, fast)
    slow_ema = _ta_ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = _ta_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _ta_crossover(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    result = np.zeros(len(a), dtype=bool)
    for i in range(1, len(a)):
        if not np.isnan(a[i]) and not np.isnan(b[i]) and not np.isnan(a[i-1]) and not np.isnan(b[i-1]):
            result[i] = a[i] > b[i] and a[i - 1] <= b[i - 1]
    return result


def _ta_crossunder(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    result = np.zeros(len(a), dtype=bool)
    for i in range(1, len(a)):
        if not np.isnan(a[i]) and not np.isnan(b[i]) and not np.isnan(a[i-1]) and not np.isnan(b[i-1]):
            result[i] = a[i] < b[i] and a[i - 1] >= b[i - 1]
    return result


def _ta_change(series: np.ndarray, length: int = 1) -> np.ndarray:
    result = np.full_like(series, np.nan)
    for i in range(length, len(series)):
        result[i] = series[i] - series[i - length]
    return result


# Bollinger Bands
def _ta_bb(series: np.ndarray, length: int = 20, mult: float = 2.0):
    basis = _ta_sma(series, length)
    dev = _ta_stdev(series, length)
    upper = basis + mult * dev
    lower = basis - mult * dev
    return basis, upper, lower


# =============================================================================
# Executor
# =============================================================================

class PineExecutor:
    """Execute Pine Script AST bar-by-bar against OHLCV data."""

    def __init__(self, df: pd.DataFrame, user_inputs: Dict[str, Any] = None):
        """
        Args:
            df: DataFrame with columns: open, high, low, close, volume (lowercase)
            user_inputs: Override default input values
        """
        self.df = df.copy()
        self.n_bars = len(df)
        self.user_inputs = user_inputs or {}

        # Built-in series
        self.series = {
            'open': df['open'].values.astype(float),
            'high': df['high'].values.astype(float),
            'low': df['low'].values.astype(float),
            'close': df['close'].values.astype(float),
            'volume': df['volume'].values.astype(float),
            'bar_index': np.arange(self.n_bars, dtype=float),
            'hl2': ((df['high'] + df['low']) / 2).values.astype(float),
            'hlc3': ((df['high'] + df['low'] + df['close']) / 3).values.astype(float),
            'ohlc4': ((df['open'] + df['high'] + df['low'] + df['close']) / 4).values.astype(float),
        }

        # Variables (computed once or bar-by-bar)
        self.variables: Dict[str, Any] = {}
        self.var_variables: Dict[str, Any] = {}  # persistent across bars

        # Strategy state
        self.position = 0  # 1=long, -1=short, 0=flat
        self.entry_price = 0.0
        self.trades: List[Dict] = []
        self.signals: List[Dict] = []

        # Current bar index (for bar-by-bar execution)
        self.bar_idx = 0

    def execute(self, ast: List[ASTNode]) -> Dict[str, Any]:
        """Execute the full AST and return backtest results."""
        # First pass: evaluate all series-level computations
        self._precompute_series(ast)

        # Second pass: bar-by-bar execution for strategy logic
        for i in range(self.n_bars):
            self.bar_idx = i
            for node in ast:
                self._exec_node(node, i)

        # Close any open position at the end
        if self.position != 0:
            exit_price = float(self.series['close'][-1])
            pnl = (exit_price - self.entry_price) * self.position
            self.trades.append({
                'entry_bar': self.trades[-1]['entry_bar'] if self.trades else 0,
                'exit_bar': self.n_bars - 1,
                'entry_date': str(self.df.index[self.trades[-1]['entry_bar'] if self.trades else 0]),
                'exit_date': str(self.df.index[-1]),
                'side': 'long' if self.position > 0 else 'short',
                'entry_price': self.entry_price,
                'exit_price': exit_price,
                'pnl': round(pnl, 2),
                'pnl_pct': round(pnl / self.entry_price * 100, 2) if self.entry_price else 0,
            })
            self.position = 0

        return self._build_results()

    def _precompute_series(self, ast: List[ASTNode]):
        """Pre-compute indicator series from assignments."""
        for node in ast:
            if isinstance(node, Assignment) and not isinstance(node.value, (StrategyCall, IfStatement)):
                try:
                    val = self._eval_series(node.value)
                    if isinstance(val, np.ndarray):
                        self.series[node.name] = val
                        self.variables[node.name] = val
                except Exception:
                    pass  # Will be evaluated bar-by-bar

    def _eval_series(self, node: ASTNode) -> Any:
        """Evaluate an expression as a full series (vectorized)."""
        if isinstance(node, NumberLiteral):
            return np.full(self.n_bars, node.value)
        if isinstance(node, BoolLiteral):
            return np.full(self.n_bars, node.value)
        if isinstance(node, NALiteral):
            return np.full(self.n_bars, np.nan)
        if isinstance(node, StringLiteral):
            return node.value
        if isinstance(node, Identifier):
            name = node.name
            if name in self.series:
                return self.series[name]
            if name in self.variables:
                return self.variables[name]
            if name in ('strategy.long', 'strategy.short'):
                return name
            return np.full(self.n_bars, np.nan)

        if isinstance(node, FunctionCall):
            return self._eval_function_series(node)

        if isinstance(node, BinaryOp):
            left = self._eval_series(node.left)
            right = self._eval_series(node.right)
            return self._binary_op_series(node.op, left, right)

        if isinstance(node, UnaryOp):
            operand = self._eval_series(node.operand)
            if node.op == '-':
                return -operand if isinstance(operand, np.ndarray) else -operand
            if node.op == 'not':
                return ~operand if isinstance(operand, np.ndarray) else not operand

        if isinstance(node, HistoryRef):
            series = self._eval_series(node.series)
            offset = self._eval_scalar(node.offset, 0)
            if isinstance(series, np.ndarray) and isinstance(offset, (int, float)):
                return np.roll(series, int(offset))

        if isinstance(node, TernaryOp):
            cond = self._eval_series(node.condition)
            true_val = self._eval_series(node.true_val)
            false_val = self._eval_series(node.false_val)
            if isinstance(cond, np.ndarray):
                return np.where(cond, true_val, false_val)

        if isinstance(node, InputCall):
            return self._eval_input(node)

        return np.full(self.n_bars, np.nan)

    def _eval_function_series(self, node: FunctionCall) -> Any:
        """Evaluate a built-in function call as a series."""
        name = node.name
        args = [self._eval_series(a) for a in node.args]
        kwargs = {k: self._eval_series(v) for k, v in node.kwargs.items()}

        # ta.* functions
        if name in ('ta.sma', 'sma'):
            src = args[0] if args else kwargs.get('source', self.series['close'])
            length = int(self._to_scalar(args[1] if len(args) > 1 else kwargs.get('length', 14)))
            return _ta_sma(src, length)

        if name in ('ta.ema', 'ema'):
            src = args[0] if args else kwargs.get('source', self.series['close'])
            length = int(self._to_scalar(args[1] if len(args) > 1 else kwargs.get('length', 14)))
            return _ta_ema(src, length)

        if name in ('ta.rsi', 'rsi'):
            src = args[0] if args else kwargs.get('source', self.series['close'])
            length = int(self._to_scalar(args[1] if len(args) > 1 else kwargs.get('length', 14)))
            return _ta_rsi(src, length)

        if name in ('ta.atr', 'atr'):
            length = int(self._to_scalar(args[0] if args else kwargs.get('length', 14)))
            return _ta_atr(self.series['high'], self.series['low'], self.series['close'], length)

        if name in ('ta.stdev', 'stdev'):
            src = args[0] if args else self.series['close']
            length = int(self._to_scalar(args[1] if len(args) > 1 else kwargs.get('length', 20)))
            return _ta_stdev(src, length)

        if name in ('ta.highest', 'highest'):
            src = args[0] if args else self.series['high']
            length = int(self._to_scalar(args[1] if len(args) > 1 else kwargs.get('length', 14)))
            return _ta_highest(src, length)

        if name in ('ta.lowest', 'lowest'):
            src = args[0] if args else self.series['low']
            length = int(self._to_scalar(args[1] if len(args) > 1 else kwargs.get('length', 14)))
            return _ta_lowest(src, length)

        if name in ('ta.crossover', 'crossover'):
            return _ta_crossover(args[0], args[1])

        if name in ('ta.crossunder', 'crossunder'):
            return _ta_crossunder(args[0], args[1])

        if name in ('ta.change', 'change'):
            src = args[0] if args else self.series['close']
            length = int(self._to_scalar(args[1] if len(args) > 1 else 1))
            return _ta_change(src, length)

        if name in ('ta.macd', 'macd'):
            src = args[0] if args else self.series['close']
            fast = int(self._to_scalar(kwargs.get('fastlen', 12)))
            slow = int(self._to_scalar(kwargs.get('slowlen', 26)))
            sig = int(self._to_scalar(kwargs.get('siglen', 9)))
            macd_line, signal_line, hist = _ta_macd(src, fast, slow, sig)
            return macd_line  # Return main line; user can access components via tuple

        if name in ('ta.bb', 'bb'):
            src = args[0] if args else self.series['close']
            length = int(self._to_scalar(args[1] if len(args) > 1 else 20))
            mult = float(self._to_scalar(args[2] if len(args) > 2 else 2.0))
            basis, upper, lower = _ta_bb(src, length, mult)
            return basis  # Return middle band

        if name in ('math.abs', 'abs'):
            return np.abs(args[0])

        if name in ('math.max', 'max'):
            return np.maximum(args[0], args[1])

        if name in ('math.min', 'min'):
            return np.minimum(args[0], args[1])

        if name in ('math.sqrt', 'sqrt'):
            return np.sqrt(args[0])

        if name in ('math.log', 'log'):
            return np.log(args[0])

        if name in ('nz',):
            val = args[0]
            replacement = args[1] if len(args) > 1 else np.zeros_like(val)
            return np.where(np.isnan(val), replacement, val)

        # Unknown — return NaN
        return np.full(self.n_bars, np.nan)

    def _binary_op_series(self, op: str, left: Any, right: Any) -> Any:
        if op == '+': return left + right
        if op == '-': return left - right
        if op == '*': return left * right
        if op == '/':
            if isinstance(right, np.ndarray):
                return np.divide(left, right, where=right != 0, out=np.full_like(left, np.nan, dtype=float))
            return left / right if right != 0 else np.nan
        if op == '%': return left % right
        if op == '>': return left > right
        if op == '<': return left < right
        if op == '>=': return left >= right
        if op == '<=': return left <= right
        if op == '==': return left == right
        if op == '!=': return left != right
        if op == 'and': return np.logical_and(left, right)
        if op == 'or': return np.logical_or(left, right)
        return np.full(self.n_bars, np.nan)

    def _to_scalar(self, val) -> float:
        if isinstance(val, np.ndarray):
            return float(val[0]) if len(val) > 0 else 0.0
        return float(val) if val is not None else 0.0

    def _eval_scalar(self, node: ASTNode, bar_idx: int) -> Any:
        """Evaluate a node at a specific bar."""
        if isinstance(node, NumberLiteral):
            return node.value
        if isinstance(node, StringLiteral):
            return node.value
        if isinstance(node, BoolLiteral):
            return node.value
        if isinstance(node, Identifier):
            name = node.name
            if name in self.series:
                s = self.series[name]
                return float(s[bar_idx]) if isinstance(s, np.ndarray) else s
            if name in self.variables:
                v = self.variables[name]
                return float(v[bar_idx]) if isinstance(v, np.ndarray) else v
            if name in ('strategy.long',):
                return 'long'
            if name in ('strategy.short',):
                return 'short'
            return np.nan
        return 0

    def _eval_input(self, node: InputCall) -> Any:
        """Evaluate an input() call, using user overrides or defaults."""
        title = None
        defval = None
        for k, v in node.kwargs.items():
            if k == 'title' and isinstance(v, StringLiteral):
                title = v.value
            if k == 'defval':
                if isinstance(v, NumberLiteral):
                    defval = v.value
                elif isinstance(v, StringLiteral):
                    defval = v.value
                elif isinstance(v, BoolLiteral):
                    defval = v.value

        # Check user overrides
        if title and title in self.user_inputs:
            defval = self.user_inputs[title]

        if defval is not None:
            return np.full(self.n_bars, defval) if isinstance(defval, (int, float)) else defval
        return np.full(self.n_bars, 14.0)  # default

    # =========================================================================
    # Bar-by-bar execution
    # =========================================================================

    def _exec_node(self, node: ASTNode, bar: int):
        """Execute a node at bar index."""
        if isinstance(node, StrategyCall):
            self._exec_strategy_call(node, bar)
        elif isinstance(node, IfStatement):
            self._exec_if(node, bar)
        elif isinstance(node, Assignment):
            if node.is_var and node.name in self.var_variables:
                return  # Already initialized
            if node.is_var:
                val = self._eval_scalar(node.value, bar)
                self.var_variables[node.name] = val

    def _exec_if(self, node: IfStatement, bar: int):
        """Execute if/else at a specific bar."""
        cond = self._eval_condition_at_bar(node.condition, bar)
        if cond:
            for stmt in node.body:
                self._exec_node(stmt, bar)
        else:
            for stmt in node.else_body:
                self._exec_node(stmt, bar)

    def _eval_condition_at_bar(self, node: ASTNode, bar: int) -> bool:
        """Evaluate a boolean expression at a specific bar."""
        if isinstance(node, BoolLiteral):
            return node.value
        if isinstance(node, Identifier):
            name = node.name
            if name in self.series:
                val = self.series[name]
                if isinstance(val, np.ndarray):
                    return bool(val[bar]) if not np.isnan(val[bar]) else False
            return False
        if isinstance(node, FunctionCall):
            name = node.name
            if name in ('ta.crossover', 'crossover', 'ta.crossunder', 'crossunder'):
                series_result = self._eval_function_series(node)
                if isinstance(series_result, np.ndarray):
                    return bool(series_result[bar])
            # Other functions — evaluate at bar
            result = self._eval_series(node)
            if isinstance(result, np.ndarray):
                return bool(result[bar]) if not np.isnan(result[bar]) else False
            return bool(result)
        if isinstance(node, BinaryOp):
            left = self._get_value_at_bar(node.left, bar)
            right = self._get_value_at_bar(node.right, bar)
            if node.op == '>': return left > right
            if node.op == '<': return left < right
            if node.op == '>=': return left >= right
            if node.op == '<=': return left <= right
            if node.op == '==': return left == right
            if node.op == '!=': return left != right
            if node.op == 'and': return bool(left) and bool(right)
            if node.op == 'or': return bool(left) or bool(right)
        if isinstance(node, UnaryOp):
            if node.op == 'not':
                return not self._eval_condition_at_bar(node.operand, bar)
        return False

    def _get_value_at_bar(self, node: ASTNode, bar: int) -> float:
        if isinstance(node, NumberLiteral):
            return node.value
        if isinstance(node, Identifier):
            name = node.name
            if name in self.series:
                val = self.series[name]
                if isinstance(val, np.ndarray) and bar < len(val):
                    return float(val[bar]) if not np.isnan(val[bar]) else 0.0
            if name in self.variables:
                val = self.variables[name]
                if isinstance(val, np.ndarray) and bar < len(val):
                    return float(val[bar]) if not np.isnan(val[bar]) else 0.0
                return float(val) if val is not None else 0.0
            return 0.0
        if isinstance(node, FunctionCall):
            result = self._eval_series(node)
            if isinstance(result, np.ndarray) and bar < len(result):
                return float(result[bar]) if not np.isnan(result[bar]) else 0.0
            return 0.0
        if isinstance(node, BinaryOp):
            left = self._get_value_at_bar(node.left, bar)
            right = self._get_value_at_bar(node.right, bar)
            if node.op == '+': return left + right
            if node.op == '-': return left - right
            if node.op == '*': return left * right
            if node.op == '/': return left / right if right != 0 else 0.0
        return self._eval_scalar(node, bar)

    def _exec_strategy_call(self, node: StrategyCall, bar: int):
        """Execute strategy.entry / strategy.close / strategy.exit."""
        method = node.method
        price = float(self.series['close'][bar])
        date = str(self.df.index[bar])

        if method == 'entry':
            label = node.args[0].value if node.args and isinstance(node.args[0], StringLiteral) else "Trade"
            direction_node = node.args[1] if len(node.args) > 1 else node.kwargs.get('direction')
            direction = 'long'
            if direction_node:
                d = self._eval_scalar(direction_node, bar)
                if d == 'short' or (isinstance(d, str) and 'short' in d.lower()):
                    direction = 'short'

            new_pos = 1 if direction == 'long' else -1

            # Close existing if opposite
            if self.position != 0 and self.position != new_pos:
                pnl = (price - self.entry_price) * self.position
                if self.trades:
                    self.trades[-1]['exit_bar'] = bar
                    self.trades[-1]['exit_date'] = date
                    self.trades[-1]['exit_price'] = price
                    self.trades[-1]['pnl'] = round(pnl, 2)
                    self.trades[-1]['pnl_pct'] = round(pnl / self.entry_price * 100, 2) if self.entry_price else 0

            if self.position != new_pos:
                self.position = new_pos
                self.entry_price = price
                self.trades.append({
                    'entry_bar': bar,
                    'exit_bar': None,
                    'entry_date': date,
                    'exit_date': None,
                    'side': direction,
                    'entry_price': price,
                    'exit_price': None,
                    'pnl': None,
                    'pnl_pct': None,
                })

        elif method in ('close', 'close_all'):
            if self.position != 0:
                pnl = (price - self.entry_price) * self.position
                if self.trades and self.trades[-1]['exit_bar'] is None:
                    self.trades[-1]['exit_bar'] = bar
                    self.trades[-1]['exit_date'] = date
                    self.trades[-1]['exit_price'] = price
                    self.trades[-1]['pnl'] = round(pnl, 2)
                    self.trades[-1]['pnl_pct'] = round(pnl / self.entry_price * 100, 2) if self.entry_price else 0
                self.position = 0
                self.entry_price = 0.0

        elif method == 'exit':
            # Similar to close but for a specific entry
            if self.position != 0:
                pnl = (price - self.entry_price) * self.position
                if self.trades and self.trades[-1]['exit_bar'] is None:
                    self.trades[-1]['exit_bar'] = bar
                    self.trades[-1]['exit_date'] = date
                    self.trades[-1]['exit_price'] = price
                    self.trades[-1]['pnl'] = round(pnl, 2)
                    self.trades[-1]['pnl_pct'] = round(pnl / self.entry_price * 100, 2) if self.entry_price else 0
                self.position = 0
                self.entry_price = 0.0

    # =========================================================================
    # Results Builder
    # =========================================================================

    def _build_results(self) -> Dict[str, Any]:
        """Build comprehensive backtest results."""
        completed = [t for t in self.trades if t['pnl'] is not None]

        if not completed:
            return {
                'trades': [],
                'summary': {
                    'total_trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0,
                    'total_pnl': 0, 'avg_pnl': 0, 'best_trade': 0, 'worst_trade': 0,
                    'max_drawdown': 0, 'sharpe_ratio': 0, 'profit_factor': 0,
                },
                'equity_curve': [0.0],
                'dates': [str(self.df.index[0])] if len(self.df) > 0 else [],
            }

        pnls = [t['pnl'] for t in completed]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        # Equity curve
        equity = [0.0]
        for p in pnls:
            equity.append(round(equity[-1] + p, 2))

        # Max drawdown
        peak = 0.0
        max_dd = 0.0
        for e in equity:
            if e > peak:
                peak = e
            dd = peak - e
            if dd > max_dd:
                max_dd = dd

        # Sharpe ratio (annualized, assuming daily)
        pnl_arr = np.array(pnls)
        sharpe = 0.0
        if len(pnl_arr) > 1 and np.std(pnl_arr) > 0:
            sharpe = round(np.mean(pnl_arr) / np.std(pnl_arr) * np.sqrt(252), 2)

        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float('inf')

        return {
            'trades': completed,
            'summary': {
                'total_trades': len(completed),
                'wins': len(wins),
                'losses': len(losses),
                'win_rate': round(len(wins) / len(completed) * 100, 1),
                'total_pnl': round(sum(pnls), 2),
                'avg_pnl': round(np.mean(pnls), 2),
                'best_trade': round(max(pnls), 2),
                'worst_trade': round(min(pnls), 2),
                'max_drawdown': round(max_dd, 2),
                'sharpe_ratio': sharpe,
                'profit_factor': profit_factor,
            },
            'equity_curve': equity,
            'dates': [str(self.df.index[0])] + [t.get('exit_date', '') for t in completed],
        }


def run_pine_backtest(
    pine_script: str,
    df: pd.DataFrame,
    user_inputs: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Parse and execute a Pine Script strategy against OHLCV data.

    Args:
        pine_script: The Pine Script source code.
        df: DataFrame with open, high, low, close, volume columns.
        user_inputs: Optional overrides for input() values.

    Returns:
        Backtest results dict with trades, summary, equity_curve.
    """
    ast, config = parse_pine_script(pine_script)
    executor = PineExecutor(df, user_inputs)
    results = executor.execute(ast)
    results['strategy_config'] = config
    return results
