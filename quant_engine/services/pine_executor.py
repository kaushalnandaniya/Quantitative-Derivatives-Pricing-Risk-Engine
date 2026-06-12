"""
Pine Script Executor
======================
Executes a parsed Pine Script AST against OHLCV data to generate
trade signals, compute P&L, and produce backtest results.

Supports: input.int/float with positional defval, strategy.opentrades,
plot() (ignored), color.* constants, multi-pass series precomputation,
complex boolean conditions with and/or/not.
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
    InputCall, parse_pine_script, TupleAssignment, FunctionDef, ForLoop,
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


def _ta_bb(series: np.ndarray, length: int = 20, mult: float = 2.0):
    basis = _ta_sma(series, length)
    dev = _ta_stdev(series, length)
    upper = basis + mult * dev
    lower = basis - mult * dev
    return basis, upper, lower


# =============================================================================
# Executor
# =============================================================================

# Functions to skip (they don't affect strategy logic)
SKIP_FUNCTIONS = {
    'plot', 'plotshape', 'plotchar', 'plotarrow', 'plotcandle',
    'bgcolor', 'barcolor', 'fill', 'hline', 'label.new', 'line.new',
    'table.new', 'table.cell', 'alert', 'alertcondition', 'log.info',
}

# Strategy built-in constants
STRATEGY_CONSTANTS = {
    'strategy.long': 'long',
    'strategy.short': 'short',
    'strategy.percent_of_equity': 'percent_of_equity',
    'strategy.fixed': 'fixed',
    'strategy.cash': 'cash',
}

# Color constants (ignored but shouldn't error)
COLOR_CONSTANTS = {
    'color.blue', 'color.red', 'color.green', 'color.yellow', 'color.white',
    'color.black', 'color.orange', 'color.purple', 'color.aqua', 'color.gray',
    'color.silver', 'color.lime', 'color.fuchsia', 'color.maroon', 'color.navy',
    'color.olive', 'color.teal', 'color.new',
}


class PineExecutor:
    """Execute Pine Script AST bar-by-bar against OHLCV data."""

    def __init__(self, df: pd.DataFrame, user_inputs: Dict[str, Any] = None):
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

        self.variables: Dict[str, Any] = {}
        self.var_variables: Dict[str, Any] = {}
        self.custom_functions: Dict[str, FunctionDef] = {}

        # Strategy state
        self.position = 0  # 1=long, -1=short, 0=flat
        self.entry_price = 0.0
        self.open_trades_count = 0  # Track strategy.opentrades
        self.trades: List[Dict] = []
        self.signals: List[Dict] = []
        self.bar_idx = 0

    def execute(self, ast: List[ASTNode]) -> Dict[str, Any]:
        """Execute the full AST and return backtest results."""
        # Register custom functions
        for node in ast:
            if isinstance(node, FunctionDef):
                self.custom_functions[node.name] = node

        # Multi-pass precomputation: keep trying until no new series are resolved
        self._precompute_series_multipass(ast)

        # Bar-by-bar execution for strategy logic
        for i in range(self.n_bars):
            self.bar_idx = i
            for node in ast:
                self._exec_node(node, i)

        # Close any open position at the end
        if self.position != 0:
            exit_price = float(self.series['close'][-1])
            pnl = (exit_price - self.entry_price) * self.position
            if self.trades and self.trades[-1]['exit_bar'] is None:
                self.trades[-1]['exit_bar'] = self.n_bars - 1
                self.trades[-1]['exit_date'] = str(self.df.index[-1])
                self.trades[-1]['exit_price'] = exit_price
                self.trades[-1]['pnl'] = round(pnl, 2)
                self.trades[-1]['pnl_pct'] = round(pnl / self.entry_price * 100, 2) if self.entry_price else 0
            self.position = 0

        return self._build_results()

    def _precompute_series_multipass(self, ast: List[ASTNode]):
        """Multi-pass precomputation to resolve dependency chains."""
        max_passes = 10
        for pass_num in range(max_passes):
            resolved_any = False
            for node in ast:
                if isinstance(node, Assignment) and not isinstance(node.value, (StrategyCall, IfStatement)):
                    if node.name in self.series:
                        continue  # Already computed
                    try:
                        val = self._eval_series(node.value)
                        if isinstance(val, np.ndarray) and not np.all(np.isnan(val)):
                            self.series[node.name] = val
                            self.variables[node.name] = val
                            resolved_any = True
                        elif isinstance(val, (int, float)):
                            arr = np.full(self.n_bars, float(val))
                            self.series[node.name] = arr
                            self.variables[node.name] = arr
                            resolved_any = True
                    except Exception:
                        pass  # Might resolve in next pass
                elif isinstance(node, TupleAssignment):
                    if all(n in self.series for n in node.names):
                        continue
                    try:
                        val = self._eval_series(node.value)
                        if isinstance(val, tuple):
                            for n, v in zip(node.names, val):
                                if isinstance(v, np.ndarray) and not np.all(np.isnan(v)):
                                    self.series[n] = v
                                    self.variables[n] = v
                            resolved_any = True
                    except Exception:
                        pass
            if not resolved_any:
                break

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
            if name in STRATEGY_CONSTANTS:
                return STRATEGY_CONSTANTS[name]
            if name in COLOR_CONSTANTS:
                return 0  # Color constants are meaningless for backtest
            # strategy.opentrades — can't precompute, return NaN
            if name == 'strategy.opentrades':
                return np.full(self.n_bars, np.nan)
            return np.full(self.n_bars, np.nan)

        if isinstance(node, FunctionCall):
            # Skip plot/visual functions
            if node.name in SKIP_FUNCTIONS:
                return np.full(self.n_bars, 0.0)
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
                if isinstance(operand, np.ndarray):
                    return ~operand.astype(bool)
                return not operand

        if isinstance(node, HistoryRef):
            series = self._eval_series(node.series)
            offset = self._eval_scalar(node.offset, 0)
            if isinstance(series, np.ndarray) and isinstance(offset, (int, float)):
                shifted = np.roll(series, int(offset))
                shifted[:int(offset)] = np.nan  # Don't look into the future
                return shifted

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
        """Evaluate a function call over the entire series."""
        name = node.name
        args = [self._eval_series(a) for a in node.args]
        kwargs = {k: self._eval_series(v) for k, v in node.kwargs.items()}

        # Custom Functions
        if name in self.custom_functions:
            func_def = self.custom_functions[name]
            result_arr = np.full(self.n_bars, np.nan)
            
            for bar in range(self.n_bars):
                # Backup variables for scope
                old_vars = self.variables.copy()
                
                # Bind parameters
                for p_name, arg_val in zip(func_def.params, args):
                    if isinstance(arg_val, np.ndarray):
                        self.variables[p_name] = arg_val[bar]
                    else:
                        self.variables[p_name] = arg_val
                        
                # Execute all but last statement
                for stmt in func_def.body[:-1]:
                    self._exec_node(stmt, bar)
                    
                # Last statement is the return value
                last_stmt = func_def.body[-1]
                try:
                    res = self._eval_scalar(last_stmt, bar)
                    result_arr[bar] = res
                except Exception:
                    pass
                    
                # Restore scope
                self.variables = old_vars
                
            return result_arr

        # Skip visual functions
        if name in SKIP_FUNCTIONS:
            return np.full(self.n_bars, 0.0)

        # ta.* functions
        if name in ('ta.sma', 'sma'):
            src = args[0] if args else kwargs.get('source', self.series['close'])
            length = int(self._to_scalar(args[1] if len(args) > 1 else kwargs.get('length', 14)))
            if isinstance(src, np.ndarray):
                return _ta_sma(src, length)
            return np.full(self.n_bars, np.nan)

        if name in ('ta.ema', 'ema'):
            src = args[0] if args else kwargs.get('source', self.series['close'])
            length = int(self._to_scalar(args[1] if len(args) > 1 else kwargs.get('length', 14)))
            if isinstance(src, np.ndarray):
                return _ta_ema(src, length)
            return np.full(self.n_bars, np.nan)

        if name in ('ta.rsi', 'rsi'):
            src = args[0] if args else kwargs.get('source', self.series['close'])
            length = int(self._to_scalar(args[1] if len(args) > 1 else kwargs.get('length', 14)))
            if isinstance(src, np.ndarray):
                return _ta_rsi(src, length)
            return np.full(self.n_bars, np.nan)

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
            if len(args) >= 2 and isinstance(args[0], np.ndarray) and isinstance(args[1], np.ndarray):
                return _ta_crossover(args[0], args[1])
            return np.zeros(self.n_bars, dtype=bool)

        if name in ('ta.crossunder', 'crossunder'):
            if len(args) >= 2 and isinstance(args[0], np.ndarray) and isinstance(args[1], np.ndarray):
                return _ta_crossunder(args[0], args[1])
            return np.zeros(self.n_bars, dtype=bool)

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
            return (macd_line, signal_line, hist)

        if name in ('ta.bb', 'bb'):
            src = args[0] if args else self.series['close']
            length = int(self._to_scalar(args[1] if len(args) > 1 else 20))
            mult = float(self._to_scalar(args[2] if len(args) > 2 else 2.0))
            basis, upper, lower = _ta_bb(src, length, mult)
            return (basis, upper, lower)

        if name in ('math.abs', 'abs'):
            return np.abs(args[0]) if args else np.full(self.n_bars, 0.0)
        if name in ('math.max', 'max'):
            return np.maximum(args[0], args[1]) if len(args) >= 2 else args[0]
        if name in ('math.min', 'min'):
            return np.minimum(args[0], args[1]) if len(args) >= 2 else args[0]
        if name in ('math.sqrt', 'sqrt'):
            return np.sqrt(args[0]) if args else np.full(self.n_bars, 0.0)
        if name in ('math.log', 'log'):
            return np.log(args[0]) if args else np.full(self.n_bars, 0.0)

        if name in ('nz',):
            val = args[0]
            replacement = args[1] if len(args) > 1 else np.zeros_like(val) if isinstance(val, np.ndarray) else 0
            if isinstance(val, np.ndarray):
                return np.where(np.isnan(val), replacement, val)
            return val

        # input.* functions handled as FunctionCall (when parsed without InputCall node)
        if name.startswith('input'):
            return self._eval_input_from_func(node)

        return np.full(self.n_bars, np.nan)

    def _binary_op_series(self, op: str, left: Any, right: Any) -> Any:
        # Ensure both operands are numeric arrays for comparison
        left = self._ensure_array(left)
        right = self._ensure_array(right)

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
        if op == 'and':
            return np.logical_and(self._to_bool_array(left), self._to_bool_array(right))
        if op == 'or':
            return np.logical_or(self._to_bool_array(left), self._to_bool_array(right))
        return np.full(self.n_bars, np.nan)

    def _ensure_array(self, val: Any) -> np.ndarray:
        if isinstance(val, np.ndarray):
            return val
        if isinstance(val, (int, float)):
            return np.full(self.n_bars, float(val))
        if isinstance(val, bool):
            return np.full(self.n_bars, val)
        return np.full(self.n_bars, 0.0)

    def _to_bool_array(self, val: Any) -> np.ndarray:
        if isinstance(val, np.ndarray):
            if val.dtype == bool:
                return val
            return ~np.isnan(val) & (val != 0)
        return np.full(self.n_bars, bool(val))

    def _to_scalar(self, val) -> float:
        if isinstance(val, np.ndarray):
            # Return first non-NaN value
            for v in val:
                if not np.isnan(v):
                    return float(v)
            return 0.0
        return float(val) if val is not None else 0.0

    def _eval_scalar(self, node: ASTNode, bar_idx: int) -> Any:
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
            if name in STRATEGY_CONSTANTS:
                return STRATEGY_CONSTANTS[name]
            if name == 'strategy.opentrades':
                return self.open_trades_count
            return np.nan
        return 0

    def _eval_input(self, node: InputCall) -> Any:
        """Evaluate an input() call, using user overrides or defaults."""
        title = None
        defval = None

        # Check kwargs
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
        return np.full(self.n_bars, 14.0)

    def _eval_input_from_func(self, node: FunctionCall) -> Any:
        """Handle input.int/input.float parsed as FunctionCall."""
        defval = None
        title = None

        # First positional arg is the default value
        if node.args:
            first = node.args[0]
            if isinstance(first, NumberLiteral):
                defval = first.value
            elif isinstance(first, StringLiteral):
                defval = first.value
            elif isinstance(first, BoolLiteral):
                defval = first.value

        # Check kwargs
        for k, v in node.kwargs.items():
            if k == 'title' and isinstance(v, StringLiteral):
                title = v.value
            if k == 'defval':
                if isinstance(v, NumberLiteral):
                    defval = v.value

        # Check user overrides
        if title and title in self.user_inputs:
            defval = self.user_inputs[title]

        if defval is not None:
            return np.full(self.n_bars, float(defval)) if isinstance(defval, (int, float)) else defval
        return np.full(self.n_bars, 14.0)

    # =========================================================================
    # Bar-by-bar execution
    # =========================================================================

    def _exec_node(self, node: ASTNode, bar: int):
        if isinstance(node, StrategyCall):
            self._exec_strategy_call(node, bar)
        elif isinstance(node, IfStatement):
            self._exec_if(node, bar)
        elif isinstance(node, Assignment):
            if node.is_var and node.name in self.var_variables:
                return
            if node.is_var:
                val = self._eval_scalar(node.value, bar)
                self.var_variables[node.name] = val
        elif isinstance(node, FunctionCall):
            pass
        elif isinstance(node, TupleAssignment):
            val = self._eval_scalar(node.value, bar)
            if isinstance(val, tuple):
                for n, v in zip(node.names, val):
                    self.variables[n] = v
        elif isinstance(node, ForLoop):
            self._exec_for_loop(node, bar)

    def _exec_for_loop(self, node: ForLoop, bar: int):
        start_val = int(self._eval_scalar(node.start, bar))
        end_val = int(self._eval_scalar(node.end, bar))
        step_val = int(self._eval_scalar(node.step, bar))
        if step_val == 0:
            step_val = 1
        
        # Determine direction
        if start_val <= end_val and step_val > 0:
            for_range = range(start_val, end_val + 1, step_val)
        elif start_val >= end_val and step_val < 0:
            for_range = range(start_val, end_val - 1, step_val)
        else:
            for_range = []

        for i in for_range:
            self.variables[node.var_name] = float(i)
            for stmt in node.body:
                self._exec_node(stmt, bar)

    def _exec_if(self, node: IfStatement, bar: int):
        cond = self._eval_condition_at_bar(node.condition, bar)
        if cond:
            for stmt in node.body:
                self._exec_node(stmt, bar)
        else:
            for stmt in node.else_body:
                self._exec_node(stmt, bar)

    def _eval_condition_at_bar(self, node: ASTNode, bar: int) -> bool:
        """Evaluate a boolean expression at a specific bar. Handles nested and/or."""
        if isinstance(node, BoolLiteral):
            return node.value

        if isinstance(node, Identifier):
            name = node.name
            if name == 'strategy.opentrades':
                return bool(self.open_trades_count)
            if name in self.series:
                val = self.series[name]
                if isinstance(val, np.ndarray) and bar < len(val):
                    v = val[bar]
                    if isinstance(v, (bool, np.bool_)):
                        return bool(v)
                    if np.isnan(v):
                        return False
                    return bool(v)
            if name in self.variables:
                val = self.variables[name]
                if isinstance(val, np.ndarray) and bar < len(val):
                    v = val[bar]
                    if isinstance(v, (bool, np.bool_)):
                        return bool(v)
                    if np.isnan(v):
                        return False
                    return bool(v)
            return False

        if isinstance(node, FunctionCall):
            name = node.name
            if name in ('ta.crossover', 'crossover', 'ta.crossunder', 'crossunder'):
                series_result = self._eval_function_series(node)
                if isinstance(series_result, np.ndarray) and bar < len(series_result):
                    return bool(series_result[bar])
            result = self._eval_series(node)
            if isinstance(result, np.ndarray) and bar < len(result):
                v = result[bar]
                return bool(v) if not (isinstance(v, float) and np.isnan(v)) else False
            return bool(result) if result is not None else False

        if isinstance(node, BinaryOp):
            # For and/or, recursively evaluate as booleans
            if node.op in ('and', 'or'):
                left_bool = self._eval_condition_at_bar(node.left, bar)
                right_bool = self._eval_condition_at_bar(node.right, bar)
                if node.op == 'and':
                    return left_bool and right_bool
                return left_bool or right_bool

            # For comparison ops, get numeric values
            left = self._get_value_at_bar(node.left, bar)
            right = self._get_value_at_bar(node.right, bar)
            try:
                if node.op == '>': return float(left) > float(right)
                if node.op == '<': return float(left) < float(right)
                if node.op == '>=': return float(left) >= float(right)
                if node.op == '<=': return float(left) <= float(right)
                if node.op == '==': return float(left) == float(right)
                if node.op == '!=': return float(left) != float(right)
            except (TypeError, ValueError):
                return False

        if isinstance(node, UnaryOp):
            if node.op == 'not':
                return not self._eval_condition_at_bar(node.operand, bar)

        return False

    def _get_value_at_bar(self, node: ASTNode, bar: int) -> float:
        if isinstance(node, NumberLiteral):
            return node.value
        if isinstance(node, Identifier):
            name = node.name
            if name == 'strategy.opentrades':
                return float(self.open_trades_count)
            if name in self.series:
                val = self.series[name]
                if isinstance(val, np.ndarray) and bar < len(val):
                    v = float(val[bar])
                    return v if not np.isnan(v) else 0.0
            if name in self.variables:
                val = self.variables[name]
                if isinstance(val, np.ndarray) and bar < len(val):
                    v = float(val[bar])
                    return v if not np.isnan(v) else 0.0
                if isinstance(val, (int, float)):
                    return float(val)
            return 0.0
        if isinstance(node, FunctionCall):
            if node.name in SKIP_FUNCTIONS:
                return 0.0
            result = self._eval_series(node)
            if isinstance(result, np.ndarray) and bar < len(result):
                v = float(result[bar])
                return v if not np.isnan(v) else 0.0
            return 0.0
        if isinstance(node, BinaryOp):
            left = self._get_value_at_bar(node.left, bar)
            right = self._get_value_at_bar(node.right, bar)
            if node.op == '+': return left + right
            if node.op == '-': return left - right
            if node.op == '*': return left * right
            if node.op == '/': return left / right if right != 0 else 0.0
            if node.op == '%': return left % right if right != 0 else 0.0
            # Boolean ops return 0/1
            if node.op == '>': return 1.0 if left > right else 0.0
            if node.op == '<': return 1.0 if left < right else 0.0
            if node.op == '>=': return 1.0 if left >= right else 0.0
            if node.op == '<=': return 1.0 if left <= right else 0.0
            if node.op == '==': return 1.0 if left == right else 0.0
            if node.op == '!=': return 1.0 if left != right else 0.0
            if node.op == 'and': return 1.0 if (left and right) else 0.0
            if node.op == 'or': return 1.0 if (left or right) else 0.0
        if isinstance(node, UnaryOp):
            if node.op == '-':
                return -self._get_value_at_bar(node.operand, bar)
            if node.op == 'not':
                return 0.0 if self._get_value_at_bar(node.operand, bar) else 1.0
        return self._eval_scalar(node, bar) or 0.0

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

            # Close existing if opposite direction
            if self.position != 0 and self.position != new_pos:
                pnl = (price - self.entry_price) * self.position
                if self.trades and self.trades[-1]['exit_bar'] is None:
                    self.trades[-1]['exit_bar'] = bar
                    self.trades[-1]['exit_date'] = date
                    self.trades[-1]['exit_price'] = price
                    self.trades[-1]['pnl'] = round(pnl, 2)
                    self.trades[-1]['pnl_pct'] = round(pnl / self.entry_price * 100, 2) if self.entry_price else 0
                self.open_trades_count = max(0, self.open_trades_count - 1)

            if self.position != new_pos:
                self.position = new_pos
                self.entry_price = price
                self.open_trades_count += 1
                self.trades.append({
                    'entry_bar': bar, 'exit_bar': None,
                    'entry_date': date, 'exit_date': None,
                    'side': direction, 'entry_price': price,
                    'exit_price': None, 'pnl': None, 'pnl_pct': None,
                })

        elif method in ('close', 'close_all'):
            # Optionally close only a specific label
            target_label = None
            if node.args and isinstance(node.args[0], StringLiteral):
                target_label = node.args[0].value

            if self.position != 0:
                if target_label:
                    lbl = target_label.lower()
                    current_side = 'long' if self.position > 0 else 'short'
                    # If label specifies a direction, skip if it doesn't match current position
                    if (lbl == 'long' and current_side == 'short') or (lbl == 'short' and current_side == 'long'):
                        return

                pnl = (price - self.entry_price) * self.position
                if self.trades and self.trades[-1]['exit_bar'] is None:
                    self.trades[-1]['exit_bar'] = bar
                    self.trades[-1]['exit_date'] = date
                    self.trades[-1]['exit_price'] = price
                    self.trades[-1]['pnl'] = round(pnl, 2)
                    self.trades[-1]['pnl_pct'] = round(pnl / self.entry_price * 100, 2) if self.entry_price else 0
                self.position = 0
                self.entry_price = 0.0
                self.open_trades_count = max(0, self.open_trades_count - 1)

        elif method == 'exit':
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
                self.open_trades_count = max(0, self.open_trades_count - 1)

    # =========================================================================
    # Results Builder
    # =========================================================================

    def _build_results(self) -> Dict[str, Any]:
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

        equity = [0.0]
        for p in pnls:
            equity.append(round(equity[-1] + p, 2))

        peak = 0.0
        max_dd = 0.0
        for e in equity:
            if e > peak:
                peak = e
            dd = peak - e
            if dd > max_dd:
                max_dd = dd

        pnl_arr = np.array(pnls)
        sharpe = 0.0
        if len(pnl_arr) > 1 and np.std(pnl_arr) > 0:
            sharpe = round(np.mean(pnl_arr) / np.std(pnl_arr) * np.sqrt(252), 2)

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 9999.99

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
    """
    ast, config = parse_pine_script(pine_script)
    executor = PineExecutor(df, user_inputs)
    results = executor.execute(ast)
    results['strategy_config'] = config
    return results
