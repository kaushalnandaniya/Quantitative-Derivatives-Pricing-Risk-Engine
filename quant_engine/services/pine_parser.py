"""
Pine Script Parser & Transpiler
=================================
Comprehensive Pine Script v5 parser that lexes, parses, and transpiles
Pine Script strategies into executable Python logic.

Supports:
    - Variable declarations (var, varip, type inference)
    - Indicator functions: ta.sma, ta.ema, ta.rsi, ta.macd, ta.atr, ta.stdev,
      ta.highest, ta.lowest, ta.crossover, ta.crossunder, ta.change
    - Strategy calls: strategy.entry, strategy.close, strategy.exit
    - Conditionals: if/else blocks
    - Comparison operators: >, <, >=, <=, ==, !=, and, or, not
    - Math: +, -, *, /, %
    - Built-in variables: open, high, low, close, volume, bar_index
    - Input functions: input.int, input.float, input.string
    - Plot functions (ignored for backtesting)
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum, auto

logger = logging.getLogger(__name__)


# =============================================================================
# Token Types
# =============================================================================

class TokenType(Enum):
    # Literals
    NUMBER = auto()
    STRING = auto()
    IDENTIFIER = auto()
    BOOL = auto()

    # Operators
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    ASSIGN = auto()
    EQ = auto()
    NEQ = auto()
    LT = auto()
    GT = auto()
    LTE = auto()
    GTE = auto()
    QUESTION = auto()
    COLON = auto()

    # Logic
    AND = auto()
    OR = auto()
    NOT = auto()

    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COMMA = auto()
    DOT = auto()
    NEWLINE = auto()

    # Keywords
    IF = auto()
    ELSE = auto()
    FOR = auto()
    WHILE = auto()
    VAR = auto()
    VARIP = auto()
    TRUE = auto()
    FALSE = auto()
    NA = auto()
    STRATEGY = auto()
    TO = auto()
    BY = auto()

    # Special
    WALRUS = auto()  # :=
    ARROW = auto()   # =>
    EOF = auto()
    COMMENT = auto()
    INDENT = auto()


# =============================================================================
# Lexer
# =============================================================================

KEYWORDS = {
    'if': TokenType.IF, 'else': TokenType.ELSE, 'for': TokenType.FOR,
    'while': TokenType.WHILE, 'var': TokenType.VAR, 'varip': TokenType.VARIP,
    'true': TokenType.TRUE, 'false': TokenType.FALSE, 'na': TokenType.NA,
    'and': TokenType.AND, 'or': TokenType.OR, 'not': TokenType.NOT,
    'strategy': TokenType.STRATEGY, 'to': TokenType.TO, 'by': TokenType.BY,
}


class Token:
    __slots__ = ('type', 'value', 'line', 'indent')

    def __init__(self, type_: TokenType, value: Any, line: int = 1, indent: int = 0):
        self.type = type_
        self.value = value
        self.line = line
        self.indent = indent

    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r})"


def tokenize(source: str) -> List[Token]:
    """Tokenize Pine Script source code."""
    tokens = []
    lines = source.split('\n')

    for line_num, line in enumerate(lines, 1):
        # Skip empty lines and comments
        stripped = line.strip()
        if not stripped or stripped.startswith('//'):
            tokens.append(Token(TokenType.NEWLINE, '\n', line_num, 0))
            continue
            
        line_indent = len(line) - len(line.lstrip())

        # Remove inline comments
        comment_idx = line.find('//')
        if comment_idx >= 0:
            line = line[:comment_idx]

        i = 0
        while i < len(line):
            c = line[i]

            # Whitespace
            if c in (' ', '\t'):
                i += 1
                continue

            # Numbers
            if c.isdigit() or (c == '.' and i + 1 < len(line) and line[i + 1].isdigit()):
                j = i
                has_dot = False
                while j < len(line) and (line[j].isdigit() or line[j] == '.'):
                    if line[j] == '.':
                        if has_dot:
                            break
                        has_dot = True
                    j += 1
                tokens.append(Token(TokenType.NUMBER, float(line[i:j]), line_num, line_indent))
                i = j
                continue

            # Strings
            if c in ('"', "'"):
                j = i + 1
                while j < len(line) and line[j] != c:
                    j += 1
                tokens.append(Token(TokenType.STRING, line[i + 1:j], line_num, line_indent))
                i = j + 1
                continue

            # Identifiers and keywords
            if c.isalpha() or c == '_':
                j = i
                while j < len(line) and (line[j].isalnum() or line[j] == '_'):
                    j += 1
                word = line[i:j]
                if word in KEYWORDS:
                    tokens.append(Token(KEYWORDS[word], word, line_num, line_indent))
                elif word in ('true', 'false'):
                    tokens.append(Token(TokenType.BOOL, word == 'true', line_num, line_indent))
                else:
                    tokens.append(Token(TokenType.IDENTIFIER, word, line_num, line_indent))
                i = j
                continue

            # Two-character operators
            if i + 1 < len(line):
                two = line[i:i + 2]
                if two == ':=':
                    tokens.append(Token(TokenType.WALRUS, ':=', line_num))
                    i += 2
                    continue
                if two == '=>':
                    tokens.append(Token(TokenType.ARROW, '=>', line_num))
                    i += 2
                    continue
                if two == '==':
                    tokens.append(Token(TokenType.EQ, '==', line_num))
                    i += 2
                    continue
                if two == '!=':
                    tokens.append(Token(TokenType.NEQ, '!=', line_num))
                    i += 2
                    continue
                if two == '<=':
                    tokens.append(Token(TokenType.LTE, '<=', line_num))
                    i += 2
                    continue
                if two == '>=':
                    tokens.append(Token(TokenType.GTE, '>=', line_num))
                    i += 2
                    continue

            # Single-character operators
            op_map = {
                '+': TokenType.PLUS, '-': TokenType.MINUS, '*': TokenType.STAR,
                '/': TokenType.SLASH, '%': TokenType.PERCENT, '=': TokenType.ASSIGN,
                '<': TokenType.LT, '>': TokenType.GT,
                '(': TokenType.LPAREN, ')': TokenType.RPAREN,
                '[': TokenType.LBRACKET, ']': TokenType.RBRACKET,
                ',': TokenType.COMMA, '.': TokenType.DOT,
                '?': TokenType.QUESTION, ':': TokenType.COLON,
            }

            if c in op_map:
                tokens.append(Token(op_map[c], c, line_num))
                i += 1
                continue

            i += 1

        tokens.append(Token(TokenType.NEWLINE, '\n', line_num))

    tokens.append(Token(TokenType.EOF, None, len(lines)))
    return tokens


# =============================================================================
# AST Nodes
# =============================================================================

class ASTNode:
    pass

class NumberLiteral(ASTNode):
    def __init__(self, value: float):
        self.value = value

class StringLiteral(ASTNode):
    def __init__(self, value: str):
        self.value = value

class BoolLiteral(ASTNode):
    def __init__(self, value: bool):
        self.value = value

class NALiteral(ASTNode):
    pass

class Identifier(ASTNode):
    def __init__(self, name: str):
        self.name = name

class BinaryOp(ASTNode):
    def __init__(self, op: str, left: ASTNode, right: ASTNode):
        self.op = op
        self.left = left
        self.right = right

class UnaryOp(ASTNode):
    def __init__(self, op: str, operand: ASTNode):
        self.op = op
        self.operand = operand

class FunctionCall(ASTNode):
    def __init__(self, name: str, args: List[ASTNode], kwargs: Dict[str, ASTNode] = None):
        self.name = name
        self.args = args
        self.kwargs = kwargs or {}

class MemberAccess(ASTNode):
    def __init__(self, obj: str, member: str):
        self.obj = obj
        self.member = member

class HistoryRef(ASTNode):
    """series[offset] — historical reference"""
    def __init__(self, series: ASTNode, offset: ASTNode):
        self.series = series
        self.offset = offset

class Assignment(ASTNode):
    def __init__(self, name: str, value: ASTNode, is_var: bool = False, is_reassign: bool = False):
        self.name = name
        self.value = value
        self.is_var = is_var
        self.is_reassign = is_reassign

class IfStatement(ASTNode):
    def __init__(self, condition: ASTNode, body: List[ASTNode], else_body: List[ASTNode] = None):
        self.condition = condition
        self.body = body
        self.else_body = else_body or []

class StrategyCall(ASTNode):
    def __init__(self, method: str, args: List[ASTNode], kwargs: Dict[str, ASTNode] = None):
        self.method = method
        self.args = args
        self.kwargs = kwargs or {}

class TernaryOp(ASTNode):
    def __init__(self, condition: ASTNode, true_val: ASTNode, false_val: ASTNode):
        self.condition = condition
        self.true_val = true_val
        self.false_val = false_val

class InputCall(ASTNode):
    def __init__(self, input_type: str, kwargs: Dict[str, ASTNode], args: List[ASTNode] = None):
        self.input_type = input_type
        self.kwargs = kwargs
        self.args = args or []

class TupleAssignment(ASTNode):
    def __init__(self, names: List[str], value: ASTNode, is_var: bool = False):
        self.names = names
        self.value = value
        self.is_var = is_var

class FunctionDef(ASTNode):
    def __init__(self, name: str, params: List[str], body: List[ASTNode]):
        self.name = name
        self.params = params
        self.body = body

class ForLoop(ASTNode):
    def __init__(self, var_name: str, start: ASTNode, end: ASTNode, step: ASTNode, body: List[ASTNode]):
        self.var_name = var_name
        self.start = start
        self.end = end
        self.step = step
        self.body = body


# =============================================================================
# Parser
# =============================================================================

class PineParser:
    """Recursive descent parser for Pine Script."""

    def __init__(self, tokens: List[Token]):
        self.tokens = [t for t in tokens if t.type != TokenType.COMMENT]
        self.pos = 0
        self.inputs: Dict[str, Any] = {}
        self.strategy_config: Dict[str, Any] = {}

    def peek(self) -> Token:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else Token(TokenType.EOF, None)

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, type_: TokenType) -> Token:
        tok = self.advance()
        if tok.type != type_:
            raise SyntaxError(f"Expected {type_.name}, got {tok.type.name} ({tok.value!r}) at line {tok.line}")
        return tok

    def skip_newlines(self):
        while self.pos < len(self.tokens) and self.peek().type == TokenType.NEWLINE:
            self.advance()

    def parse(self) -> List[ASTNode]:
        """Parse the full script and return a list of AST nodes."""
        statements = []
        self.skip_newlines()

        while self.peek().type != TokenType.EOF:
            # Skip the //@version and strategy() header declarations
            if self.peek().type == TokenType.SLASH:
                self._skip_to_newline()
                self.skip_newlines()
                continue

            stmt = self._parse_statement()
            if stmt is not None:
                statements.append(stmt)
            self.skip_newlines()

        return statements

    def _skip_to_newline(self):
        while self.pos < len(self.tokens) and self.peek().type != TokenType.NEWLINE:
            self.advance()

    def _parse_statement(self) -> Optional[ASTNode]:
        tok = self.peek()

        # strategy(...)  or  strategy.entry(...) etc.
        if tok.type == TokenType.STRATEGY:
            return self._parse_strategy_statement()

        # var declarations
        if tok.type in (TokenType.VAR, TokenType.VARIP):
            return self._parse_var_declaration()

        # if statement
        if tok.type == TokenType.IF:
            return self._parse_if()

        # for loop
        if tok.type == TokenType.FOR:
            return self._parse_for_loop()

        # Tuple assignment
        if tok.type == TokenType.LBRACKET:
            return self._parse_tuple_assignment()

        # identifier = ... or identifier := ... or function call
        if tok.type == TokenType.IDENTIFIER:
            return self._parse_identifier_statement()

        # Skip anything else (plot, bgcolor, etc.)
        self._skip_to_newline()
        return None

    def _parse_strategy_statement(self) -> Optional[ASTNode]:
        self.advance()  # consume 'strategy'

        if self.peek().type == TokenType.DOT:
            self.advance()  # consume '.'
            method = self.advance().value  # entry, close, exit, etc.

            if self.peek().type == TokenType.LPAREN:
                args, kwargs = self._parse_call_args()
                return StrategyCall(method, args, kwargs)
        elif self.peek().type == TokenType.LPAREN:
            # strategy() header — extract config
            args, kwargs = self._parse_call_args()
            for k, v in kwargs.items():
                if isinstance(v, (StringLiteral, NumberLiteral, BoolLiteral)):
                    self.strategy_config[k] = v.value
            if args and isinstance(args[0], StringLiteral):
                self.strategy_config['title'] = args[0].value
            return None

        self._skip_to_newline()
        return None

    def _parse_tuple_assignment(self) -> ASTNode:
        self.advance()  # consume '['
        names = []
        while self.peek().type != TokenType.RBRACKET and self.peek().type != TokenType.EOF:
            self.skip_newlines()
            if self.peek().type == TokenType.IDENTIFIER:
                names.append(self.advance().value)
            if self.peek().type == TokenType.COMMA:
                self.advance()
        self.expect(TokenType.RBRACKET)
        self.expect(TokenType.ASSIGN)
        value = self._parse_expression()
        return TupleAssignment(names, value)

    def _parse_for_loop(self) -> ASTNode:
        self.advance()  # consume 'for'
        var_name = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.ASSIGN)
        start = self._parse_expression()
        self.expect(TokenType.TO)
        end = self._parse_expression()
        step = NumberLiteral(1)
        if self.peek().type == TokenType.BY:
            self.advance()
            step = self._parse_expression()
        
        base_indent = self.tokens[self.pos - 1].indent if self.pos > 0 else 0
        body = self._parse_block(base_indent)
        return ForLoop(var_name, start, end, step, body)

    def _parse_var_declaration(self) -> ASTNode:
        self.advance()  # consume var/varip

        # Optional type annotation (int, float, bool, string)
        if self.peek().type == TokenType.IDENTIFIER and self.peek().value in ('int', 'float', 'bool', 'string', 'color', 'series'):
            self.advance()

        name = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.ASSIGN)
        value = self._parse_expression()
        return Assignment(name, value, is_var=True)

    def _parse_if(self) -> ASTNode:
        base_indent = self.peek().indent
        self.advance()  # consume 'if'
        condition = self._parse_expression()
        self.skip_newlines()

        body = self._parse_block(base_indent)
        else_body = []

        self.skip_newlines()
        if self.peek().type == TokenType.ELSE:
            self.advance()
            self.skip_newlines()
            if self.peek().type == TokenType.IF:
                else_body = [self._parse_if()]
            else:
                else_body = self._parse_block(base_indent)

        return IfStatement(condition, body, else_body)

    def _parse_block(self, base_indent: int = 0) -> List[ASTNode]:
        """Parse an indented block of statements."""
        stmts = []
        self.skip_newlines()

        while self.peek().type != TokenType.EOF:
            if self.peek().type == TokenType.NEWLINE:
                self.skip_newlines()
                continue

            if self.peek().type in (TokenType.ELSE, TokenType.EOF):
                break
                
            if self.peek().indent <= base_indent:
                break

            start_pos = self.pos
            stmt = self._parse_statement()
            if stmt:
                stmts.append(stmt)
            if self.pos == start_pos:
                break

        return stmts

    def _parse_identifier_statement(self) -> Optional[ASTNode]:
        name = self.advance().value

        # Member access: ta.sma(...), strategy.long, input.int(...)
        if self.peek().type == TokenType.DOT:
            self.advance()
            member = self.advance().value
            full_name = f"{name}.{member}"

            if self.peek().type == TokenType.LPAREN:
                args, kwargs = self._parse_call_args()

                if name == 'input':
                    # Treat as FunctionCall so executor can access both args and kwargs
                    return FunctionCall(full_name, args, kwargs)

                if name == 'strategy':
                    return StrategyCall(member, args, kwargs)

                return FunctionCall(full_name, args, kwargs)

            # strategy.long, strategy.short constants
            return Identifier(full_name)

        # Assignment: x = expr
        if self.peek().type == TokenType.ASSIGN:
            self.advance()
            value = self._parse_expression()
            return Assignment(name, value)

        # Reassignment: x := expr
        if self.peek().type == TokenType.WALRUS:
            self.advance()
            value = self._parse_expression()
            return Assignment(name, value, is_reassign=True)

        # Function call: func(...) or Function Definition: func(a, b) => ...
        if self.peek().type == TokenType.LPAREN:
            args, kwargs = self._parse_call_args()
            if self.peek().type == TokenType.ARROW:
                self.advance()  # consume =>
                params = [a.name for a in args if isinstance(a, Identifier)]
                base_indent = self.tokens[self.pos - 1].indent if self.pos > 0 else 0
                body = self._parse_block(base_indent)
                return FunctionDef(name, params, body)
            return FunctionCall(name, args, kwargs)

        # Expression statement — just skip
        self._skip_to_newline()
        return None

    def _parse_call_args(self) -> Tuple[List[ASTNode], Dict[str, ASTNode]]:
        self.expect(TokenType.LPAREN)
        args = []
        kwargs = {}

        while self.peek().type != TokenType.RPAREN and self.peek().type != TokenType.EOF:
            self.skip_newlines()
            if self.peek().type == TokenType.RPAREN:
                break

            # Check for keyword argument: name = value
            saved_pos = self.pos
            if self.peek().type == TokenType.IDENTIFIER:
                name_tok = self.advance()
                if self.peek().type == TokenType.ASSIGN:
                    self.advance()
                    value = self._parse_expression()
                    kwargs[name_tok.value] = value
                else:
                    self.pos = saved_pos
                    args.append(self._parse_expression())
            else:
                args.append(self._parse_expression())

            if self.peek().type == TokenType.COMMA:
                self.advance()

        self.expect(TokenType.RPAREN)
        return args, kwargs

    # =========================================================================
    # Expression Parser (Pratt-style precedence climbing)
    # =========================================================================

    def _parse_expression(self) -> ASTNode:
        return self._parse_ternary()

    def _parse_ternary(self) -> ASTNode:
        expr = self._parse_or()
        if self.peek().type == TokenType.QUESTION:
            self.advance()
            true_val = self._parse_expression()
            self.expect(TokenType.COLON)
            false_val = self._parse_expression()
            return TernaryOp(expr, true_val, false_val)
        return expr

    def _parse_or(self) -> ASTNode:
        left = self._parse_and()
        while self.peek().type == TokenType.OR:
            self.advance()
            right = self._parse_and()
            left = BinaryOp('or', left, right)
        return left

    def _parse_and(self) -> ASTNode:
        left = self._parse_comparison()
        while self.peek().type == TokenType.AND:
            self.advance()
            right = self._parse_comparison()
            left = BinaryOp('and', left, right)
        return left

    def _parse_comparison(self) -> ASTNode:
        left = self._parse_addition()
        comp_tokens = {TokenType.EQ: '==', TokenType.NEQ: '!=', TokenType.LT: '<',
                       TokenType.GT: '>', TokenType.LTE: '<=', TokenType.GTE: '>='}
        while self.peek().type in comp_tokens:
            op = comp_tokens[self.advance().type]
            right = self._parse_addition()
            left = BinaryOp(op, left, right)
        return left

    def _parse_addition(self) -> ASTNode:
        left = self._parse_multiplication()
        while self.peek().type in (TokenType.PLUS, TokenType.MINUS):
            op = '+' if self.advance().type == TokenType.PLUS else '-'
            right = self._parse_multiplication()
            left = BinaryOp(op, left, right)
        return left

    def _parse_multiplication(self) -> ASTNode:
        left = self._parse_unary()
        while self.peek().type in (TokenType.STAR, TokenType.SLASH, TokenType.PERCENT):
            tok = self.advance()
            op = {TokenType.STAR: '*', TokenType.SLASH: '/', TokenType.PERCENT: '%'}[tok.type]
            right = self._parse_unary()
            left = BinaryOp(op, left, right)
        return left

    def _parse_unary(self) -> ASTNode:
        if self.peek().type == TokenType.MINUS:
            self.advance()
            return UnaryOp('-', self._parse_unary())
        if self.peek().type == TokenType.NOT:
            self.advance()
            return UnaryOp('not', self._parse_unary())
        return self._parse_primary()

    def _parse_primary(self) -> ASTNode:
        tok = self.peek()

        if tok.type == TokenType.NUMBER:
            self.advance()
            node = NumberLiteral(tok.value)
        elif tok.type == TokenType.STRING:
            self.advance()
            node = StringLiteral(tok.value)
        elif tok.type in (TokenType.TRUE, TokenType.FALSE):
            self.advance()
            node = BoolLiteral(tok.type == TokenType.TRUE)
        elif tok.type == TokenType.NA:
            self.advance()
            node = NALiteral()
        elif tok.type == TokenType.LPAREN:
            self.advance()
            node = self._parse_expression()
            self.expect(TokenType.RPAREN)
        elif tok.type == TokenType.IDENTIFIER:
            self.advance()
            name = tok.value

            # Member access chain: ta.sma, math.abs, etc.
            while self.peek().type == TokenType.DOT:
                self.advance()
                member = self.advance().value
                name = f"{name}.{member}"

            # Function call
            if self.peek().type == TokenType.LPAREN:
                args, kwargs = self._parse_call_args()
                node = FunctionCall(name, args, kwargs)
            else:
                node = Identifier(name)
        elif tok.type == TokenType.STRATEGY:
            self.advance()
            if self.peek().type == TokenType.DOT:
                self.advance()
                member = self.advance().value
                name = f"strategy.{member}"
                if self.peek().type == TokenType.LPAREN:
                    args, kwargs = self._parse_call_args()
                    node = FunctionCall(name, args, kwargs)
                else:
                    node = Identifier(name)
            else:
                node = Identifier("strategy")
        else:
            # Unknown token — return a placeholder
            self.advance()
            node = NALiteral()

        # History reference: series[n]
        if self.peek().type == TokenType.LBRACKET:
            self.advance()
            offset = self._parse_expression()
            self.expect(TokenType.RBRACKET)
            node = HistoryRef(node, offset)

        return node


def parse_pine_script(source: str) -> Tuple[List[ASTNode], Dict[str, Any]]:
    """
    Parse a Pine Script source and return (AST, strategy_config).
    """
    tokens = tokenize(source)
    parser = PineParser(tokens)
    ast = parser.parse()
    return ast, parser.strategy_config
