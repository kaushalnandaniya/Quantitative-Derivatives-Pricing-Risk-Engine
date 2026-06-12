import pytest
from services.pine_parser import parse_pine_script, FunctionDef
script = """
my_func(src, len) =>
    sum = 0.0
    for i = 1 to len
        sum := sum + src
    sum / len
"""
ast, _ = parse_pine_script(script)
print("AST:", ast)
for n in ast:
    if isinstance(n, FunctionDef):
        print("Function:", n.name, n.params)
