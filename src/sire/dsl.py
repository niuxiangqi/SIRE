"""Typed symbolic formula DSL.

Exactly six leaves (D, A, G, U, B, R) and six operators (Neg, Abs, Add, Sub,
Mul, SafeDiv). Formulas are immutable nested tuples:

    ('LEAF', 'A')
    ('Neg', child)
    ('Abs', child)
    ('Add', left, right)      # and Sub / Mul / SafeDiv

Leaves are per-tensor normalised by a fixed positive RMS before evaluation, so
every subexpression is dimensionless.
"""
from __future__ import annotations

import ast
import hashlib
import math

LEAVES = ("D", "A", "G", "U", "B", "R")
UNARY = ("Neg", "Abs")
BINARY = ("Add", "Sub", "Mul", "SafeDiv")
OPS = UNARY + BINARY
MAX_DEPTH = 3
MAX_OPS = 4

_COMMUTATIVE = {"Add", "Mul"}


class FormulaError(ValueError):
    pass


# ----------------------------------------------------------------------------
# Construction / validation
# ----------------------------------------------------------------------------
def leaf(name: str):
    if name not in LEAVES:
        raise FormulaError(f"unknown leaf {name}")
    return ("LEAF", name)


def unary(op: str, child):
    if op not in UNARY:
        raise FormulaError(op)
    return (op, child)


def binary(op: str, left, right):
    if op not in BINARY:
        raise FormulaError(op)
    return (op, left, right)


def is_leaf(e) -> bool:
    return isinstance(e, tuple) and len(e) == 2 and e[0] == "LEAF"


def children(e):
    if is_leaf(e):
        return ()
    if e[0] in UNARY:
        return (e[1],)
    return (e[1], e[2])


def walk(e):
    yield e
    for c in children(e):
        yield from walk(c)


def leaves_of(e) -> set[str]:
    return {n[1] for n in walk(e) if is_leaf(n)}


def n_ops(e) -> int:
    return sum(1 for n in walk(e) if not is_leaf(n))


def depth(e) -> int:
    if is_leaf(e):
        return 0
    return 1 + max(depth(c) for c in children(e))


def validate(e) -> None:
    if is_leaf(e):
        if e[1] not in LEAVES:
            raise FormulaError(f"bad leaf {e}")
        return
    op = e[0]
    if op not in OPS:
        raise FormulaError(f"bad op {op}")
    if op in UNARY and len(e) != 2:
        raise FormulaError(f"bad arity {e}")
    if op in BINARY and len(e) != 3:
        raise FormulaError(f"bad arity {e}")
    for c in children(e):
        validate(c)
    if depth(e) > MAX_DEPTH:
        raise FormulaError(f"depth {depth(e)} > {MAX_DEPTH}")
    if n_ops(e) > MAX_OPS:
        raise FormulaError(f"ops {n_ops(e)} > {MAX_OPS}")


# ----------------------------------------------------------------------------
# Canonicalisation
# ----------------------------------------------------------------------------
def canonical(e):
    """Fold double negation, order commutative arguments by canonical key."""
    if is_leaf(e):
        return e
    op = e[0]
    if op in UNARY:
        child = canonical(e[1])
        if op == "Neg" and child[0] == "Neg":
            return child[1]
        if op == "Abs" and child[0] == "Abs":
            return child
        return (op, child)
    left = canonical(e[1])
    right = canonical(e[2])
    if op in _COMMUTATIVE:
        if canonical_str(right) < canonical_str(left):
            left, right = right, left
    return (op, left, right)


def canonical_str(e) -> str:
    e = canonical(e)
    if is_leaf(e):
        return e[1]
    if e[0] in UNARY:
        return f"({e[0]} {canonical_str(e[1])})"
    return f"({e[0]} {canonical_str(e[1])} {canonical_str(e[2])})"


def infix(e, top: bool = True) -> str:
    e = canonical(e)
    if is_leaf(e):
        return e[1]
    if e[0] == "Neg":
        return f"(-{infix(e[1], False)})"
    if e[0] == "Abs":
        return f"|{infix(e[1], False)}|"
    sym = {"Add": "+", "Sub": "-", "Mul": "*", "SafeDiv": "/"}[e[0]]
    return f"({infix(e[1], False)} {sym} {infix(e[2], False)})"


def parse(s: str):
    """Safe parser for infix seed strings (no eval, no attribute access)."""
    tree = ast.parse(s, mode="eval")
    def conv(node):
        if isinstance(node, ast.Name):
            return leaf(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return unary("Neg", conv(node.operand))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
            return conv(node.operand)
        if isinstance(node, ast.BinOp):
            l, r = conv(node.left), conv(node.right)
            if isinstance(node.op, ast.Add):
                return binary("Add", l, r)
            if isinstance(node.op, ast.Sub):
                return binary("Sub", l, r)
            if isinstance(node.op, ast.Mult):
                return binary("Mul", l, r)
            if isinstance(node.op, ast.Div):
                return binary("SafeDiv", l, r)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            args = [conv(a) for a in node.args]
            if name in UNARY and len(args) == 1:
                return unary(name, args[0])
            if name in BINARY and len(args) == 2:
                return binary(name, *args)
        raise FormulaError(f"unsupported syntax: {ast.dump(node)}")

    out = canonical(conv(tree.body))
    validate(out)
    return out


def parse_sexpr(s: str):
    """Parse the canonical AST form, e.g. '(Add A (Neg U))'."""
    import re

    tokens = re.findall(r"\(|\)|[A-Za-z_]+", s or "")
    pos = 0

    def atom():
        nonlocal pos
        if pos >= len(tokens):
            raise FormulaError("unexpected end")
        t = tokens[pos]
        pos += 1
        if t == "(":
            op = tokens[pos]
            pos += 1
            args = []
            while pos < len(tokens) and tokens[pos] != ")":
                args.append(atom())
            if pos >= len(tokens) or tokens[pos] != ")":
                raise FormulaError("unbalanced")
            pos += 1
            if op in UNARY and len(args) == 1:
                return (op, args[0])
            if op in BINARY and len(args) == 2:
                return (op, args[0], args[1])
            raise FormulaError(f"bad node {op} arity {len(args)}")
        return leaf(t)

    out = canonical(atom())
    if pos != len(tokens):
        raise FormulaError("trailing tokens")
    validate(out)
    return out


# ----------------------------------------------------------------------------
# Complexity
# ----------------------------------------------------------------------------
def complexity(e) -> int:
    """Unique operator nodes in the expanded DAG, plus one per used G/U leaf
    (the Mean and Mean∘Abs reductions are not free)."""
    e = canonical(e)
    op_nodes = {canonical_str(n) for n in walk(e) if not is_leaf(n)}
    cost = len(op_nodes)
    lv = leaves_of(e)
    cost += int("G" in lv) + int("U" in lv)
    return cost


def formula_id(e) -> str:
    """Stable identifier independent of formatting."""
    return hashlib.sha256(canonical_str(e).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------
# Evaluation over torch tensors
# ----------------------------------------------------------------------------
def evaluate(e, leaves: dict):
    """Evaluate on per-tensor FP32 tensors. leaves: name -> tensor."""
    import torch

    e = canonical(e)
    if is_leaf(e):
        name = e[1]
        if name not in leaves:
            raise KeyError(f"leaf {name} not provided")
        return leaves[name]
    if e[0] == "Neg":
        return -evaluate(e[1], leaves)
    if e[0] == "Abs":
        return torch.abs(evaluate(e[1], leaves))
    a = evaluate(e[1], leaves)
    b = evaluate(e[2], leaves)
    if e[0] == "Add":
        return a + b
    if e[0] == "Sub":
        return a - b
    if e[0] == "Mul":
        return a * b
    if e[0] == "SafeDiv":
        sign = torch.sign(b)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        denom = sign * torch.clamp(torch.abs(b), min=1e-6)
        return a / denom
    raise FormulaError(e[0])


def safe_div_reference(x: float, y: float) -> float:
    s = 1.0 if y >= 0 else -1.0
    return x / (s * max(abs(y), 1e-6))


# ----------------------------------------------------------------------------
# Small random tree generator (search proposals)
# ----------------------------------------------------------------------------
def random_tree(rng, max_ops: int = MAX_OPS, max_depth: int = MAX_DEPTH):
    def gen(d: int, budget: int):
        if d >= max_depth or budget <= 0 or rng.random() < 0.25:
            return leaf(rng.choice(LEAVES))
        op = rng.choice(OPS)
        if op in UNARY:
            return (op, gen(d + 1, budget - 1))
        return (op, gen(d + 1, budget - 1), gen(d + 1, budget - 1))

    for _ in range(50):
        e = canonical(gen(0, max_ops))
        if n_ops(e) == 0:
            continue
        try:
            validate(e)
            return e
        except FormulaError:
            continue
    return canonical(leaf("A"))


def all_subtrees(e):
    yield e
    for c in children(e):
        yield from all_subtrees(c)


def replace_subtree(e, target_str: str, replacement):
    if canonical_str(e) == target_str:
        return replacement
    if is_leaf(e):
        return e
    if e[0] in UNARY:
        return (e[0], replace_subtree(e[1], target_str, replacement))
    return (e[0], replace_subtree(e[1], target_str, replacement),
            replace_subtree(e[2], target_str, replacement))
