"""Safe answer extraction and comparison.

No generated code is ever executed. Parsing is a bounded recursive-descent
parser over a whitelisted arithmetic AST, returning exact Fractions where
possible. A parse failure is wrong, never an exception that removes an item.
"""
from __future__ import annotations

import ast
import re
from fractions import Fraction

MAX_NODES = 200
MAX_EXPONENT = 100

_BOX_START = re.compile(r"\\boxed\s*\{")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?")


def _balanced_braces(text: str, start: int) -> str | None:
    depth = 1
    i = start
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
        i += 1
    return None


def extract_boxed(text: str) -> str | None:
    matches = list(_BOX_START.finditer(text))
    for m in reversed(matches):
        inner = _balanced_braces(text, m.end())
        if inner is not None and inner.strip():
            return inner.strip()
    return None


def extract_answer(text: str) -> str | None:
    """Return the candidate final-answer string, or None."""
    if not text:
        return None
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed
    if "####" in text:
        tail = text.rsplit("####", 1)[1].strip()
        if tail:
            return tail.splitlines()[0].strip()
    # Conservative fallbacks: explicit final-answer phrases, then last number.
    for pat in (r"(?:final answer|answer is|the answer is)\s*[:：]?\s*([^\n.]+)",
                r"(?:答案是|答案为|所以|因此)\s*[:：]?\s*([^\n。]+)"):
        found = re.findall(pat, text, flags=re.IGNORECASE)
        if found:
            return found[-1].strip()
    nums = _NUM_RE.findall(text)
    return nums[-1] if nums else None


def _strip_math(s: str) -> str:
    s = s.strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "").replace("\\!", "").replace("\\ ", "")
    s = s.replace("$", "").replace("€", "")
    s = s.replace("\\%", "%").replace("％", "%")
    # thousands separators only when they look like 3-digit groups
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    s = s.replace("−", "-").replace("–", "-").replace("×", "*").replace("÷", "/")
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\(?:text|mathrm|mbox|operatorname)\s*\{[^{}]*\}", "", s)
    s = s.replace("\\pi", "pi")
    return s.strip()


def _frac_latex(s: str) -> str:
    """Rewrite \\frac{a}{b} and \\sqrt{a} into parseable text (bounded loops)."""
    for _ in range(20):
        m = re.search(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", s)
        if not m:
            break
        s = s[:m.start()] + f"(({m.group(1)})/({m.group(2)}))" + s[m.end():]
    for _ in range(10):
        m = re.search(r"\\sqrt\s*\{([^{}]*)\}", s)
        if not m:
            break
        s = s[:m.start()] + f"sqrt({m.group(1)})" + s[m.end():]
    return s


def _eval_ast(node) -> Fraction | float:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Fraction(str(node.value)) if isinstance(node.value, float) else Fraction(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _eval_ast(node.operand)
        return v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.BinOp):
        a, b = _eval_ast(node.left), _eval_ast(node.right)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        if isinstance(node.op, ast.Div):
            if b == 0:
                raise ZeroDivisionError
            return a / b
        if isinstance(node.op, ast.Pow):
            if isinstance(b, Fraction) and b.denominator == 1 and abs(int(b)) <= MAX_EXPONENT:
                return a ** int(b)
            raise ValueError("bad exponent")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sqrt":
        if len(node.args) != 1:
            raise ValueError("sqrt arity")
        v = _eval_ast(node.args[0])
        fv = float(v)
        if fv < 0:
            raise ValueError("sqrt negative")
        r = fv ** 0.5
        fr = Fraction(r).limit_denominator(10**6)
        if abs(float(fr) ** 2 - fv) < 1e-9:
            return fr
        return r
    raise ValueError("unsupported node")


def parse_value(s: str):
    """Return Fraction or float, or None if not a single verifiable scalar."""
    if s is None:
        return None
    s = _strip_math(str(s))
    if not s:
        return None
    s = _frac_latex(s)
    s = s.strip().strip(".").strip()
    percent = s.endswith("%")
    if percent:
        s = s[:-1].strip()
    # Drop common trailing units after a number, without touching expressions.
    m = re.fullmatch(r"([^A-Za-z\u4e00-\u9fff]*\d(?:[^A-Za-z\u4e00-\u9fff]*))"
                     r"\s*(?:cm|mm|km|m|kg|g|h|min|s|元|米|厘米|千米|千克|克|小时|分钟|秒|个|岁|度)\^?\d*", s)
    if m:
        s = m.group(1)
    # Reject multi-answer / text soup early.
    if re.search(r"[;；]", s) or len(s) > 120:
        return None
    s = s.replace("^", "**")
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(s, mode="eval")
    except SyntaxError:
        num = _NUM_RE.search(s)
        if num and not re.search(r"[A-Za-z\u4e00-\u9fff]", s.replace("sqrt", "")):
            s = num.group(0)
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    tree = ast.parse(s, mode="eval")
            except SyntaxError:
                return None
        else:
            return None
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        return None
    try:
        v = _eval_ast(tree)
    except Exception:
        return None
    if isinstance(v, Fraction) and v.denominator == 0:
        return None
    try:
        fv = float(v)
    except Exception:
        return None
    if fv != fv or abs(fv) == float("inf"):
        return None
    if percent:
        v = v / 100
    return v


def values_equal(a, b, tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return False
    if isinstance(a, Fraction) and isinstance(b, Fraction):
        return a == b
    try:
        fa, fb = float(a), float(b)
    except Exception:
        return False
    return abs(fa - fb) <= tol * max(1.0, abs(fa), abs(fb))


def is_correct(pred_text: str, gt_answer: str) -> tuple[bool, str | None, str | None]:
    """Return (correct, parsed_pred, parsed_gt)."""
    pred = extract_answer(pred_text)
    pv = parse_value(pred) if pred is not None else None
    gv = parse_value(gt_answer)
    if pv is None or gv is None:
        # A literal string fallback for non-numeric single-token answers.
        if pred is not None and str(pred).strip().lower() == str(gt_answer).strip().lower():
            return True, pred, str(gt_answer)
        return False, pred, None if gv is None else str(gv)
    return values_equal(pv, gv), str(pv), str(gv)
