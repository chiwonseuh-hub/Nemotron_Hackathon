"""Shared answer extraction / scoring utilities.

Mirrors the Kaggle evaluation as described:
  1. extract \\boxed{...} content (last occurrence, nested braces handled)
  2. fallback: last number in the generation
Match = exact string match OR numeric match within relative tolerance.

NOTE: the hidden metric's exact tolerance is unknown. Defaults below are a
best guess (answers are 2-decimal floats, so abs 0.01 / rel 1e-2). Recalibrate
if the official metric code becomes visible.
"""
import re

_BOXED = "\\boxed{"
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_boxed(text: str):
    """Return content of the LAST \\boxed{...} in text, or None."""
    idx = text.rfind(_BOXED)
    if idx == -1:
        return None
    i = idx + len(_BOXED)
    depth = 1
    out = []
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
        out.append(c)
        i += 1
    return None  # unbalanced


def extract_answer(text: str) -> str:
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed
    nums = _NUM_RE.findall(text)
    return nums[-1] if nums else ""


def _to_float(s):
    try:
        return float(str(s).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None


def normalize(s: str) -> str:
    return " ".join(str(s).strip().split())


def is_correct(pred, gold, rel_tol=1e-2, abs_tol=1e-2) -> bool:
    pred_s, gold_s = normalize(pred), normalize(gold)
    if pred_s == gold_s:
        return True
    if pred_s.lower() == gold_s.lower():
        return True
    p, g = _to_float(pred_s), _to_float(gold_s)
    if p is not None and g is not None:
        if abs(p - g) <= abs_tol:
            return True
        if g != 0 and abs(p - g) / abs(g) <= rel_tol:
            return True
    return False
