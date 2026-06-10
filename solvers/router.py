"""Puzzle-type detection + dispatch.

Keyword heuristics below are provisional — tune them against real train.csv
prompts via scripts/validate_solvers.py (it reports routing coverage).
"""
import re

from . import bits, cipher, equations, gravity, numeral, units

_BIN_RE = re.compile(r"\b[01]{6,32}\b")

_KEYWORDS = {
    "gravity": ["gravity", "fall", "drop", "planet", "meters", "height", "seconds"],
    "units": ["convert", "unit", "glorbs", "measurement", "equals", "ratio"],
    "cipher": ["cipher", "decode", "decrypt", "encrypt", "encoded", "translat"],
    "numeral": ["numeral", "roman", "notation", "number system", "symbols represent"],
    "equations": ["equation", "expression", "evaluate", "solve", "operator"],
}


def detect_type(prompt: str) -> str:
    p = prompt.lower()
    if len(_BIN_RE.findall(prompt)) >= 3:
        return "bits"
    scores = {t: sum(k in p for k in kws) for t, kws in _KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unknown"


_SOLVERS = {
    "bits": bits.solve,
    "gravity": gravity.solve,
    "units": units.solve,
    "cipher": cipher.solve,
    "numeral": numeral.solve,
    "equations": equations.solve,
}


def solve(prompt: str, ptype: str = None):
    """Returns (ptype, answer_or_None). If ptype not given, auto-detect; on
    'unknown', tries every solver and returns the first non-None answer."""
    if ptype is None:
        ptype = detect_type(prompt)
    if ptype in _SOLVERS:
        return ptype, _SOLVERS[ptype](prompt)
    for t, fn in _SOLVERS.items():
        try:
            ans = fn(prompt)
        except Exception:
            ans = None
        if ans is not None:
            return t, ans
    return "unknown", None
