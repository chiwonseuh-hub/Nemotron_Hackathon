"""Solver for 'units': infer a linear unit-conversion y = r*x (or affine
y = a*x + b as OOD fallback) from example pairs, apply to the query value.
Answer: 2-decimal float string.
"""
from .common import extract_number_pairs

_REL_TOL = 1e-3


def _fit_ratio(pairs):
    rs = [y / x for x, y in pairs if x != 0]
    if not rs:
        return None
    # consensus rather than strict all-agree: robust to spurious number pairs
    # parsed out of template boilerplate
    best, best_n = None, 0
    for r in rs:
        n = sum(1 for v in rs if abs(v - r) <= _REL_TOL * max(abs(r), 1e-9))
        if n > best_n:
            best, best_n = r, n
    return best if best_n >= 2 else None


def _fit_affine(pairs):
    if len(pairs) < 2:
        return None
    (x1, y1), (x2, y2) = pairs[0], pairs[1]
    if x1 == x2:
        return None
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    ok = all(abs(a * x + b - y) <= _REL_TOL * max(abs(y), 1e-9) for x, y in pairs)
    return (a, b) if ok else None


def _ratio_exact(pairs):
    rs = [y / x for x, y in pairs if x != 0]
    if rs and all(abs(v - rs[0]) <= _REL_TOL * max(abs(rs[0]), 1e-9) for v in rs):
        return rs[0]
    return None


def solve(prompt: str):
    pairs, query = extract_number_pairs(prompt)
    if not pairs or query is None:
        return None
    # exact ratio on ALL pairs -> exact affine on ALL pairs -> consensus ratio
    # (consensus last: an affine relation has near-equal ratios that could
    # otherwise win by accident; consensus only exists to survive spurious
    # pairs parsed from boilerplate)
    r = _ratio_exact(pairs)
    if r is not None:
        return f"{r * query:.2f}"
    ab = _fit_affine(pairs)
    if ab is not None:
        a, b = ab
        return f"{a * query + b:.2f}"
    r = _fit_ratio(pairs)
    if r is not None:
        return f"{r * query:.2f}"
    return None
