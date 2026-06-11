"""Solver for 'gravity': examples give (t, d) observations under d = 0.5*g*t^2
with a modified g. Recover g = 2d/t^2 from the examples, verify consistency,
then compute d for the queried t. Answer: 2-decimal float string.

Also tries the swapped column order (d, t) and, as an OOD fallback, the linear
law v = g*t.
"""
import re

from .common import extract_number_pairs, lines_of

_REL_TOL = 1e-3


def _consensus(values, min_count=2):
    """Most-agreed-upon value among candidates (robust to spurious pairs
    parsed from template boilerplate). Returns (value, count) or (None, 0)."""
    best, best_n = None, 0
    for v in values:
        n = sum(1 for u in values if abs(u - v) <= _REL_TOL * max(abs(v), 1e-9))
        if n > best_n:
            best, best_n = v, n
    return (best, best_n) if best_n >= min_count else (None, 0)


_QUERY_RE = re.compile(r"t\s*=\s*(-?\d+(?:\.\d+)?)\s*s")


def _extract_query_t(prompt, pairs):
    """The train query line ('... for t = 4.41s given d = 0.5*g*t^2.') has 3
    numbers, so extract_number_pairs misses it. Take the t= value on the last
    line that is not an example pair."""
    pair_ts = {t for t, _ in pairs} | {d for _, d in pairs}
    for ln in reversed(lines_of(prompt)):
        nums = _QUERY_RE.findall(ln)
        if nums and float(nums[0]) not in pair_ts:
            return float(nums[0])
        if nums:
            return float(nums[0])
    return None


def solve(prompt: str):
    pairs, query_t = extract_number_pairs(prompt)
    if not pairs:
        return None
    qt = _extract_query_t(prompt, pairs)
    if qt is not None:
        query_t = qt
    if query_t is None:
        return None
    orderings = (pairs, [(b, a) for a, b in pairs])
    # quadratic law d = 0.5*g*t^2 first, linear v = g*t as OOD fallback;
    # try both column orders and keep the best-supported fit
    candidates = []
    for ordered in orderings:
        g, n = _consensus([2.0 * d / (t * t) for t, d in ordered if t != 0])
        if g is not None:
            candidates.append((n, f"{0.5 * g * query_t * query_t:.2f}"))
    if not candidates:
        for ordered in orderings:
            g, n = _consensus([d / t for t, d in ordered if t != 0])
            if g is not None:
                candidates.append((n, f"{g * query_t:.2f}"))
    if not candidates:
        return None
    # ties (e.g. a perfectly linear relation fits both column orders) go to
    # the original column order, hence key on count only
    return max(candidates, key=lambda c: c[0])[1]
