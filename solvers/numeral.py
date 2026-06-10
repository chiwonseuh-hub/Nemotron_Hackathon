"""Solver for 'numeral': numbers rendered in a Roman-numeral-like system with
custom symbols. Strategy:
  1. Render each example number in standard Roman numerals; if the rendered
     string aligns 1:1 with the observed string for every example, learn the
     symbol substitution and apply it to the query.
  2. Fallback: base-b digit representation (b = 2..16) with custom digit
     symbols, learned the same way.
"""
import re

from .common import lines_of

_ROMAN = (
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"),
    (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"),
    (5, "V"), (4, "IV"), (1, "I"),
)


def to_roman(n: int) -> str:
    out = []
    for v, s in _ROMAN:
        while n >= v:
            out.append(s)
            n -= v
    return "".join(out)


def _to_base(n: int, b: int) -> str:
    if n == 0:
        return "0"
    digs = []
    while n:
        digs.append("0123456789abcdef"[n % b])
        n //= b
    return "".join(reversed(digs))


def _learn_subst(canon_obs_pairs):
    table = {}
    for canon, obs in canon_obs_pairs:
        if len(canon) != len(obs):
            return None
        for a, b in zip(canon, obs):
            if table.get(a, b) != b:
                return None
            table[a] = b
    return table


def _extract_examples(prompt):
    """(int, representation) pairs: lines containing an integer and a
    non-numeric token; the query is the last line with only an integer."""
    pairs, query = [], None
    for ln in lines_of(prompt):
        nums = re.findall(r"\b\d+\b", ln)
        toks = re.findall(r"[^\W\d_]{1,40}", ln, flags=re.UNICODE)
        # representation token: longest alphabetic-ish token on the line
        rep = max(toks, key=len) if toks else None
        if len(nums) == 1 and rep and len(rep) >= 1:
            pairs.append((int(nums[0]), rep))
        elif len(nums) == 1:
            query = int(nums[0])
    return pairs, query


def solve(prompt: str):
    pairs, query = _extract_examples(prompt)
    if not pairs or query is None:
        return None
    # hypothesis 1: roman numerals with substituted symbols
    table = _learn_subst([(to_roman(n), s) for n, s in pairs])
    if table is not None:
        canon = to_roman(query)
        if all(c in table for c in canon):
            return "".join(table[c] for c in canon)
    # hypothesis 2: base-b digits with substituted symbols
    for b in range(2, 17):
        table = _learn_subst([(_to_base(n, b), s) for n, s in pairs])
        if table is not None:
            canon = _to_base(query, b)
            if all(c in table for c in canon):
                return "".join(table[c] for c in canon)
    return None
