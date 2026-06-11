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
    """Example pairs come from 'N -> SYMBOLS' lines; the query is the number
    on the final line without '->' ('Now, write the number 38 in ...')."""
    pairs, query = [], None
    for ln in lines_of(prompt):
        m = re.match(r"(\d+)\s*(?:->|=>|→)\s*(\S+)\s*$", ln)
        if m:
            pairs.append((int(m.group(1)), m.group(2)))
            continue
        nums = re.findall(r"\b\d+\b", ln)
        if nums:
            query = int(nums[-1])
    return pairs, query


def solve(prompt: str):
    pairs, query = _extract_examples(prompt)
    if not pairs or query is None:
        return None
    # hypothesis 1: roman numerals with substituted symbols
    table = _learn_subst([(to_roman(n), s) for n, s in pairs])
    if table is not None:
        # identity substitution => plain roman; covers query chars the
        # examples never showed (e.g. examples up to L, query needs C)
        if all(k == v for k, v in table.items()):
            return to_roman(query)
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
