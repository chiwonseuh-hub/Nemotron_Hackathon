"""Solver for the 'bits' puzzle type: infer an 8-bit (or 12/16-bit for OOD)
transformation rule from example (input -> output) pairs and apply it to the
query. Rule space: compositions of shift / rotate / XOR / AND / OR / NOT /
add-const plus SHA-style maj / ch over shifted/rotated copies of the input.

The candidate library is also reused by generators/gen_bits.py so that every
generated puzzle is solvable by this solver.
"""
from itertools import product

from .common import extract_binary_pairs


def _mask(w):
    return (1 << w) - 1


def rotl(x, k, w):
    k %= w
    return ((x << k) | (x >> (w - k))) & _mask(w)


def rotr(x, k, w):
    return rotl(x, w - (k % w), w)


def maj(a, b, c):
    return (a & b) ^ (a & c) ^ (b & c)


def ch(a, b, c):
    return (a & b) ^ (~a & c)


def _unaries(w):
    """Small library of unary ops used as building blocks. Returns list of
    (name, fn) where fn: int -> int."""
    m = _mask(w)
    ops = [("id", lambda x: x), ("not", lambda x: ~x & m)]
    for k in range(1, w):
        ops.append((f"rotl{k}", lambda x, k=k: rotl(x, k, w)))
        ops.append((f"rotr{k}", lambda x, k=k: rotr(x, k, w)))
        ops.append((f"shl{k}", lambda x, k=k: (x << k) & m))
        ops.append((f"shr{k}", lambda x, k=k: x >> k))
    return ops


def candidate_functions(w, pairs):
    """Yield (name, fn) candidates. Parametric constants (xor/and/or/add) are
    solved from the first example pair, then everything is verified by the
    caller against all pairs."""
    m = _mask(w)
    uns = _unaries(w)
    x0, y0 = pairs[0]

    # 1) chains of up to 2 unaries
    for n1, f1 in uns:
        yield (n1, f1)
    for (n1, f1), (n2, f2) in product(uns, repeat=2):
        yield (f"{n2}({n1})", lambda x, f1=f1, f2=f2: f2(f1(x)))

    # 2) const ops, constant solved from first pair (possibly after a unary)
    for n1, f1 in uns:
        fx0 = f1(x0)
        c_xor = fx0 ^ y0
        yield (f"xor{c_xor:0{w}b}({n1})", lambda x, f1=f1, c=c_xor: f1(x) ^ c)
        c_add = (y0 - fx0) & m
        yield (f"add{c_add}({n1})", lambda x, f1=f1, c=c_add, m=m: (f1(x) + c) & m)
        for c_and in {y0 | (~fx0 & m), y0}:
            yield (f"and{c_and:0{w}b}({n1})", lambda x, f1=f1, c=c_and: f1(x) & c)
        for c_or in {y0 & (~fx0 & m), y0}:
            yield (f"or{c_or:0{w}b}({n1})", lambda x, f1=f1, c=c_or: f1(x) | c)

    # 3) binary combos of two unary copies
    bin_ops = [("xor", lambda a, b: a ^ b), ("and", lambda a, b: a & b),
               ("or", lambda a, b: a | b)]
    small = [u for u in uns if not u[0].startswith("shl")]
    for (n1, f1), (n2, f2) in product(small, repeat=2):
        for on, op in bin_ops:
            yield (f"{on}({n1},{n2})",
                   lambda x, f1=f1, f2=f2, op=op: op(f1(x), f2(x)))

    # 4) maj / ch over three unary copies (SHA-style sigma functions)
    rots = [u for u in small if u[0].startswith(("rot", "shr", "id"))]
    tern = [("maj", maj), ("ch", lambda a, b, c, m=m: ch(a, b, c) & m)]
    for (n1, f1), (n2, f2), (n3, f3) in product(rots, repeat=3):
        for on, op in tern:
            yield (f"{on}({n1},{n2},{n3})",
                   lambda x, f1=f1, f2=f2, f3=f3, op=op: op(f1(x), f2(x), f3(x)))


def fit_rule(pairs, w):
    """Return (name, fn) of the first candidate consistent with all pairs."""
    for name, fn in candidate_functions(w, pairs):
        if all(fn(x) == y for x, y in pairs):
            return name, fn
    return None, None


def solve(prompt: str):
    str_pairs, query = extract_binary_pairs(prompt)
    if not str_pairs or query is None:
        return None
    w = len(str_pairs[0][0])
    if any(len(a) != w or len(b) != w for a, b in str_pairs) or len(query) != w:
        return None
    pairs = [(int(a, 2), int(b, 2)) for a, b in str_pairs]
    name, fn = fit_rule(pairs, w)
    if fn is None:
        return None
    return format(fn(int(query, 2)), f"0{w}b")
