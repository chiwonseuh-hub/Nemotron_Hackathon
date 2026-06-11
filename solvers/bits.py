"""Solver for the 'bits' puzzle type: infer a w-bit (train: 8) transformation
rule from example (input -> output) pairs and apply it to the query.

Observed rule family (reverse-engineered from train.csv): unary chains
(shift / rotate / NOT) possibly combined by a binary (XOR / AND / OR / ADD)
or ternary (maj / ch) bitwise op, plus constant XOR/ADD/AND/OR variants.

Search cascade (cheap -> expensive):
  1. unary chain-2 + const-op-after-unary candidates (pure python)
  2. binary ops over chain-2 pairs (numpy, meet-in-middle for XOR)
  3. maj / ch over unary/not-chain triples (numpy)

Ambiguity guard: within the stage that first fits, ALL fitting rules must
agree on the query, otherwise None is returned (important when this solver is
used as the oracle for synthetic data generation).
"""
from itertools import product

import numpy as np

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
    m = _mask(w)
    ops = [("id", lambda x: x), ("not", lambda x: ~x & m)]
    for k in range(1, w):
        ops.append((f"rotl{k}", lambda x, k=k: rotl(x, k, w)))
        ops.append((f"rotr{k}", lambda x, k=k: rotr(x, k, w)))
        ops.append((f"shl{k}", lambda x, k=k: (x << k) & m))
        ops.append((f"shr{k}", lambda x, k=k: x >> k))
    return ops


_CHAIN_CACHE = {}


def _chains2(w):
    """Distinct unary chain-2 functions as full lookup tables (len 2^w)."""
    if w not in _CHAIN_CACHE:
        uns = _unaries(w)
        seen = {}
        for n1, f1 in uns:
            for n2, f2 in uns:
                tab = tuple(f2(f1(x)) for x in range(1 << w))
                name = n2 if n1 == "id" else f"{n2}({n1})"
                if tab not in seen:
                    seen[tab] = name
        _CHAIN_CACHE[w] = [(name, np.array(tab, dtype=np.int64))
                           for tab, name in seen.items()]
    return _CHAIN_CACHE[w]


def candidate_functions(w, pairs):
    """Stage-1 candidates: unary chains and const ops (constants solved from
    the first pair). Kept for speed and reused by the generator."""
    m = _mask(w)
    uns = _unaries(w)
    x0, y0 = pairs[0]
    for n1, f1 in uns:
        yield (n1, f1)
    for (n1, f1), (n2, f2) in product(uns, repeat=2):
        yield (f"{n2}({n1})", lambda x, f1=f1, f2=f2: f2(f1(x)))
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


def _stage1(pairs, w, qx):
    outs = set()
    for _, fn in candidate_functions(w, pairs):
        if all(fn(x) == y for x, y in pairs):
            outs.add(fn(qx))
            if len(outs) > 1:
                return None
    return outs.pop() if outs else None


def _stage2_binary(pairs, w, qx):
    """op(g1(x), g2(x)) for chain-2 g1,g2 and op in xor/and/or/add."""
    m = _mask(w)
    chains = _chains2(w)
    xs = np.array([p[0] for p in pairs])
    ys = np.array([p[1] for p in pairs])
    A = np.stack([tab[xs] for _, tab in chains])          # (n_chains, n_ex)
    Aq = np.array([tab[qx] for _, tab in chains])         # (n_chains,)
    outs = set()

    # xor: meet-in-the-middle on row signatures
    sig = {}
    for i in range(len(chains)):
        sig.setdefault(tuple(A[i]), []).append(i)
    for i in range(len(chains)):
        tgt = tuple(ys ^ A[i])
        for j in sig.get(tgt, ()):
            outs.add(int(Aq[i] ^ Aq[j]))
            if len(outs) > 1:
                return None

    ops = [(np.bitwise_and, lambda a, b: a & b),
           (np.bitwise_or, lambda a, b: a | b),
           (lambda a, b: (a + b) & m, lambda a, b: (a + b) & m)]
    for vop, sop in ops:
        for i in range(len(chains)):
            hit = (vop(A[i], A) == ys).all(axis=1)
            for j in np.nonzero(hit)[0]:
                outs.add(int(sop(int(Aq[i]), int(Aq[j]))))
                if len(outs) > 1:
                    return None
    return outs.pop() if outs else None


def _stage3_ternary(pairs, w, qx):
    """maj/ch over chain-2 triples (numpy-pruned)."""
    m = _mask(w)
    chains = _chains2(w)
    xs = np.array([p[0] for p in pairs])
    ys = np.array([p[1] for p in pairs])
    A = np.stack([tab[xs] for _, tab in chains])
    Aq = np.array([tab[qx] for _, tab in chains])
    n = len(chains)
    outs = set()

    # ch(a,b,c): bits of y where a=1 come from b, where a=0 come from c
    for i in range(n):
        a, aq = A[i], int(Aq[i])
        b_ok = np.nonzero(((A & a) == (ys & a)).all(axis=1))[0]
        c_ok = np.nonzero(((A & ~a & m) == (ys & ~a & m)).all(axis=1))[0]
        if len(b_ok) and len(c_ok):
            bq = {int(Aq[j]) & aq for j in b_ok}
            cq = {int(Aq[k]) & ~aq & m for k in c_ok}
            for vb in bq:
                for vc in cq:
                    outs.add(vb | vc)
                    if len(outs) > 1:
                        return None

    # maj(a,b,c) = (a&b) ^ ((a^b)&c)
    for i in range(n):
        for j in range(i, n):
            ab = A[i] & A[j]
            mask_ij = A[i] ^ A[j]
            if ((ys ^ ab) & ~mask_ij & m).any():
                continue
            need = (ys ^ ab) & mask_ij
            hit = np.nonzero(((A & mask_ij) == need).all(axis=1))[0]
            if len(hit):
                abq = int(Aq[i]) & int(Aq[j])
                mq = int(Aq[i]) ^ int(Aq[j])
                for k in hit:
                    outs.add(abq ^ (mq & int(Aq[k])))
                    if len(outs) > 1:
                        return None
    return outs.pop() if outs else None


def _operands(w):
    """Single unaries + not-wrapped unaries as lookup tables (~58 for w=8)."""
    m = _mask(w)
    seen = {}
    for n, f in _unaries(w):
        for pn, pf in (("", lambda v: v), ("not.", lambda v, m=m: ~v & m)):
            tab = tuple(pf(f(x)) for x in range(1 << w))
            seen.setdefault(tab, f"{pn}{n}")
    return [(name, np.array(tab, dtype=np.int64)) for tab, name in seen.items()]


def _stage4_twolevel(pairs, w, qx):
    """xor(g(x), op2(h1(x), h2(x))) over the small operand set."""
    chains = _operands(w)
    xs = np.array([p[0] for p in pairs])
    ys = np.array([p[1] for p in pairs])
    A = np.stack([tab[xs] for _, tab in chains])
    Aq = np.array([int(tab[qx]) for _, tab in chains])
    n = len(chains)
    outs = set()
    sig = {}
    for j in range(n):
        for k in range(j, n):
            for op in (np.bitwise_xor, np.bitwise_and, np.bitwise_or):
                v = op(A[j], A[k])
                sig.setdefault(v.tobytes(), []).append(int(op(Aq[j:j+1], Aq[k:k+1])[0]))
    for i in range(n):
        tgt = (ys ^ A[i]).tobytes()
        for hq in sig.get(tgt, ()):
            outs.add(int(Aq[i]) ^ hq)
            if len(outs) > 1:
                return None
    return outs.pop() if outs else None


def fit_rule(pairs, w):
    """Back-compat helper for the generator: first stage-1 rule fitting all
    pairs, as (name, fn)."""
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
    if w > 16:
        return None  # chain table would be too large
    pairs = [(int(a, 2), int(b, 2)) for a, b in str_pairs]
    qx = int(query, 2)
    for stage in (_stage1, _stage2_binary, _stage4_twolevel, _stage3_ternary):
        out = stage(pairs, w, qx)
        if out is not None:
            return format(out, f"0{w}b")
    return None
