"""Solver for the 'bits' puzzle type: infer a w-bit (train: 8) transformation
rule from example (input -> output) pairs and apply it to the query.

Reverse-engineered rule family (train.csv): boolean/arithmetic expression
trees over shifted/rotated/negated copies of the input, e.g.
  rotr2(x),  xor(shl2(x), shr5(x)),  maj(shr1(x), rotr2(x), shl2(x)),
  or(xor(rotl1(x), not(shl3(x))), shr3(x))

Search cascade (cheap & high-precision first); within a stage every fitting
rule votes on the query output and the majority wins:
  1. unary chain-2 + const-op-after-unary (pure python)
  2. binary ops over chain-2 pairs (numpy, meet-in-middle for XOR)
  3. 2-level xor-top trees over single/not operands
  4. maj / ch over chain-2 triples
  5. 3-leaf left-assoc trees with ops {xor,and,or,add,sub} over single/not
     operands (covers the or(xnor(..),..) family)

~4% of train instances use even deeper trees and stay unsolved (None).
"""
from collections import Counter
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
_OPERAND_CACHE = {}


def _chains2(w):
    """Distinct unary chain-2 functions as full lookup tables."""
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


def _operands(w):
    """Single unaries + not-wrapped unaries (~58 for w=8)."""
    if w not in _OPERAND_CACHE:
        m = _mask(w)
        seen = {}
        for n, f in _unaries(w):
            seen.setdefault(tuple(f(x) for x in range(1 << w)), n)
            seen.setdefault(tuple(~f(x) & m for x in range(1 << w)), f"not({n})")
        _OPERAND_CACHE[w] = [(name, np.array(tab, dtype=np.int64))
                             for tab, name in seen.items()]
    return _OPERAND_CACHE[w]


def candidate_functions(w, pairs):
    """Stage-1 candidates: unary chains and const ops (constants solved from
    the first pair). Reused by the synthetic generator."""
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


def fit_rule(pairs, w):
    """First stage-1 rule fitting all pairs (generator helper)."""
    for name, fn in candidate_functions(w, pairs):
        if all(fn(x) == y for x, y in pairs):
            return name, fn
    return None, None


def _stage1(pairs, w, qx, A=None, Aq=None, ys=None):
    votes = Counter()
    for _, fn in candidate_functions(w, pairs):
        if all(fn(x) == y for x, y in pairs):
            votes[fn(qx)] += 1
    return votes


def _stage2_binary(pairs, w, qx, A, Aq, ys):
    m = _mask(w)
    votes = Counter()
    n = A.shape[0]
    sig = {}
    for i in range(n):
        sig.setdefault(A[i].tobytes(), []).append(i)
    for i in range(n):
        tgt = (ys ^ A[i]).tobytes()
        for j in sig.get(tgt, ()):
            votes[int(Aq[i] ^ Aq[j])] += 1
    ops = [(np.bitwise_and, lambda a, b: a & b),
           (np.bitwise_or, lambda a, b: a | b),
           (lambda a, b: (a + b) & m, lambda a, b: (a + b) & m),
           (lambda a, b: (a - b) & m, lambda a, b: (a - b) & m)]
    for vop, sop in ops:
        for i in range(n):
            hit = (vop(A[i], A) == ys).all(axis=1)
            for j in np.nonzero(hit)[0]:
                votes[int(sop(int(Aq[i]), int(Aq[j])))] += 1
    return votes


def _stage3_twolevel(pairs, w, qx, A, Aq, ys, B_=None, Bq=None):
    """xor(g(x), op2(h1(x), h2(x))) over the small operand set."""
    chains = _operands(w)
    xs = np.array([p[0] for p in pairs])
    B_ = np.stack([tab[xs] for _, tab in chains])
    Bq = np.array([int(tab[qx]) for _, tab in chains])
    n = len(chains)
    votes = Counter()
    sig = {}
    for j in range(n):
        for k in range(j, n):
            for op in (np.bitwise_xor, np.bitwise_and, np.bitwise_or):
                v = op(B_[j], B_[k])
                sig.setdefault(v.tobytes(), []).append(int(op(Bq[j:j + 1], Bq[k:k + 1])[0]))
    for i in range(n):
        tgt = (ys ^ B_[i]).tobytes()
        for hq in sig.get(tgt, ()):
            votes[int(Bq[i]) ^ hq] += 1
    return votes


def _stage4_ternary(pairs, w, qx, A, Aq, ys):
    """maj/ch over chain-2 triples (numpy-pruned)."""
    m = _mask(w)
    n = A.shape[0]
    votes = Counter()
    for i in range(n):
        a, aq = A[i], int(Aq[i])
        b_ok = np.nonzero(((A & a) == (ys & a)).all(axis=1))[0]
        c_ok = np.nonzero(((A & ~a & m) == (ys & ~a & m)).all(axis=1))[0]
        for j in b_ok:
            for k in c_ok:
                votes[(int(Aq[j]) & aq) | (int(Aq[k]) & ~aq & m)] += 1
    if votes:
        return votes
    for i in range(n):
        for j in range(i, n):
            ab = A[i] & A[j]
            mask_ij = A[i] ^ A[j]
            if ((ys ^ ab) & ~mask_ij & m).any():
                continue
            need = (ys ^ ab) & mask_ij
            hit = np.nonzero(((A & mask_ij) == need).all(axis=1))[0]
            abq = int(Aq[i]) & int(Aq[j])
            mq = int(Aq[i]) ^ int(Aq[j])
            for k in hit:
                votes[abq ^ (mq & int(Aq[k]))] += 1
    return votes


def _stage5_threeleaf(pairs, w, qx, A, Aq, ys):
    """op2(op1(l1,l2), l3) over single/not operands, ops incl add/sub."""
    m = _mask(w)
    chains = _operands(w)
    xs = np.array([p[0] for p in pairs])
    B_ = np.stack([tab[xs] for _, tab in chains])
    Bq = np.array([int(tab[qx]) for _, tab in chains])
    n = len(chains)
    nex = len(pairs)
    votes = Counter()
    ops = [(lambda a, b: (a + b) & m), (lambda a, b: (a - b) & m),
           np.bitwise_xor, np.bitwise_and, np.bitwise_or]
    for op1 in ops:
        for i in range(n):
            mid = op1(B_[i], B_)          # (n, nex)
            midq = op1(int(Bq[i]), Bq)    # (n,)
            for op2 in ops:
                for k in range(n):
                    hit = (op2(mid, B_[k]) == ys).all(axis=1)
                    for j in np.nonzero(hit)[0]:
                        votes[int(op2(int(midq[j]), int(Bq[k])))] += 1
    return votes


def solve(prompt: str):
    str_pairs, query = extract_binary_pairs(prompt)
    if not str_pairs or query is None:
        return None
    w = len(str_pairs[0][0])
    if any(len(a) != w or len(b) != w for a, b in str_pairs) or len(query) != w:
        return None
    if w > 16:
        return None
    pairs = [(int(a, 2), int(b, 2)) for a, b in str_pairs]
    qx = int(query, 2)
    chains = _chains2(w)
    xs = np.array([p[0] for p in pairs])
    ys = np.array([p[1] for p in pairs])
    A = np.stack([tab[xs] for _, tab in chains])
    Aq = np.array([int(tab[qx]) for _, tab in chains])
    for stage in (_stage1, _stage2_binary, _stage3_twolevel,
                  _stage4_ternary, _stage5_threeleaf):
        votes = stage(pairs, w, qx, A, Aq, ys)
        if votes:
            return format(votes.most_common(1)[0][0], f"0{w}b")
    return None
