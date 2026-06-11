"""Solver for 'equations' (reverse-engineered from train.csv, partial).

Known generator family: a plain equation (e.g. "44-43 = 1") is encoded by
substituting characters injectively (digits usually keep identity; operators
may be remapped to symbols) and — in many instances — REVERSING each side's
string. Both forward and reversed instances exist, e.g.
  reversed: plain 25-96 = -71   ->  "69/52 = 17/"   ('/' encodes '-')
  forward:  37-29 = 8 ; 21'16 = 38  (plain digits, "'" encodes a+b+1)

Observed op pool so far: +, -, *, abs(a-b), a+b+1, integer //, %, digit
concatenation. NOTE: ~70% of train instances follow some further variant
that is still uncracked (solver returns None for those); see experiments.md.

Search: positional domains + backtracking over symbol -> (digit|op)
assignments with identity-first value ordering, fast path that pins all
canonical chars (digits and + - * / %) to their identity meaning.
"""
import re

from .common import lines_of

OPS = {
    "-": lambda a, b: a - b,
    "+": lambda a, b: a + b,
    "*": lambda a, b: a * b,
    "abs-": lambda a, b: abs(a - b),
    "//": lambda a, b: a // b if b else None,
    "%": lambda a, b: a % b if b else None,
    "+1": lambda a, b: a + b + 1,
    "cat": lambda a, b: int(str(a) + str(b)) if a >= 0 <= b else None,
}
_OP_ORDER = ["-", "+", "*", "abs-", "//", "%", "+1", "cat"]
_DIGITS = "0123456789"
_CANON_OPS = {"+": "+", "-": "-", "*": "*", "/": "//", "%": "%"}


def _extract(prompt):
    examples, query = [], None
    for ln in lines_of(prompt):
        if ln.lower().startswith("now"):
            m = re.search(r":\s*(.+?)\s*$", ln)
            if m:
                query = m.group(1)
            continue
        if ln.endswith(":"):
            continue
        parts = re.split(r"\s=\s", ln)
        if len(parts) == 2:
            examples.append((parts[0].strip(), parts[1].strip()))
    return examples, query


def solve(prompt: str, node_budget=3_000_000):
    examples, query = _extract(prompt)
    if not examples or query is None:
        return None
    for reverse in (False, True):
        out = _search(examples, query, reverse=reverse, fixed_identity=True,
                      node_budget=node_budget // 10)
        if out is not None:
            return out
    for reverse in (False, True):
        out = _search(examples, query, reverse=reverse, fixed_identity=False,
                      node_budget=node_budget)
        if out is not None:
            return out
    return None


def _search(examples, query, reverse, fixed_identity, node_budget):
    if reverse:
        examples = [(l[::-1], r[::-1]) for l, r in examples]
        query = query[::-1]
    syms = sorted({c for lhs, rhs in examples for c in lhs + rhs} | set(query))

    def decode_num(s, table):
        """Number with optional minus sign (string-leading in plain
        orientation => trailing in reversed display)."""
        neg = False
        if len(s) > 1 and table.get(s[-1 if reverse else 0]) == "-":
            neg = True
            s = s[:-1] if reverse else s[1:]
        if not s:
            return None
        digs = [table.get(c) for c in s]
        if any(d is None or d not in _DIGITS for d in digs):
            return None
        v = int("".join(digs))
        return -v if neg else v

    def eval_lhs(s, table):
        ops_pos = [i for i, c in enumerate(s) if table.get(c) in OPS]
        if len(ops_pos) != 1:
            return None
        i = ops_pos[0]
        if i == 0 or i == len(s) - 1:
            return None
        a = decode_num(s[:i], table)
        b = decode_num(s[i + 1:], table)
        if a is None or b is None:
            return None
        try:
            return OPS[table[s[i]]](a, b)
        except (ZeroDivisionError, ValueError):
            return None

    pre_table, pre_used = {}, set()
    if fixed_identity:
        for c in syms:
            if c in _DIGITS:
                pre_table[c] = c
                pre_used.add(c)
            elif c in _CANON_OPS:
                pre_table[c] = _CANON_OPS[c]
                pre_used.add(_CANON_OPS[c])

    ex_syms = [(frozenset(lhs + rhs), lhs, rhs) for lhs, rhs in examples]

    # positional restriction: a symbol at a position where only a digit is
    # legal can't be an operator (number edges; rhs interior)
    digit_only = set()
    for lhs, rhs in examples + [(query, "")]:
        if lhs:
            digit_only.add(lhs[0])
            digit_only.add(lhs[-1])
        digit_only.update(rhs[:-1] if reverse else rhs[1:])

    def values_for(c):
        vals = []
        if c in _DIGITS:
            vals.append(c)
        vals.extend(d for d in _DIGITS if d != c)
        if c not in digit_only:
            vals.extend(_OP_ORDER)
        return vals

    def consistent(table, assigned):
        for es, lhs, rhs in ex_syms:
            if not es <= assigned:
                continue
            lv = eval_lhs(lhs, table)
            rv = decode_num(rhs, table)
            if lv is None or rv is None or lv != rv:
                return False
        return True

    order, seen = [], set(pre_table)
    for lhs, rhs in sorted(examples, key=lambda e: len(e[0]) + len(e[1])):
        for c in lhs + rhs:
            if c not in seen:
                seen.add(c)
                order.append(c)
    for c in syms:
        if c not in seen:
            seen.add(c)
            order.append(c)

    budget = [node_budget]
    solutions = []

    def backtrack(i, table, used, assigned):
        if budget[0] <= 0 or solutions:
            return
        if i == len(order):
            val = eval_lhs(query, table)
            if val is None:
                return
            inv = {v: k for k, v in table.items()}
            s = str(val)
            neg = s.startswith("-")
            if neg:
                s = s[1:]
            if reverse:
                s = s[::-1]
            out = []
            for c in s:
                if c not in inv:
                    return
                out.append(inv[c])
            enc = "".join(out)
            if neg:
                sign = inv.get("-", "-")
                enc = enc + sign if reverse else sign + enc
            solutions.append(enc)
            return
        c = order[i]
        for v in values_for(c):
            if v in used:
                continue
            budget[0] -= 1
            if budget[0] <= 0:
                return
            table[c] = v
            assigned.add(c)
            if consistent(table, assigned):
                used.add(v)
                backtrack(i + 1, table, used, assigned)
                used.discard(v)
            assigned.discard(c)
            del table[c]

    if not consistent(pre_table, set(pre_table)):
        return None
    backtrack(0, dict(pre_table), set(pre_used), set(pre_table))
    return solutions[0] if solutions else None
